"""
=============================================================================
 vjepa_real.py  —  A REAL (small) V-JEPA implemented in PyTorch
=============================================================================
This is NOT a simulation with random vectors. It is a genuine
Joint-Embedding Predictive Architecture that:

  1. Encodes video clips into patch/tubelet token embeddings (context encoder).
  2. Maintains an EMA target encoder (the real anti-collapse mechanism,
     exactly as in V-JEPA / BYOL — NOT orthogonal construction).
  3. Masks a set of spatiotemporal tubelets and PREDICTS their target-encoder
     embeddings from the visible context, using a Transformer predictor.
  4. Trains by minimising the distance between predicted and target
     embeddings IN EMBEDDING SPACE (smooth-L1), with real backprop.
  5. Demonstrably reduces the loss over training (curve is saved),
     and we verify the learned space did NOT collapse (embedding variance
     and rank stay high).

Faithful to the V-JEPA recipe (Bardes et al., 2024, arXiv:2404.08471):
  - feature prediction (not pixel reconstruction)
  - context/target split via masking
  - EMA target encoder, stop-gradient on the target branch
  - predictor conditioned on masked-token positions
Scaled down so it trains on CPU in a couple of minutes on synthetic video
whose latent factors are actually learnable (moving oriented bars).

Author: Marcelo Feliciano Filho Calixto — CRAN / UL & PUCPR
=============================================================================
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

# ─── Dimensions ──────────────────────────────────────────────────────────────
T, H, W   = 8, 32, 32     # frames, height, width of each clip
PT, PH, PW = 2, 8, 8      # tubelet size (temporal, height, width)
GT, GH, GW = T//PT, H//PH, W//PW          # grid: 4 x 4 x 4 = 64 tubelets
N_TOKENS  = GT * GH * GW                    # 64 tokens per clip
TUBELET_DIM = PT * PH * PW                  # 128 raw values per tubelet
EMBED_DIM = 128
DEPTH     = 4
HEADS     = 4
PRED_DIM  = 96
PRED_DEPTH = 3

DEVICE = "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC VIDEO with REAL latent structure (so there is something to learn)
# ═══════════════════════════════════════════════════════════════════════════════
class MovingBarVideos:
    """
    Generates clips of an oriented bar translating across the frame.
    Latent factors: orientation, velocity, phase, thickness.
    These are genuinely predictable from spatiotemporal context, so a model
    that only outputs a constant (collapse) will do measurably worse than
    one that actually learns the motion — which is the whole point.

    A "regime" bundles orientation + speed + thickness into a distinct motion
    signature (used as CDT world-states). Within a regime only the start
    position varies, so a well-trained encoder places same-regime clips close
    together and different regimes far apart.
    """
    def __init__(self, n_orient=8):
        self.n_orient = n_orient

    def _render(self, theta, speed, thick, x0, y0):
        vx, vy = math.cos(theta), math.sin(theta)
        ys, xs = np.mgrid[0:H, 0:W]
        clip = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            cx = (x0 + vx * speed * t) % W
            cy = (y0 + vy * speed * t) % W
            d = np.abs(-vy * (xs - cx) + vx * (ys - cy))
            clip[t] = np.exp(-(d**2) / (2 * thick**2))
        return clip

    def sample(self, batch: int) -> torch.Tensor:
        """Random clips spanning the full factor space (for self-sup training)."""
        clips = np.zeros((batch, T, H, W), dtype=np.float32)
        for b in range(batch):
            theta = np.random.randint(self.n_orient) * math.pi / self.n_orient
            speed = np.random.uniform(1.5, 3.5)
            thick = np.random.uniform(2.0, 4.0)
            x0, y0 = np.random.uniform(0, W), np.random.uniform(0, H)
            clips[b] = self._render(theta, speed, thick, x0, y0)
        return torch.from_numpy(clips)

    # ── distinct motion regimes for CDT world-states ─────────────────────────
    REGIMES = {
        # name: (orientation index, speed, thickness)
        0: (0, 1.6, 3.5),   # slow horizontal, thick
        1: (2, 3.2, 2.2),   # fast diagonal, thin
        2: (4, 2.4, 3.0),   # medium vertical
        3: (6, 3.4, 2.0),   # fast anti-diagonal, thin
        4: (1, 1.8, 3.8),   # slow shallow, very thick
        5: (5, 2.8, 2.4),   # medium steep
        6: (3, 3.5, 1.8),   # fastest, thinnest  (= ALERT: abnormal fast motion)
    }

    def sample_regime(self, regime_id, batch=1):
        theta_i, speed, thick = self.REGIMES[regime_id]
        theta = theta_i * math.pi / self.n_orient
        clips = np.zeros((batch, T, H, W), dtype=np.float32)
        for b in range(batch):
            x0, y0 = np.random.uniform(0, W), np.random.uniform(0, H)
            clips[b] = self._render(theta, speed, thick, x0, y0)
        return torch.from_numpy(clips)


# ═══════════════════════════════════════════════════════════════════════════════
# TUBELET EMBEDDING  (patchify spatiotemporal video)
# ═══════════════════════════════════════════════════════════════════════════════
class TubeletEmbed(nn.Module):
    """Split a clip into non-overlapping tubelets and linearly embed each."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(TUBELET_DIM, EMBED_DIM)
        # learned positional embedding for each of the N_TOKENS positions
        self.pos = nn.Parameter(torch.zeros(1, N_TOKENS, EMBED_DIM))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        B = clips.shape[0]
        # (B,T,H,W) -> tubelets (B, N_TOKENS, TUBELET_DIM)
        x = clips.reshape(B, GT, PT, GH, PH, GW, PW)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        x = x.reshape(B, N_TOKENS, TUBELET_DIM)
        x = self.proj(x) + self.pos
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER ENCODER  (context & target share this class; weights differ)
# ═══════════════════════════════════════════════════════════════════════════════
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    """ViT-style encoder over tubelet tokens."""
    def __init__(self, depth=DEPTH):
        super().__init__()
        self.embed  = TubeletEmbed()
        self.blocks = nn.ModuleList(
            [TransformerBlock(EMBED_DIM, HEADS) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(EMBED_DIM)

    def forward(self, clips=None, tokens=None, keep_idx=None):
        """
        Either pass raw `clips` (will be embedded) or pre-embedded `tokens`.
        If keep_idx is given, only those token positions are kept (context).
        """
        x = self.embed(clips) if tokens is None else tokens
        if keep_idx is not None:
            x = torch.gather(
                x, 1, keep_idx.unsqueeze(-1).expand(-1, -1, EMBED_DIM)
            )
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTOR  (predicts target embeddings at MASKED positions from context)
# ═══════════════════════════════════════════════════════════════════════════════
class Predictor(nn.Module):
    """
    Narrow Transformer. Takes encoded context tokens + learnable mask tokens
    placed at the masked positions (with their positional embeddings), and
    predicts the target encoder's embeddings for those masked positions.
    """
    def __init__(self):
        super().__init__()
        self.in_proj  = nn.Linear(EMBED_DIM, PRED_DIM)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, PRED_DIM))
        nn.init.trunc_normal_(self.mask_tok, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, N_TOKENS, PRED_DIM))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(PRED_DIM, HEADS) for _ in range(PRED_DEPTH)]
        )
        self.norm = nn.LayerNorm(PRED_DIM)
        self.out_proj = nn.Linear(PRED_DIM, EMBED_DIM)

    def forward(self, ctx, ctx_idx, pred_idx):
        """
        ctx      : (B, Nctx, EMBED_DIM)  encoded context tokens
        ctx_idx  : (B, Nctx)             their original positions
        pred_idx : (B, Npred)            masked positions to predict
        returns  : (B, Npred, EMBED_DIM) predicted target embeddings
        """
        B = ctx.shape[0]
        ctx = self.in_proj(ctx) + torch.gather(
            self.pos.expand(B, -1, -1), 1,
            ctx_idx.unsqueeze(-1).expand(-1, -1, PRED_DIM))
        # mask tokens at predicted positions (+ their positional embedding)
        mask = self.mask_tok.expand(B, pred_idx.shape[1], -1)
        mask = mask + torch.gather(
            self.pos.expand(B, -1, -1), 1,
            pred_idx.unsqueeze(-1).expand(-1, -1, PRED_DIM))
        x = torch.cat([ctx, mask], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pred = x[:, ctx.shape[1]:]           # take the mask-token outputs
        return self.out_proj(pred)


# ═══════════════════════════════════════════════════════════════════════════════
# V-JEPA MODEL  (context encoder + EMA target encoder + predictor)
# ═══════════════════════════════════════════════════════════════════════════════
class VJEPA(nn.Module):
    def __init__(self, ema_base=0.996):
        super().__init__()
        self.context_encoder = Encoder()
        self.target_encoder  = Encoder()
        # init target = context, and freeze it from gradient (EMA-only updates)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = Predictor()
        self.ema_base = ema_base

    @torch.no_grad()
    def update_target(self, tau):
        """EMA update — the real anti-collapse mechanism."""
        for pt, pc in zip(self.target_encoder.parameters(),
                          self.context_encoder.parameters()):
            pt.mul_(tau).add_(pc.detach(), alpha=1 - tau)

    def forward(self, clips, ctx_idx, pred_idx):
        # ── target branch: full clip through EMA encoder, stop-grad ──────────
        with torch.no_grad():
            full_tgt = self.target_encoder(clips=clips)          # (B,N,D)
            target = torch.gather(
                full_tgt, 1,
                pred_idx.unsqueeze(-1).expand(-1, -1, EMBED_DIM))
            # normalise target features (standard in V-JEPA)
            target = F.layer_norm(target, (EMBED_DIM,))
        # ── context branch: only visible tokens through context encoder ──────
        ctx = self.context_encoder(clips=clips, keep_idx=ctx_idx)
        # ── predict masked target embeddings ─────────────────────────────────
        pred = self.predictor(ctx, ctx_idx, pred_idx)
        return pred, target


# ═══════════════════════════════════════════════════════════════════════════════
# MASKING  (multi-block spatiotemporal mask — split tokens into context/predict)
# ═══════════════════════════════════════════════════════════════════════════════
def make_masks(batch, mask_ratio=0.6):
    """Random split of the N_TOKENS positions into context vs predicted."""
    n_pred = int(N_TOKENS * mask_ratio)
    ctx_idx, pred_idx = [], []
    for _ in range(batch):
        perm = torch.randperm(N_TOKENS)
        pred_idx.append(perm[:n_pred])
        ctx_idx.append(perm[n_pred:])
    return (torch.stack(ctx_idx).to(DEVICE),
            torch.stack(pred_idx).to(DEVICE))


# ═══════════════════════════════════════════════════════════════════════════════
# ANTI-COLLAPSE DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def collapse_metrics(model, data, n=64):
    """
    A collapsed encoder maps everything to (nearly) the same vector.
    We measure:
      - mean per-dimension std of embeddings (low => collapse)
      - effective rank of the embedding matrix (low => collapse)
    """
    clips = data.sample(n).to(DEVICE)
    emb = model.target_encoder(clips=clips)      # (n, N, D)
    emb = emb.reshape(-1, EMBED_DIM)             # (n*N, D)
    std = emb.std(dim=0).mean().item()
    # effective rank via singular value entropy
    emb_c = emb - emb.mean(0, keepdim=True)
    s = torch.linalg.svdvals(emb_c)
    p = s / s.sum()
    eff_rank = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
    return std, eff_rank


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
def train(steps=600, batch=32, lr=1.5e-3, log_every=25):
    data  = MovingBarVideos()
    model = VJEPA().to(DEVICE)
    opt   = torch.optim.AdamW(
        list(model.context_encoder.parameters()) +
        list(model.predictor.parameters()), lr=lr, weight_decay=0.04)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    history = {"step": [], "loss": [], "std": [], "rank": []}
    print(f"{'step':>5} {'loss':>9} {'emb_std':>9} {'eff_rank':>9}   (anti-collapse check)")
    print("-" * 55)

    for step in range(1, steps + 1):
        clips = data.sample(batch).to(DEVICE)
        ctx_idx, pred_idx = make_masks(batch)

        pred, target = model(clips, ctx_idx, pred_idx)
        loss = F.smooth_l1_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.context_encoder.parameters()) +
            list(model.predictor.parameters()), 1.0)
        opt.step()
        sched.step()

        # EMA schedule ramps tau from base -> 1.0
        tau = 1 - (1 - model.ema_base) * (math.cos(math.pi * step / steps) + 1) / 2
        model.update_target(tau)

        if step % log_every == 0 or step == 1:
            std, rank = collapse_metrics(model, data)
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["std"].append(std)
            history["rank"].append(rank)
            print(f"{step:>5} {loss.item():>9.5f} {std:>9.4f} {rank:>9.2f}")

    return model, data, history


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION: does the learned space encode the latent motion?
# ═══════════════════════════════════════════════════════════════════════════════
def probe_orientation(model, data, n=400):
    """
    Freeze the encoder, train a tiny linear probe to read orientation out of
    the pooled embedding. High accuracy => the embeddings are meaningful
    (i.e. the JEPA learned real structure, not noise, and did not collapse).
    """
    # build a labelled set
    clips_list, labels = [], []
    for _ in range(n):
        theta_idx = np.random.randint(data.n_orient)
        # regenerate one clip with a fixed orientation
        c = _one_clip(theta_idx, data.n_orient)
        clips_list.append(c)
        labels.append(theta_idx)
    clips = torch.stack(clips_list).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        emb = model.target_encoder(clips=clips).mean(dim=1)   # pool tokens -> (n,D)

    # split
    ntr = int(0.7 * n)
    Xtr, Xte = emb[:ntr], emb[ntr:]
    ytr, yte = y[:ntr], y[ntr:]

    probe = nn.Linear(EMBED_DIM, data.n_orient).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(probe(Xtr), ytr)
        loss.backward(); opt.step()
    acc = (probe(Xte).argmax(1) == yte).float().mean().item()
    chance = 1.0 / data.n_orient
    return acc, chance


def _one_clip(theta_idx, n_orient):
    theta = theta_idx * math.pi / n_orient
    vx, vy = math.cos(theta), math.sin(theta)
    speed = np.random.uniform(1.5, 3.0)
    thick = np.random.uniform(2.0, 4.0)
    x0, y0 = np.random.uniform(0, W), np.random.uniform(0, H)
    ys, xs = np.mgrid[0:H, 0:W]
    clip = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        cx = (x0 + vx * speed * t) % W
        cy = (y0 + vy * speed * t) % W
        d = np.abs(-vy * (xs - cx) + vx * (ys - cy))
        clip[t] = np.exp(-(d**2) / (2 * thick**2))
    return torch.from_numpy(clip)


if __name__ == "__main__":
    model, data, hist = train()
    print("\nFinal loss reduction: "
          f"{hist['loss'][0]:.4f} -> {hist['loss'][-1]:.4f} "
          f"({hist['loss'][0]/max(hist['loss'][-1],1e-9):.1f}x lower)")
    acc, chance = probe_orientation(model, data)
    print(f"Linear-probe orientation accuracy: {acc*100:.1f}% "
          f"(chance {chance*100:.1f}%)")
    torch.save(model.state_dict(), "/home/claude/vjepa_trained.pt")
    np.save("/home/claude/vjepa_history.npy", hist, allow_pickle=True)
    print("Saved model + history.")
