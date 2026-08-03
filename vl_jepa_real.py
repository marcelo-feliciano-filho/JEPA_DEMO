"""
=============================================================================
 vl_jepa_real.py  —  VL-JEPA: Vision-Language Joint Embedding Architecture
 (Chen et al. 2025, arXiv:2512.10942)
=============================================================================
Extends V-JEPA by joint-embedding visual tubelet features e_vis and 
natural language action descriptions e_text into a shared D=128 space.
=============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Re-use core V-JEPA constants and blocks
from vjepa_real import (
    EMBED_DIM, N_TOKENS,
    TubeletEmbed, Encoder, Predictor, make_masks
)

# Candidate Natural Language Action Prompts for UR5 Demonstration
ACTION_PROMPTS = [
    "robot arm idle stationary at rest",
    "robot arm moving downward towards object",
    "robot arm active movement trajectory",
    "gripper releasing and dropping red cube",
    "robot arm lifting upward away from table"
]


class TextEncoder(nn.Module):
    """
    Lightweight semantic text projection encoder.
    Maps word/character n-gram frequency bags or token IDs to 128D semantic space.
    """
    def __init__(self, vocab_size=500, embed_dim=EMBED_DIM):
        super().__init__()
        self.vocab = {}
        self.vocab_size = vocab_size
        self.tok_embed = nn.Embedding(vocab_size, 64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self._init_vocab()

    def _init_vocab(self):
        # Build character/word token dictionary
        words = ["robot", "arm", "idle", "stationary", "at", "rest",
                 "moving", "downward", "towards", "object", "active",
                 "movement", "trajectory", "gripper", "releasing", "and",
                 "dropping", "red", "cube", "lifting", "upward", "away", "from", "table"]
        for idx, w in enumerate(words):
            self.vocab[w] = idx + 1

    def tokenize(self, text):
        tokens = [self.vocab.get(w.lower(), 0) for w in text.split()]
        if not tokens:
            tokens = [0]
        return torch.tensor(tokens, dtype=torch.long)

    def forward(self, prompt_list, device=None):
        """
        prompt_list : List of str
        returns     : (M, EMBED_DIM) normalized text embeddings
        """
        embs = []
        for text in prompt_list:
            toks = self.tokenize(text)
            if device is not None:
                toks = toks.to(device)
            vecs = self.tok_embed(toks)            # (L, 64)
            pooled = vecs.mean(dim=0, keepdim=True) # (1, 64)
            proj = self.mlp(pooled)                 # (1, EMBED_DIM)
            embs.append(proj)
        out = torch.cat(embs, dim=0)               # (M, EMBED_DIM)
        return F.normalize(out, p=2, dim=-1)


class VLJEPA(nn.Module):
    """
    VL-JEPA Model:
      - Visual Context Encoder (ViT)
      - EMA Visual Target Encoder (ViT)
      - Visual Predictor (Predicts target tubelet embeddings from context)
      - Text Encoder / Projection Head (Maps language prompts to shared 128D space)
    """
    def __init__(self, ema_base=0.996):
        super().__init__()
        self.context_encoder = Encoder()
        self.target_encoder  = Encoder()
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = Predictor()
        self.text_encoder = TextEncoder(embed_dim=EMBED_DIM)
        
        # Visual-to-Language Shared Space Projection Head
        self.vis_proj = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.GELU(),
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM)
        )
        self.ema_base = ema_base

    @torch.no_grad()
    def update_target(self, tau):
        for pt, pc in zip(self.target_encoder.parameters(),
                          self.context_encoder.parameters()):
            pt.mul_(tau).add_(pc.detach(), alpha=1 - tau)

    def forward(self, clips, ctx_idx, pred_idx, text_prompts=None):
        """
        clips        : (B, T, H, W) video clips
        ctx_idx      : (B, Nctx) unmasked context indices
        pred_idx     : (B, Npred) masked target indices
        text_prompts : List[str] of natural language prompts
        returns      : (pred_vis, target_vis, vl_align_loss)
        """
        # 1. Target visual branch (EMA encoder, stop-grad)
        with torch.no_grad():
            full_tgt = self.target_encoder(clips=clips)
            target = torch.gather(
                full_tgt, 1,
                pred_idx.unsqueeze(-1).expand(-1, -1, EMBED_DIM))
            target = F.layer_norm(target, (EMBED_DIM,))

        # 2. Context visual branch & predictor
        ctx = self.context_encoder(clips=clips, keep_idx=ctx_idx)
        pred = self.predictor(ctx, ctx_idx, pred_idx)

        # 3. Vision-Language Alignment Branch
        vl_loss = torch.tensor(0.0, device=clips.device)
        if text_prompts is not None:
            # Pooled visual representation for full clip
            vis_pooled = full_tgt.mean(dim=1)                  # (B, EMBED_DIM)
            vis_shared = F.normalize(self.vis_proj(vis_pooled), p=2, dim=-1)
            
            # Encode text prompts
            text_embs = self.text_encoder(text_prompts, device=clips.device) # (M, EMBED_DIM)

            # Cosine similarity matrix between clips and text prompts
            sim_matrix = vis_shared @ text_embs.T               # (B, M)
            
            # Softmax assignment over prompts
            probs = F.softmax(sim_matrix / 0.1, dim=-1)         # (B, M)
            mean_probs = probs.mean(dim=0) + 1e-9              # (M,)
            
            # Diversity loss: maximize entropy of batch prompt distribution (prevents collapse to 1 prompt)
            diversity_loss = (mean_probs * torch.log(mean_probs)).sum() # negative entropy
            
            # Alignment loss: maximize margin between top matched prompt and non-matched prompts
            max_sim, _ = sim_matrix.max(dim=-1)
            align_loss = 1.0 - max_sim.mean()

            vl_loss = align_loss + 1.5 * diversity_loss

        return pred, target, vl_loss
