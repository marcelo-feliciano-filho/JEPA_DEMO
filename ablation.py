"""Ablation: what happens WITHOUT the EMA anti-collapse mechanism?
If we make the target encoder identical to the context encoder at every step
(tau=0, no stop-grad separation of dynamics), the model can cheat by mapping
everything toward a trivial solution — collapse. We measure it."""
import math, numpy as np, torch, torch.nn.functional as F
import vjepa_real as V

torch.manual_seed(0); np.random.seed(0)

def train_variant(use_ema, steps=400, batch=32, lr=1.5e-3):
    data = V.MovingBarVideos()
    model = V.VJEPA().to(V.DEVICE)
    opt = torch.optim.AdamW(
        list(model.context_encoder.parameters())+list(model.predictor.parameters()),
        lr=lr, weight_decay=0.04)
    for step in range(1, steps+1):
        clips = data.sample(batch).to(V.DEVICE)
        ctx_idx, pred_idx = V.make_masks(batch)
        pred, target = model(clips, ctx_idx, pred_idx)
        loss = F.smooth_l1_loss(pred, target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.context_encoder.parameters())+list(model.predictor.parameters()),1.0)
        opt.step()
        if use_ema:
            tau = 1-(1-model.ema_base)*(math.cos(math.pi*step/steps)+1)/2
            model.update_target(tau)
        else:
            # NO EMA: copy context weights straight into target every step (tau=0)
            model.target_encoder.load_state_dict(model.context_encoder.state_dict())
    std, rank = V.collapse_metrics(model, data)
    acc, chance = V.probe_orientation(model, data)
    return std, rank, acc, chance

print("Variant            emb_std   eff_rank   probe_acc  (chance 12.5%)")
print("-"*64)
s1,r1,a1,c = train_variant(use_ema=True)
print(f"WITH EMA (real)    {s1:7.4f}   {r1:7.2f}    {a1*100:6.1f}%")
s2,r2,a2,c = train_variant(use_ema=False)
print(f"NO EMA (ablated)   {s2:7.4f}   {r2:7.2f}    {a2*100:6.1f}%")
print("-"*64)
print(f"Rank drop without anti-collapse: {r1:.1f} -> {r2:.1f}")
print(f"Probe accuracy drop:             {a1*100:.1f}% -> {a2*100:.1f}%")
