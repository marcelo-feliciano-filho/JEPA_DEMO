"""
=============================================================================
 train_vl_jepa.py  —  Self-Supervised & Vision-Language Joint Training
 (VL-JEPA: Chen et al. 2025, arXiv:2512.10942)
=============================================================================
"""
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from vl_jepa_real import VLJEPA, ACTION_PROMPTS, make_masks

def main():
    print("=" * 65)
    print("  VL-JEPA Joint Vision-Language Self-Supervised Training")
    print("=" * 65)

    # 1. Load real video clips and learned phase labels
    clips_np = np.load("real_clips.npy")                       # (668, 8, 64, 64)
    labels_np = np.load("real_labels.npy")                     # (668,)
    min_len = min(len(clips_np), len(labels_np))
    clips_np = clips_np[:min_len]
    labels_np = labels_np[:min_len]

    # Downsample spatially to 32x32 for CPU efficiency
    import torch.nn.functional as F_sp
    clips_t = torch.from_numpy(clips_np).float()
    labels_t = torch.from_numpy(labels_np).long()
    B, T, H, W = clips_t.shape
    clips32 = F_sp.interpolate(clips_t.view(B * T, 1, H, W), size=(32, 32), mode='bilinear', align_corners=False).view(B, T, 32, 32)
    print(f"Loaded training pool: {clips32.shape}, labels: {labels_t.shape}")

    # 2. Instantiate VL-JEPA model & optimizer
    device = torch.device("cpu")
    model = VLJEPA().to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.05
    )

    steps = 600
    batch_size = 32
    print(f"Training VL-JEPA for {steps} steps (batch size {batch_size})...")
    print(f"{'step':>6} {'pred_loss':>11} {'vl_loss':>11} {'total_loss':>11}")
    print("-" * 50)

    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(clips32), (batch_size,))
        batch_clips = clips32[idx].to(device)
        batch_labels = labels_t[idx].to(device)

        ctx_idx, pred_idx = make_masks(batch_size, mask_ratio=0.6)
        ctx_idx = ctx_idx.to(device)
        pred_idx = pred_idx.to(device)

        pred_vis, target_vis, vl_align_loss = model(
            clips=batch_clips, ctx_idx=ctx_idx, pred_idx=pred_idx,
            text_prompts=ACTION_PROMPTS, prompt_targets=batch_labels
        )

        # Smooth L1 predictor loss + Vision-Language cross-entropy alignment loss
        loss_pred = F.smooth_l1_loss(pred_vis, target_vis)
        loss_total = loss_pred + 0.3 * vl_align_loss

        opt.zero_grad()
        loss_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Target EMA update
        tau = 0.996 + (1.0 - 0.996) * (step / steps)
        model.update_target(tau)

        if step % 100 == 0 or step == 1:
            print(f"{step:>6} {loss_pred.item():>11.5f} {vl_align_loss.item():>11.5f} {loss_total.item():>11.5f}")

    dur = time.time() - t0
    torch.save(model.state_dict(), "vljepa_realvid.pt")
    print("-" * 50)
    print(f"VL-JEPA Training Complete in {dur:.1f}s. Saved weights to vljepa_realvid.pt")

if __name__ == "__main__":
    main()
