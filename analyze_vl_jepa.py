"""
=============================================================================
 analyze_vl_jepa.py  —  Zero-Shot Vision-Language Action Discovery
 (VL-JEPA: Chen et al. 2025, arXiv:2512.10942)
=============================================================================
"""
import sys, numpy as np, torch, torch.nn.functional as F
from vl_jepa_real import VLJEPA, ACTION_PROMPTS

def main():
    print("=" * 65)
    print("  VL-JEPA Zero-Shot Vision-Language Action Analysis")
    print("=" * 65)

    # 1. Load real video clips
    clips_np = np.load("real_clips.npy")                       # (668, 8, 64, 64)
    clips_t = torch.from_numpy(clips_np).float()
    B, T, H, W = clips_t.shape
    clips32 = F.interpolate(clips_t.view(B * T, 1, H, W), size=(32, 32), mode='bilinear', align_corners=False).view(B, T, 32, 32)

    # 2. Load trained VL-JEPA model
    device = torch.device("cpu")
    model = VLJEPA().to(device)
    model.load_state_dict(torch.load("vljepa_realvid.pt"))
    model.eval()

    print("Extracting joint spatiotemporal visual & text embeddings...")
    with torch.no_grad():
        # Encode visual clips using target encoder
        full_tgt = model.target_encoder(clips=clips32)          # (B, N, D)
        vis_pooled = full_tgt.mean(dim=1)                      # (B, D)
        vis_shared = F.normalize(model.vis_proj(vis_pooled), p=2, dim=-1).cpu().numpy() # (B, D)

        # Encode candidate natural language prompts
        text_embs = model.text_encoder(ACTION_PROMPTS, device=device).cpu().numpy()     # (M, D)

    # Load learned spatiotemporal phase clusters from V-JEPA target encoder
    vjepa_labels = np.load("real_labels.npy")
    min_len = min(len(vis_shared), len(vjepa_labels))
    vis_shared = vis_shared[:min_len]
    vjepa_labels = vjepa_labels[:min_len]

    # 100% Pure Raw Cosine Similarity Matrix between trained visual embeddings and text prompt embeddings
    sim_matrix = vis_shared @ text_embs.T                       # (B, M)

    tau = 0.1
    probs_matrix = F.softmax(torch.from_numpy(sim_matrix) / tau, dim=-1).numpy()
    vl_labels = np.argmax(probs_matrix, axis=1)

    print(f"Encoded {len(vis_shared)} video windows into joint 128D space.")
    print(f"Zero-Shot VL-JEPA Phase Distributions across video:")
    for m, prompt in enumerate(ACTION_PROMPTS):
        count = np.sum(vl_labels == m)
        pct = (count / len(vl_labels)) * 100
        print(f"  Prompt {m}: '{prompt}' -> {count} windows ({pct:.1f}%)")
    vjepa_labels = vjepa_labels[:min_len]
    sim_matrix = sim_matrix[:min_len]
    probs_matrix = probs_matrix[:min_len]
    vis_shared_np = vis_shared[:min_len]

    # Save output numpy arrays
    np.save("vl_embs.npy", vis_shared_np)
    np.save("vl_labels.npy", vl_labels)
    np.save("vl_sims.npy", probs_matrix)
    np.save("vl_prompt_embs.npy", text_embs)

    print("Saved vl_embs.npy, vl_labels.npy, vl_sims.npy, vl_prompt_embs.npy.")
    print("=" * 65)

if __name__ == "__main__":
    main()
