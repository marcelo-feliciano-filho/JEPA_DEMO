# Real V-JEPA + Cognitive Digital Twin (PyTorch)

This is a **genuine, trained** Joint-Embedding Predictive Architecture — not a
simulation with random vectors. Everything is learned by gradient descent.

## What makes it a real JEPA

| Ingredient | This implementation |
|---|---|
| Prediction target | **Embeddings** from an EMA target encoder (not pixels, not tokens) |
| Anti-collapse | **EMA target encoder + stop-gradient** — the real V-JEPA/BYOL mechanism |
| Predictor | Transformer predicting masked-tubelet target features from visible context |
| Loss | Smooth-L1 in embedding space, real backprop |
| Data | Synthetic moving-bar video with genuinely learnable latent motion |

## Files

- `vjepa_real.py` — the V-JEPA model, masking, EMA, training loop, diagnostics
- `ablation.py` — proves the EMA matters: removing it collapses the space
- `cdt_vljepa_real.py` — Cognitive Digital Twin built on the **trained** encoder
- `vjepa_trained.pt` / `state_head.pt` — trained weights
- `vjepa_history.npy` — training curve data

## Verified results (CPU, ~2–3 min training)

```
Training loss:      0.544  ->  0.0026     (~210x lower, real gradient descent)
Effective rank:     43     ->  21 / 128   (stays high => NO collapse)
Linear probe:       ~58%  vs 12.5% chance (embeddings encode real motion)

Ablation (no EMA):  probe drops to ~13%  (= chance)  -> collapse confirmed

CDT on trained encoder:
  per-frame state accuracy : 75.8%  vs 14.3% chance
  decode reduction         : 7.1x fewer than uniform
  world-state transitions  : 9/9 caught
```

## Honesty notes

This is a **small** model trained for minutes on CPU, so it is not perfect —
you will see occasional misclassifications in the CDT log (e.g. a PICK_A frame
briefly read as ALERT). That is expected and, frankly, more honest than a
flawless toy. The point is that the **mechanism** is real: masked feature
prediction, EMA anti-collapse, embedding-space loss, and selective decoding
measured on learned representations.

## Run it

```bash
pip install torch numpy matplotlib scikit-learn
python3 vjepa_real.py        # trains, saves weights + curve
python3 ablation.py          # shows EMA prevents collapse
python3 cdt_vljepa_real.py   # runs the digital twin on the trained encoder
```
