# UR5 Industrial Action Perception & Cognitive Digital Twin via Self-Supervised JEPA Architectures

This repository implements **Visual Joint-Embedding Predictive Architecture (V-JEPA)** and **Vision-Language JEPA (VL-JEPA)** (Chen et al. 2025, arXiv:2512.10942) for self-supervised kinematic phase discovery, anomaly detection, and state tracking in industrial UR5 robotic manipulation.

---

## 🌟 Interactive Demos (GitHub Pages)

The self-supervised perceptual representations and kinematic phase timelines are hosted interactively:

* 🌐 **[Pure V-JEPA Kinematic Phase Discovery](https://marcelo-feliciano-filho.github.io/JEPA_DEMO/)**
* 🌐 **[Zero-Shot Vision-Language Action Alignment](https://marcelo-feliciano-filho.github.io/JEPA_DEMO/UR5_VL_JEPA_interactive.html)**

---

## 🔬 Core Architecture Overview

| Component | Implementation Details |
| :--- | :--- |
| **Backbone Encoder** | Spatiotemporal ViT for 3D tubelet patch embedding extraction |
| **Self-Supervised Objective** | Masked tubelet representation prediction in latent space $\mathbb{R}^{128}$ |
| **Anti-Collapse Mechanism** | Exponential Moving Average (EMA) target encoder update ($\tau = 0.996$) |
| **Vision-Language Projection** | Shared unit hyper-sphere projection ($\mathbb{S}^{127}$) for zero-shot action prompt alignment |
| **Cognitive Digital Twin (CDT)** | Latent drift thresholding ($\Delta e_t$) for physical state monitoring |

---

## 📊 Discovered Kinematic Phases (UR5 Demonstration Benchmark)

Without human annotations or supervisory labels during training, pure V-JEPA self-supervised embeddings discover 5 distinct kinematic operational phases across the UR5 movement cycle:

1. **`Phase 0: IDLE / STILL`** — Robot arm stationary at rest position
2. **`Phase 1: MOVE DOWN`** — Downward trajectory approach toward target object
3. **`Phase 2: MOVING / TRAJECTORY`** — Active spatial manipulation trajectory
4. **`Phase 3: ITEM DROP / RELEASE`** — End-effector gripper actuation and payload release
5. **`Phase 4: MOVE UP / LIFT`** — Ascending retraction away from work surface

---

## 📁 Repository Structure

* `vjepa_real.py` — Core PyTorch V-JEPA model architecture (ViT Encoder, Predictor, EMA update)
* `vl_jepa_real.py` — Extension for Vision-Language Joint Embedding (Chen et al. 2025)
* `train_real.py` / `train_vl_jepa.py` — Self-supervised training pipelines
* `analyze_real.py` / `analyze_vl_jepa.py` — Latent space analysis and phase segmentation
* `rebuild_html.py` / `rebuild_vl_html.py` — Interactive HTML report generators
* `UR5_REAL_interactive.html` / `UR5_VL_JEPA_interactive.html` — Interactive web visualizations

---

## 🐳 Execution in Docker

All experiments and model evaluations run reproducibly inside Docker:

```bash
# Build & run V-JEPA pipeline
docker run --rm -v $(pwd):/workspace vjepa-pipeline bash run_pipeline.sh

# Run VL-JEPA Vision-Language pipeline
docker run --rm -v $(pwd):/workspace vjepa-pipeline bash run_vl_pipeline.sh
```
