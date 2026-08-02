#!/bin/bash
set -e

echo "=============================================="
echo "  V-JEPA Pipeline — Real UR5 Video (Pure Unsupervised)"
echo "=============================================="

echo ""
echo "=== STEP 1: Extract frames from MP4 ==="
python3 -u extract_frames.py

echo ""
echo "=== STEP 2: Train V-JEPA self-supervised ==="
python3 -u train_real.py

echo ""
echo "=== STEP 3: Analyze embeddings + discover phases ==="
python3 -u analyze_real.py

echo ""
echo "=== STEP 4: Rebuild HTML data blob ==="
python3 -u rebuild_html.py

echo ""
echo "=============================================="
echo "  PIPELINE COMPLETE"
echo "=============================================="
echo "Outputs:"
ls -lh real_*.npy vjepa_realvid.pt UR5_REAL_interactive.html 2>/dev/null || true
