#!/bin/bash
set -e

echo "=============================================="
echo "  VL-JEPA Pipeline — Real UR5 Video (Vision-Language)"
echo "=============================================="

echo ""
echo "=== STEP 1: Train VL-JEPA model ==="
python3 -u train_vl_jepa.py

echo ""
echo "=== STEP 2: Analyze Zero-Shot Vision-Language Action Discovery ==="
python3 -u analyze_vl_jepa.py

echo ""
echo "=== STEP 3: Generate UR5_VL_JEPA_interactive.html ==="
python3 -u rebuild_vl_html.py

echo ""
echo "=============================================="
echo "  VL-JEPA PIPELINE COMPLETE"
echo "=============================================="
ls -lh vljepa_realvid.pt vl_embs.npy UR5_VL_JEPA_interactive.html 2>/dev/null || true
