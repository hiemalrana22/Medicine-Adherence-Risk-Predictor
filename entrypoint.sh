#!/bin/bash
# ============================================================
# entrypoint.sh — Medical Adherence Predictor
# ============================================================
# This script is the Docker container entry point.
# It:
#   1. Runs the full ML pipeline (if models don't exist yet)
#   2. Starts the Streamlit dashboard
# ============================================================

set -e   # Exit immediately on any error

echo "============================================================"
echo "  Medical Adherence Predictor — Starting Up"
echo "============================================================"

# ── Check if models already exist (skip pipeline on re-runs) ──
MODELS_DIR="/app/outputs/models"
PIPELINE_DONE_FLAG="$MODELS_DIR/random_forest.pkl"

if [ -f "$PIPELINE_DONE_FLAG" ]; then
    echo "[SKIP] Models already exist — skipping pipeline run."
    echo "       Delete outputs/models/ to force re-training."
else
    echo ""
    echo "► Step 1/4: Preprocessing data..."
    python src/preprocessing.py
    echo ""

    echo "► Step 2/4: Feature engineering..."
    python src/feature_engineering.py
    echo ""

    echo "► Step 3/4: Training ML models..."
    python src/train.py
    echo ""

    echo "► Step 4/4: Evaluating models..."
    python src/evaluate.py
    echo ""

    echo "✅ Pipeline complete!"
fi

echo ""
echo "============================================================"
echo "  Launching Streamlit Dashboard"
echo "  Open: http://localhost:8501"
echo "============================================================"
echo ""

# ── Start Streamlit ──────────────────────────────────────────
exec streamlit run app/dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.base=light \
    --theme.primaryColor="#667eea" \
    --theme.backgroundColor="#ffffff" \
    --theme.secondaryBackgroundColor="#f0f2f6"
