# ============================================================
# Dockerfile — Medical Adherence Predictor
# ============================================================
# Architecture:
#   1. Run full ML pipeline  (preprocessing → training → evaluation)
#   2. Serve interactive Streamlit dashboard on port 8501
#
# Build:  docker build -t medical-adherence-predictor .
# Run:    docker-compose up
# Open:   http://localhost:8501
# ============================================================

FROM python:3.10-slim

# ── Metadata ──────────────────────────────────────────────────
LABEL maintainer="Medical Adherence Predictor"
LABEL description="ML pipeline + Streamlit dashboard for medication adherence prediction"

# ── Environment ───────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_BASE=light

WORKDIR /app

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────
COPY src/         ./src/
COPY app/         ./app/
COPY data/        ./data/
COPY scripts/     ./scripts/

# ── Create output directories ─────────────────────────────────
RUN mkdir -p outputs/figures outputs/models outputs/reports

# ── Copy entrypoint ───────────────────────────────────────────
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# ── Expose Streamlit port ─────────────────────────────────────
EXPOSE 8501

# ── Healthcheck ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Start: pipeline → dashboard ───────────────────────────────
ENTRYPOINT ["./entrypoint.sh"]
