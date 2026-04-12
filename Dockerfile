# ============================================================
# Dockerfile - Medical Adherence Predictor
# ============================================================
# This file defines the Docker container that packages
# the entire Python ML pipeline so it can run anywhere.
# ============================================================

# Start from official Python 3.10 slim image (lightweight)
FROM python:3.10-slim

# Set metadata
LABEL maintainer="Medical Adherence Predictor"
LABEL description="End-to-end ML pipeline for medication adherence prediction"

# Set environment variables
# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures Python output is sent straight to terminal (no buffering)
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# ── Install system dependencies ──────────────────────────────
# These are needed for some Python packages to compile
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ───────────────────────────────
# Copy requirements first (Docker layer caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project files into container ─────────────────────────
COPY src/ ./src/
COPY data/ ./data/

# ── Create output directories ─────────────────────────────────
RUN mkdir -p outputs/figures outputs/models outputs/reports

# ── Default command: run the full ML pipeline ─────────────────
# This runs all steps in sequence when the container starts
CMD ["sh", "-c", "\
    echo 'Step 1: Preprocessing...' && python src/preprocessing.py && \
    echo 'Step 2: Feature Engineering...' && python src/feature_engineering.py && \
    echo 'Step 3: Training Models...' && python src/train.py && \
    echo 'Step 4: Evaluating Models...' && python src/evaluate.py && \
    echo 'Pipeline complete. Check outputs/ folder.' \
"]

# ── How to use this Dockerfile ────────────────────────────────
# BUILD:  docker build -t medical-adherence-predictor .
# RUN:    docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs medical-adherence-predictor
# SHELL:  docker run -it medical-adherence-predictor bash
