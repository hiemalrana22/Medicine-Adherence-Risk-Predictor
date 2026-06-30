"""Shared pytest fixtures + path setup so tests can import the src/ modules."""
import os
import sys

import pytest

# Make `src/` importable as top-level modules (preprocessing, feature_engineering, ...)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture(scope="session")
def synthetic_df():
    """A small, deterministic synthetic dataset shared across tests."""
    from preprocessing import generate_synthetic_data
    return generate_synthetic_data(n=500, seed=7)
