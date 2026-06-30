"""
Unit tests for src/preprocessing.py

These cover the data-cleaning building blocks: synthetic generation,
missing-value imputation, duplicate removal, IQR outlier capping, and
categorical encoding. They are deterministic and run in well under a second.
"""
import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    generate_synthetic_data,
    handle_missing_values,
    remove_duplicates,
    handle_outliers,
    encode_categoricals,
)

EXPECTED_COLS = {
    "patient_id", "age", "gender", "insurance_type", "annual_contribution",
    "claim_amount", "expected_refills", "refills_received", "days_supply",
    "chronic_condition", "num_medications", "adherent",
}


def test_generate_shape_and_columns(synthetic_df):
    assert len(synthetic_df) == 500
    assert EXPECTED_COLS.issubset(set(synthetic_df.columns))


def test_generate_is_deterministic():
    a = generate_synthetic_data(n=200, seed=123)
    b = generate_synthetic_data(n=200, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_target_is_binary_and_not_degenerate(synthetic_df):
    vals = set(synthetic_df["adherent"].dropna().unique())
    assert vals.issubset({0, 1})
    # Both classes must be present and reasonably balanced (not a constant target).
    rate = synthetic_df["adherent"].mean()
    assert 0.25 < rate < 0.75, f"target rate {rate:.2f} is too skewed"


def test_refills_received_never_exceeds_expected(synthetic_df):
    assert (synthetic_df["refills_received"] <= synthetic_df["expected_refills"]).all()
    assert (synthetic_df["refills_received"] >= 0).all()


def test_generate_injects_missingness(synthetic_df):
    # The generator deliberately adds ~4% NaNs to mimic real-world data.
    assert synthetic_df.isnull().sum().sum() > 0


def test_handle_missing_values_removes_all_nans(synthetic_df):
    out = handle_missing_values(synthetic_df)
    assert out.isnull().sum().sum() == 0
    # Imputation must not drop or add rows.
    assert len(out) == len(synthetic_df)


def test_remove_duplicates_drops_exact_repeats():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out = remove_duplicates(df)
    assert len(out) == 2


def test_handle_outliers_caps_extremes():
    df = pd.DataFrame({
        "age": [30, 31, 32, 33, 5000],          # 5000 is an extreme outlier
        "annual_contribution": [3000] * 5,
        "claim_amount": [1000] * 5,
        "expected_refills": [6] * 5,
        "refills_received": [5] * 5,
        "num_medications": [3] * 5,
    })
    out = handle_outliers(df)
    assert out["age"].max() < 5000          # extreme value was capped down
    assert len(out) == len(df)              # capping preserves row count


def test_encode_categoricals_produces_numeric(synthetic_df):
    clean = handle_missing_values(synthetic_df)
    encoded = encode_categoricals(clean)
    # Original text columns and the ID must be gone.
    for col in ["gender", "insurance_type", "chronic_condition", "patient_id"]:
        assert col not in encoded.columns
    assert "gender_encoded" in encoded.columns
    # No text/object columns may remain — everything must be model-ready
    # (numeric or boolean one-hot dummies).
    assert not any(dt == object for dt in encoded.dtypes)
    assert all(np.issubdtype(dt, np.number) or np.issubdtype(dt, np.bool_)
               for dt in encoded.dtypes)
