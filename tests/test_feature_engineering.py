"""
Unit tests for src/feature_engineering.py

These verify that each engineered feature is computed correctly and stays
within its documented range — the features the model (and the live demo)
depend on.
"""
import numpy as np
import pandas as pd
import pytest

from feature_engineering import (
    create_refill_ratio,
    create_financial_burden,
    create_age_group,
    create_medication_complexity,
    create_days_supply_category,
    create_refill_gap,
)


@pytest.fixture
def sample():
    return pd.DataFrame({
        "age": [25, 50, 70, 18, 85],
        "annual_contribution": [3000, 2000, 4000, 1000, 5000],
        "claim_amount": [1500, 4000, 1000, 0, 2500],
        "expected_refills": [10, 8, 6, 4, 12],
        "refills_received": [8, 0, 6, 4, 3],
        "days_supply": [30, 60, 90, 30, 90],
        "num_medications": [1, 3, 6, 2, 7],
    })


def test_refill_ratio_value_and_range(sample):
    out = create_refill_ratio(sample)
    assert out["refill_ratio"].between(0, 1).all()
    # 8/10 = 0.8 ; 0/8 = 0.0 ; 6/6 = 1.0
    assert out["refill_ratio"].iloc[0] == pytest.approx(0.8)
    assert out["refill_ratio"].iloc[1] == pytest.approx(0.0)
    assert out["refill_ratio"].iloc[2] == pytest.approx(1.0)


def test_refill_ratio_handles_zero_expected():
    df = pd.DataFrame({"expected_refills": [0], "refills_received": [3]})
    out = create_refill_ratio(df)
    assert out["refill_ratio"].iloc[0] == 0  # no division-by-zero blow-up


def test_financial_burden_value_and_cap(sample):
    out = create_financial_burden(sample)
    assert (out["financial_burden"] >= 0).all()
    assert (out["financial_burden"] <= 5).all()         # capped at 5x
    # 1500/3000 = 0.5
    assert out["financial_burden"].iloc[0] == pytest.approx(0.5)


def test_age_group_buckets(sample):
    out = create_age_group(sample)
    # 25->Young(0), 50->Adult(1), 70->Elderly(2), 18->Young(0), 85->Elderly(2)
    assert list(out["age_group"]) == [0, 1, 2, 0, 2]


def test_medication_complexity_buckets(sample):
    out = create_medication_complexity(sample)
    # 1->Low(0), 3->Med(1), 6->High(2), 2->Low(0), 7->High(2)
    assert list(out["medication_complexity"]) == [0, 1, 2, 0, 2]


def test_days_supply_category(sample):
    out = create_days_supply_category(sample)
    # 30->Short(0), 60->Medium(1), 90->Long(2)
    assert list(out["supply_category"]) == [0, 1, 2, 0, 2]


def test_refill_gap_non_negative(sample):
    out = create_refill_gap(sample)
    assert (out["refill_gap"] >= 0).all()
    # expected - received: 10-8=2, 8-0=8, 6-6=0
    assert list(out["refill_gap"].iloc[:3]) == [2, 8, 0]
