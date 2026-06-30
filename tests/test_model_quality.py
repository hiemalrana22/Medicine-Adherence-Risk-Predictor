"""
Model-quality guardrail tests.

The whole point of this project is a *learnable* adherence signal. These
tests train a small model on freshly generated data and assert it clears a
held-out performance floor — so the metrics can never silently regress back
to coin-flip territory (the bug this repo previously had, where the README
claimed 0.89 AUC but the committed model scored ~0.55).
"""
import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score

from preprocessing import generate_synthetic_data
from feature_engineering import (
    create_refill_ratio, create_financial_burden, create_age_group,
    create_medication_complexity, create_days_supply_category, create_refill_gap,
)


def _featurize(df):
    df = create_refill_ratio(df)
    df = create_financial_burden(df)
    df = create_age_group(df)
    df = create_medication_complexity(df)
    df = create_days_supply_category(df)
    df = create_refill_gap(df)
    return df


@pytest.fixture(scope="module")
def trained():
    df = generate_synthetic_data(n=2000, seed=42)
    df = _featurize(df)
    feats = ["age", "annual_contribution", "claim_amount", "days_supply",
             "num_medications", "refill_ratio", "financial_burden", "refill_gap"]
    df = df.dropna(subset=feats + ["adherent"])
    X, y = df[feats], df["adherent"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=8,
        random_state=42, n_jobs=-1).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    return model, Xte, yte, proba, feats


def test_beats_random_on_holdout(trained):
    _, _, yte, proba, _ = trained
    auc = roc_auc_score(yte, proba)
    assert auc > 0.70, f"held-out ROC-AUC {auc:.3f} regressed toward random"


def test_at_risk_pr_auc_floor(trained):
    _, _, yte, proba, _ = trained
    at_risk_true = (yte == 0).astype(int)
    ap = average_precision_score(at_risk_true, 1 - proba)
    assert ap > 0.65, f"at-risk PR-AUC {ap:.3f} below floor"


def test_at_risk_recall_floor(trained):
    model, Xte, yte, _, _ = trained
    rec = recall_score(yte, model.predict(Xte), pos_label=0)
    assert rec > 0.55, f"at-risk recall {rec:.3f} too low to be useful"


def test_refill_ratio_is_a_top_predictor(trained):
    model, _, _, _, feats = trained
    importances = pd.Series(model.feature_importances_, index=feats)
    # The README's headline insight: refill ratio dominates.
    assert importances.idxmax() == "refill_ratio"
