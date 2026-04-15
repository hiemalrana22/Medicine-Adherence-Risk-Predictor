"""
============================================================
feature_engineering.py - Creating Meaningful Features
============================================================
PURPOSE:
    Transform raw cleaned data into new features that capture
    domain knowledge about medication adherence.

    Good features = better model performance + interpretability.

INPUT:  data/processed/cleaned_data.csv
        data/raw/medication_adherence.csv  (for original values)
OUTPUT: data/processed/featured_data.csv
============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


# ── PATHS ─────────────────────────────────────────────────────
RAW_PATH      = "data/raw/medication_adherence.csv"
CLEANED_PATH  = "data/processed/cleaned_data.csv"
OUTPUT_PATH   = "data/processed/featured_data.csv"


def load_raw_for_features(raw_path: str, cleaned_path: str):
    """
    We need the ORIGINAL (unscaled) values to engineer features
    that make intuitive sense. Then we'll merge with cleaned data.
    """
    raw = pd.read_csv(raw_path)
    cleaned = pd.read_csv(cleaned_path)
    return raw, cleaned


def create_refill_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    FEATURE: refill_ratio = refills_received / expected_refills

    WHY IT MATTERS:
        This is the most direct measure of adherence behavior.
        A patient who refilled 10 out of 12 expected times (ratio=0.83)
        is much more likely to be adherent than one with ratio=0.25.

    RANGE: 0.0 (never refilled) → 1.0 (always refilled)
    """
    df = df.copy()
    df['refill_ratio'] = np.where(
        df['expected_refills'] > 0,
        df['refills_received'] / df['expected_refills'],
        0  # Avoid division by zero
    ).clip(0, 1)

    print(f"   [OK] 'refill_ratio' created | Mean: {df['refill_ratio'].mean():.3f}")
    return df


def create_financial_burden(df: pd.DataFrame) -> pd.DataFrame:
    """
    FEATURE: financial_burden = claim_amount / annual_contribution

    WHY IT MATTERS:
        High out-of-pocket costs are a major barrier to adherence.
        If a patient's claims greatly exceed what they contribute,
        they may stop picking up prescriptions due to cost.

    INTERPRETATION: >1.0 means patient claims more than they pay in (high burden)
    """
    df = df.copy()
    df['financial_burden'] = np.where(
        df['annual_contribution'] > 0,
        df['claim_amount'] / df['annual_contribution'],
        0
    ).clip(0, 5)  # Cap at 5× to handle extreme values

    print(f"   [OK] 'financial_burden' created | Mean: {df['financial_burden'].mean():.3f}")
    return df


def create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    FEATURE: age_group (Young / Adult / Elderly)

    WHY IT MATTERS:
        Adherence patterns differ significantly by age:
        - Young (18–35): Often forget, underestimate severity
        - Adult (36–64): Generally better adherence
        - Elderly (65+): May have more barriers (mobility, cost, complexity)

    Encoded as ordinal: Young=0, Adult=1, Elderly=2
    """
    df = df.copy()

    def assign_age_group(age):
        if age <= 35:
            return 0   # Young
        elif age <= 64:
            return 1   # Adult
        else:
            return 2   # Elderly

    # 'age' may already be scaled — use raw age if available
    # We'll work on the raw df and merge, so age is original here
    if df['age'].between(0, 120).all():
        df['age_group'] = df['age'].apply(assign_age_group)
    else:
        # If scaled, create approximate groups from quantiles
        df['age_group'] = pd.qcut(df['age'], q=3, labels=[0, 1, 2]).astype(int)

    counts = df['age_group'].value_counts().sort_index()
    labels = {0: 'Young', 1: 'Adult', 2: 'Elderly'}
    for k, v in counts.items():
        print(f"   [OK] age_group={labels[k]}: {v} patients")

    return df


def create_medication_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """
    FEATURE: medication_complexity (Low / Medium / High = 0/1/2)

    WHY IT MATTERS:
        Patients taking many medications (polypharmacy) face:
        - Complex schedules → more likely to miss doses
        - Drug interactions → side effects → stopping medications
        - Higher costs → financial barrier

    Threshold: 1–2 meds = Low, 3–5 = Medium, 6+ = High
    """
    df = df.copy()

    def complexity(n):
        if n <= 2:
            return 0   # Low
        elif n <= 5:
            return 1   # Medium
        else:
            return 2   # High

    if 'num_medications' in df.columns:
        # If raw values (not scaled)
        if df['num_medications'].between(1, 20).all():
            df['medication_complexity'] = df['num_medications'].apply(complexity)
        else:
            df['medication_complexity'] = pd.qcut(
                df['num_medications'], q=3, labels=[0, 1, 2]
            ).astype(int)

        print(f"   [OK] 'medication_complexity' created | "
              f"Distribution: {df['medication_complexity'].value_counts().sort_index().to_dict()}")
    return df


def create_days_supply_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    FEATURE: supply_category (Short/Medium/Long = 0/1/2)

    WHY IT MATTERS:
        Longer prescription supplies (90-day) are associated with
        better adherence — fewer trips to pharmacy, lower per-unit cost.
        Studies show 90-day fills improve adherence by 10–15%.
    """
    df = df.copy()

    if 'days_supply' in df.columns:
        if df['days_supply'].between(1, 365).all():
            df['supply_category'] = pd.cut(
                df['days_supply'],
                bins=[0, 30, 60, 365],
                labels=[0, 1, 2]  # Short, Medium, Long
            ).astype(int)
        else:
            df['supply_category'] = pd.qcut(
                df['days_supply'], q=3, labels=[0, 1, 2]
            ).astype(int)

        print(f"   [OK] 'supply_category' created | "
              f"Distribution: {df['supply_category'].value_counts().sort_index().to_dict()}")
    return df


def create_refill_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    FEATURE: refill_gap = expected_refills - refills_received

    WHY IT MATTERS:
        The absolute gap in refills (not just ratio) gives the model
        a count-based signal. Gap of 0 = perfectly adherent.
        Gap of 5+ = major non-adherence.
    """
    df = df.copy()

    if all(c in df.columns for c in ['expected_refills', 'refills_received']):
        df['refill_gap'] = (df['expected_refills'] - df['refills_received']).clip(0, None)
        print(f"   [OK] 'refill_gap' created | Mean gap: {df['refill_gap'].mean():.2f} refills")

    return df


def build_feature_matrix(raw_path: str, cleaned_path: str) -> pd.DataFrame:
    """
    Main feature engineering pipeline:
    1. Start from raw data (for unscaled values)
    2. Create all new features
    3. Merge with cleaned/encoded data
    4. Return final feature matrix
    """
    print("=" * 60)
    print("STEP 4: Feature Engineering")
    print("=" * 60)

    raw_df = pd.read_csv(raw_path)
    cleaned_df = pd.read_csv(cleaned_path)

    # ── Apply feature engineering on raw values ────────────────
    print("\nCreating healthcare-domain features...")
    raw_df = create_refill_ratio(raw_df)
    raw_df = create_financial_burden(raw_df)
    raw_df = create_age_group(raw_df)
    raw_df = create_medication_complexity(raw_df)
    raw_df = create_days_supply_category(raw_df)
    raw_df = create_refill_gap(raw_df)

    # ── Select engineered features to add ─────────────────────
    new_features = [
        'refill_ratio',
        'financial_burden',
        'age_group',
        'medication_complexity',
        'supply_category',
        'refill_gap',
    ]

    # Add patient_id for merging, then drop
    if 'patient_id' in raw_df.columns:
        eng_features = raw_df[['patient_id'] + new_features].copy()
    else:
        # If no patient_id, just concat by index
        for feat in new_features:
            cleaned_df[feat] = raw_df[feat].values
        print(f"\n[OK] Added {len(new_features)} new features to dataset")
        print(f"   Final shape: {cleaned_df.shape}")
        return cleaned_df

    # ── Merge engineered features with cleaned data ────────────
    # Re-add patient_id to cleaned if needed
    if 'patient_id' not in cleaned_df.columns:
        cleaned_df['patient_id'] = raw_df['patient_id'].values

    final_df = cleaned_df.merge(eng_features, on='patient_id', how='left')
    final_df = final_df.drop(columns=['patient_id'], errors='ignore')

    print(f"\n[OK] Feature engineering complete!")
    print(f"   Original features: {len(cleaned_df.columns)}")
    print(f"   New features added: {len(new_features)}")
    print(f"   Final feature count: {len(final_df.columns)}")
    print(f"   Final shape: {final_df.shape}")

    # ── Print feature importance summary ──────────────────────
    print("\nCorrelation of new features with adherence:")
    if 'adherent' in final_df.columns:
        for feat in new_features:
            if feat in final_df.columns:
                corr = final_df[feat].corr(final_df['adherent'])
                direction = "+" if corr > 0 else "-"
                print(f"   {direction} {feat:<25}: r = {corr:+.3f}")

    return final_df


def main():
    """Run the full feature engineering pipeline."""
    os.makedirs("data/processed", exist_ok=True)

    final_df = build_feature_matrix(RAW_PATH, CLEANED_PATH)

    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Featured dataset saved to '{OUTPUT_PATH}'")

    return final_df


if __name__ == "__main__":
    main()
