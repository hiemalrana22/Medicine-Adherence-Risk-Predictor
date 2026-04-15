"""
============================================================
preprocessing.py - Data Cleaning & Preparation
============================================================
PURPOSE:
    This script loads the raw dataset and performs all
    necessary cleaning steps before machine learning:
    - Handle missing values
    - Remove duplicates
    - Encode categorical variables
    - Scale numerical features
    - Detect and handle outliers

INPUT:  data/raw/medication_adherence.csv
OUTPUT: data/processed/cleaned_data.csv
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')


# ── CONFIGURATION ─────────────────────────────────────────────
RAW_DATA_PATH    = "data/raw/medication_adherence.csv"
PROCESSED_PATH   = "data/processed/cleaned_data.csv"


def load_data(path: str) -> pd.DataFrame:
    """
    Load raw CSV dataset into a pandas DataFrame.
    Also prints basic info so we can understand the data.
    """
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)

    if not os.path.exists(path):
        print(f"[WARN] File not found at '{path}'.")
        print("    Generating synthetic dataset for demonstration...")
        df = generate_synthetic_data()
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[OK] Synthetic dataset saved to '{path}'")
        return df

    df = pd.read_csv(path)
    print(f"[OK] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def generate_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic medication adherence data.

    This mirrors the structure of the Mendeley dataset:
    https://data.mendeley.com/datasets/zkp7sbbx64/2

    Features:
    - Patient demographics (age, gender)
    - Insurance/financial data (claim_amount, annual_contribution)
    - Prescription data (refills_received, expected_refills, days_supply)
    - Diagnosis info (chronic_condition, num_medications)
    - Target: adherent (1 = yes, 0 = no)
    """
    np.random.seed(seed)

    age           = np.random.randint(18, 85, n)
    gender        = np.random.choice(['Male', 'Female'], n, p=[0.48, 0.52])
    insurance_type= np.random.choice(['HMO', 'PPO', 'Medicare', 'Medicaid'], n,
                                     p=[0.30, 0.35, 0.20, 0.15])

    # Financial features
    annual_contribution = np.random.normal(3000, 800, n).clip(500, 8000)
    claim_amount        = np.random.normal(1200, 600, n).clip(50, 6000)

    # Prescription features
    expected_refills = np.random.randint(3, 13, n)
    refills_received = np.clip(
        expected_refills - np.random.poisson(1.5, n),
        0, expected_refills
    )
    days_supply      = np.random.choice([30, 60, 90], n, p=[0.5, 0.3, 0.2])

    # Clinical features
    chronic_condition = np.random.choice(
        ['Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 'None'],
        n, p=[0.22, 0.28, 0.15, 0.10, 0.25]
    )
    num_medications = np.random.randint(1, 8, n)

    # Adherence target (influenced by realistic factors)
    adherence_prob = (
        0.5
        + 0.15 * (refills_received / expected_refills)
        - 0.10 * (claim_amount / annual_contribution).clip(0, 1)
        - 0.08 * (age > 65).astype(float)
        + 0.05 * (days_supply == 90).astype(float)
        - 0.07 * (num_medications > 4).astype(float)
    ).clip(0.05, 0.95)

    adherent = np.random.binomial(1, adherence_prob)

    # Introduce some missing values (~3–5%) to simulate real-world data
    df = pd.DataFrame({
        'patient_id'         : range(1, n + 1),
        'age'                : age,
        'gender'             : gender,
        'insurance_type'     : insurance_type,
        'annual_contribution': annual_contribution.round(2),
        'claim_amount'       : claim_amount.round(2),
        'expected_refills'   : expected_refills,
        'refills_received'   : refills_received,
        'days_supply'        : days_supply,
        'chronic_condition'  : chronic_condition,
        'num_medications'    : num_medications,
        'adherent'           : adherent,
    })

    # Add realistic missingness
    for col in ['annual_contribution', 'claim_amount', 'chronic_condition']:
        mask = np.random.random(n) < 0.04
        df.loc[mask, col] = np.nan

    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Print a comprehensive summary of the dataset.
    This is STEP 2: Data Understanding.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Data Understanding")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

    print("\nColumn Names and Data Types:")
    print(df.dtypes.to_string())

    print("\nFirst 5 Rows:")
    print(df.head().to_string())

    print("\nNumerical Summary:")
    print(df.describe().round(2).to_string())

    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Count': missing, 'Percentage (%)': missing_pct})
    print(missing_df[missing_df['Count'] > 0].to_string())

    print("\nTarget Variable Distribution (adherent):")
    target_dist = df['adherent'].value_counts()
    print(target_dist.to_string())
    print(f"   → Class balance: {(target_dist[1]/len(df)*100):.1f}% adherent")

    print("\nFeature Descriptions:")
    descriptions = {
        'patient_id'         : 'Unique patient identifier (drop before modeling)',
        'age'                : 'Patient age in years',
        'gender'             : 'Patient gender (Male/Female)',
        'insurance_type'     : 'Type of insurance (HMO, PPO, Medicare, Medicaid)',
        'annual_contribution': 'Annual amount patient pays for insurance ($)',
        'claim_amount'       : 'Total insurance claim amount ($)',
        'expected_refills'   : 'Number of refills expected over treatment period',
        'refills_received'   : 'Actual number of refills the patient received',
        'days_supply'        : 'Days of medication supply per prescription',
        'chronic_condition'  : 'Primary chronic condition diagnosed',
        'num_medications'    : 'Number of concurrent medications',
        'adherent'           : 'TARGET: 1 = Adherent, 0 = Non-Adherent',
    }
    for col, desc in descriptions.items():
        print(f"   • {col:<22}: {desc}")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values separately for numerical and categorical columns.

    Strategy:
    - Numerical → fill with MEDIAN (robust to outliers)
    - Categorical → fill with MODE (most frequent value)
    """
    print("\n" + "=" * 60)
    print("STEP 3a: Handling Missing Values")
    print("=" * 60)

    df = df.copy()

    # Separate column types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Fill numerical columns with median
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   [OK] '{col}' → filled NaN with median ({median_val:.2f})")

    # Fill categorical columns with mode
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"   [OK] '{col}' → filled NaN with mode ('{mode_val}')")

    print(f"\n   Total missing after imputation: {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows to prevent data leakage and bias.
    """
    print("\n" + "=" * 60)
    print("STEP 3b: Removing Duplicates")
    print("=" * 60)

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"   Removed {before - after} duplicate rows ({before} → {after})")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and cap outliers using IQR (Interquartile Range) method.

    Why IQR?
    - Any value below Q1 - 1.5×IQR or above Q3 + 1.5×IQR is an outlier
    - We CAP (clip) rather than remove — preserving data size
    """
    print("\n" + "=" * 60)
    print("STEP 3c: Handling Outliers (IQR Capping)")
    print("=" * 60)

    df = df.copy()
    # Only apply to key numerical columns (not IDs or targets)
    outlier_cols = ['age', 'annual_contribution', 'claim_amount',
                    'expected_refills', 'refills_received', 'num_medications']

    for col in outlier_cols:
        if col not in df.columns:
            continue
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)

        if outliers_count > 0:
            print(f"   [OK] '{col}': {outliers_count} outliers capped "
                  f"[{lower:.1f}, {upper:.1f}]")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical text columns to numbers for ML models.

    Strategy:
    - Binary columns (gender) → Label Encoding (0/1)
    - Multi-class columns → One-Hot Encoding (avoids ordinal assumption)
    """
    print("\n" + "=" * 60)
    print("STEP 3d: Encoding Categorical Variables")
    print("=" * 60)

    df = df.copy()

    # Label encode gender (binary)
    le = LabelEncoder()
    df['gender_encoded'] = le.fit_transform(df['gender'])
    print(f"   [OK] 'gender' label-encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # One-hot encode multi-class categoricals
    ohe_cols = ['insurance_type', 'chronic_condition']
    for col in ohe_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            print(f"   [OK] '{col}' one-hot encoded → {list(dummies.columns)}")

    # Drop original text columns and patient_id (not useful for ML)
    df = df.drop(columns=['gender', 'insurance_type', 'chronic_condition',
                          'patient_id'], errors='ignore')

    return df


def scale_features(df: pd.DataFrame, target_col: str = 'adherent') -> pd.DataFrame:
    """
    Standardize numerical features using StandardScaler.

    ⚠️  BUG FIX (Data Leakage): Scaling is NO LONGER applied here.
    Previously, StandardScaler was fit on the entire dataset (train + test)
    before the train/test split, leaking test set statistics into training.

    Scaling is now done correctly inside train.py AFTER the train/test split:
      - Scaler is fit ONLY on X_train
      - Same scaler is applied (transform only) to X_test

    This function is kept for backward compatibility but returns df unchanged.
    """
    print("\n" + "=" * 60)
    print("STEP 3e: Feature Scaling (deferred to train.py — see BUG FIX note)")
    print("=" * 60)
    print("   [INFO] Scaling moved to train.py to prevent data leakage.")
    print("   [INFO] Scaler will be fit on X_train only, then applied to X_test.")
    return df


def main():
    """Run the full preprocessing pipeline."""
    os.makedirs("data/processed", exist_ok=True)

    # Load
    df = load_data(RAW_DATA_PATH)

    # Understand
    explore_data(df)

    # Clean
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = encode_categoricals(df)
    df = scale_features(df)

    # Save
    df.to_csv(PROCESSED_PATH, index=False)
    print("\n" + "=" * 60)
    print(f"[OK] Cleaned data saved to '{PROCESSED_PATH}'")
    print(f"   Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
