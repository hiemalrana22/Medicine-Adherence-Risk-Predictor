"""
============================================================
train.py - Machine Learning Model Training
============================================================
PURPOSE:
    Train and compare three ML models:
    1. Logistic Regression  (baseline, interpretable)
    2. Decision Tree        (visual, easy to explain)
    3. Random Forest        (main model, best performance)

    Also handles:
    - Train/test split (stratified to preserve class balance)
    - Class imbalance via SMOTE oversampling
    - Model saving for later evaluation

INPUT:  data/processed/featured_data.csv
OUTPUT: outputs/models/  (saved .pkl models)
        outputs/figures/  (EDA plots)
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Docker/scripts
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# ── CONFIGURATION ─────────────────────────────────────────────
DATA_PATH    = "data/processed/featured_data.csv"
MODEL_DIR    = "outputs/models"
FIGURES_DIR  = "outputs/figures"
TARGET_COL   = "adherent"
TEST_SIZE    = 0.2       # 80% train, 20% test
RANDOM_STATE = 42


def load_data(path: str):
    """Load the feature-engineered dataset."""
    df = pd.read_csv(path)
    print(f"[OK] Data loaded: {df.shape}")
    return df


def prepare_features(df: pd.DataFrame, target: str):
    """
    Separate features (X) from the target variable (y).
    Drop any columns that shouldn't be used for prediction.
    """
    # Drop target column
    X = df.drop(columns=[target], errors='ignore')
    y = df[target]

    # Drop any remaining ID-like columns
    drop_cols = [c for c in X.columns if 'id' in c.lower()]
    X.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Fill any remaining NaN with 0 (safety net)
    X = X.fillna(0)

    print(f"\nFeatures (X): {X.shape[1]} columns")
    print(f"Target (y): {y.value_counts().to_dict()}")

    return X, y


def split_data(X, y):
    """
    Split into training and test sets.

    WHY STRATIFIED SPLIT?
    - Ensures both classes are proportionally represented
    - Prevents a scenario where test set has only one class
    - Critical when classes are imbalanced
    """
    print("\n" + "=" * 60)
    print("Train/Test Split")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Preserve class distribution!
    )

    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set:     {X_test.shape[0]} samples")
    print(f"   Train adherence rate: {y_train.mean():.1%}")
    print(f"   Test adherence rate:  {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """
    Handle class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

    WHY SMOTE?
    - Medical datasets often have more adherent than non-adherent patients
    - A model trained on imbalanced data learns to predict the majority class
    - SMOTE creates synthetic samples for the minority class
    - Result: balanced training set → fairer model

    NOTE: Apply SMOTE ONLY on training data, never on test data
    (that would be data leakage!)

    Falls back to random oversampling if imblearn is not installed.
    """
    print("\n" + "=" * 60)
    print("Class Imbalance Handling")
    print("=" * 60)

    print(f"   Before resampling: {pd.Series(y_train).value_counts().to_dict()}")

    if SMOTE_AVAILABLE:
        print("   Using SMOTE oversampling...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    else:
        print("   imblearn not available — using random oversampling fallback...")
        # Simple random oversampling of minority class
        X_df = X_train.copy()
        X_df['__target__'] = y_train.values
        majority = X_df[X_df['__target__'] == 1]
        minority = X_df[X_df['__target__'] == 0]
        if len(minority) < len(majority):
            minority_upsampled = minority.sample(len(majority), replace=True,
                                                  random_state=RANDOM_STATE)
            balanced = pd.concat([majority, minority_upsampled])
        else:
            balanced = X_df
        balanced = balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        y_resampled = balanced['__target__']
        X_resampled = balanced.drop(columns=['__target__'])

    print(f"   After resampling:  {pd.Series(y_resampled).value_counts().to_dict()}")
    print(f"   Training size grew: {len(y_train)} → {len(y_resampled)}")

    return X_resampled, y_resampled


def run_eda(df: pd.DataFrame):
    """
    STEP 5: Exploratory Data Analysis
    Generate and save key visualization plots.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Exploratory Data Analysis")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams.update({'figure.dpi': 120, 'figure.facecolor': 'white'})

    # ── 1. Target Distribution ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Target Variable: Medication Adherence", fontsize=14, fontweight='bold')

    target_counts = df[TARGET_COL].value_counts()
    axes[0].bar(['Non-Adherent (0)', 'Adherent (1)'],
                [target_counts.get(0, 0), target_counts.get(1, 0)],
                color=['#E74C3C', '#27AE60'], edgecolor='black')
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate([target_counts.get(0, 0), target_counts.get(1, 0)]):
        axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

    axes[1].pie(
        [target_counts.get(0, 0), target_counts.get(1, 0)],
        labels=['Non-Adherent', 'Adherent'],
        colors=['#E74C3C', '#27AE60'],
        autopct='%1.1f%%',
        startangle=90
    )
    axes[1].set_title("Adherence Proportion")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/01_target_distribution.png", bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: 01_target_distribution.png")

    # ── 2. Refill Ratio Distribution by Adherence ──────────────
    if 'refill_ratio' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, color in [(0, '#E74C3C'), (1, '#27AE60')]:
            subset = df[df[TARGET_COL] == label]['refill_ratio']
            ax.hist(subset, bins=30, alpha=0.6, color=color,
                    label=f"{'Adherent' if label else 'Non-Adherent'}")
        ax.set_title("Refill Ratio Distribution by Adherence", fontsize=13, fontweight='bold')
        ax.set_xlabel("Refill Ratio (refills received / expected)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.axvline(df[df[TARGET_COL]==1]['refill_ratio'].mean(),
                   color='#27AE60', linestyle='--', label='Adherent mean')
        ax.axvline(df[df[TARGET_COL]==0]['refill_ratio'].mean(),
                   color='#E74C3C', linestyle='--', label='Non-Adherent mean')
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/02_refill_ratio_distribution.png", bbox_inches='tight')
        plt.close()
        print(f"   [OK] Saved: 02_refill_ratio_distribution.png")

    # ── 3. Financial Burden vs Adherence ──────────────────────
    if 'financial_burden' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        df.boxplot(column='financial_burden', by=TARGET_COL, ax=ax,
                   patch_artist=True)
        ax.set_title("Financial Burden by Adherence Status", fontsize=13, fontweight='bold')
        ax.set_xlabel("Adherent (0=No, 1=Yes)")
        ax.set_ylabel("Financial Burden (claim / contribution)")
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/03_financial_burden_adherence.png", bbox_inches='tight')
        plt.close()
        print(f"   [OK] Saved: 03_financial_burden_adherence.png")

    # ── 4. Correlation Heatmap ─────────────────────────────────
    num_df = df.select_dtypes(include=[np.number])
    # Limit to top columns for readability
    key_cols = [TARGET_COL, 'refill_ratio', 'financial_burden',
                'age_group', 'refill_gap', 'medication_complexity',
                'supply_category']
    key_cols = [c for c in key_cols if c in num_df.columns]

    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = num_df[key_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/04_correlation_heatmap.png", bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: 04_correlation_heatmap.png")

    # ── 5. Age Group vs Adherence ─────────────────────────────
    if 'age_group' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        age_adherence = df.groupby('age_group')[TARGET_COL].mean()
        bars = ax.bar(
            ['Young\n(18–35)', 'Adult\n(36–64)', 'Elderly\n(65+)'],
            age_adherence.values,
            color=['#3498DB', '#F39C12', '#9B59B6'],
            edgecolor='black'
        )
        ax.set_title("Adherence Rate by Age Group", fontsize=13, fontweight='bold')
        ax.set_ylabel("Adherence Rate")
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, age_adherence.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/05_age_group_adherence.png", bbox_inches='tight')
        plt.close()
        print(f"   [OK] Saved: 05_age_group_adherence.png")

    print("\n   All EDA plots saved to outputs/figures/")


def train_models(X_train, y_train, X_test, y_test):
    """
    STEP 6: Train three ML models and compare performance.

    Models:
    1. Logistic Regression – Simple, fast, interpretable baseline
    2. Decision Tree – Visualizable, easy to explain to stakeholders
    3. Random Forest – Ensemble of trees, usually best performance
    """
    print("\n" + "=" * 60)
    print("STEP 6: Training Machine Learning Models")
    print("=" * 60)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',  # Additional imbalance handling
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,              # Prevent overfitting
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,         # 100 trees
            max_depth=10,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1                 # Use all CPU cores
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n   Training {name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Cross-validation (5-fold) for robust accuracy estimate
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                    scoring='roc_auc', n_jobs=-1)

        # Test set predictions
        y_pred = model.predict(X_test)

        # Quick accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"   [OK] Test Accuracy: {acc:.3f} | CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        results[name] = {
            'model'    : model,
            'y_pred'   : y_pred,
            'cv_scores': cv_scores,
            'accuracy' : acc,
        }

    return results


def save_models(results: dict, feature_names: list):
    """Save trained models to disk for later evaluation."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, info in results.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump(info['model'], path)
        print(f"   [OK] Saved: {filename}")

    # Also save feature names (needed for evaluation)
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
    print(f"   [OK] Saved: feature_names.pkl")


def main():
    """Run the full training pipeline."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load data
    df = load_data(DATA_PATH)

    # EDA before modeling
    run_eda(df)

    # Prepare features
    X, y = prepare_features(df, TARGET_COL)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Handle class imbalance
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    # Train models
    results = train_models(X_train_resampled, y_train_resampled, X_test, y_test)

    # Save models
    print("\n" + "=" * 60)
    print("Saving Models")
    print("=" * 60)
    save_models(results, list(X.columns))

    # Save test data for evaluation
    test_data = X_test.copy()
    test_data['adherent'] = y_test.values
    test_data.to_csv("data/processed/test_data.csv", index=False)

    print("\n[OK] Training complete! Models saved to outputs/models/")

    return results, X_test, y_test


if __name__ == "__main__":
    main()
