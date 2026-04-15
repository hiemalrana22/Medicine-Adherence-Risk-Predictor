"""
============================================================
evaluate.py - Model Evaluation & Power BI Export
============================================================
PURPOSE:
    Load saved models and evaluate them thoroughly with:
    - Classification metrics (Accuracy, Precision, Recall, F1)
    - ROC-AUC scores and ROC curves
    - Confusion matrices
    - Feature importance (Random Forest)
    - Export final predictions CSV for Power BI

    WHY RECALL IS CRITICAL IN HEALTHCARE:
    In medical settings, False Negatives are very dangerous.
    If a patient IS non-adherent but our model predicts they ARE
    adherent → they get no intervention → health outcomes worsen.
    Recall = TP / (TP + FN) → minimizes these missed cases.

INPUT:  outputs/models/*.pkl, data/processed/test_data.csv
OUTPUT: outputs/figures/ (evaluation plots)
        outputs/reports/final_predictions.csv (Power BI)
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)


# ── CONFIGURATION ─────────────────────────────────────────────
MODEL_DIR   = "outputs/models"
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"
TEST_PATH   = "data/processed/test_data.csv"
RAW_PATH    = "data/raw/medication_adherence.csv"
TARGET_COL  = "adherent"


def load_models_and_data():
    """Load all saved models and test data."""
    print("=" * 60)
    print("STEP 7: Model Evaluation")
    print("=" * 60)

    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree"      : "decision_tree.pkl",
        "Random Forest"      : "random_forest.pkl",
        "Gradient Boosting"  : "gradient_boosting.pkl",   # New model added
    }

    for name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
                print(f"   [OK] Loaded: {filename}")
            except Exception as exc:
                print(f"   [WARN] Skipped {filename}: {exc}")

    return models, X_test, y_test


def compute_metrics(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Compute all evaluation metrics for each model.

    Metrics Explained:
    - Accuracy:  % of all predictions that are correct
    - Precision: Of predicted adherent, how many truly are?
    - Recall:    Of truly non-adherent, how many did we catch?
    - F1 Score:  Harmonic mean of Precision and Recall
    - ROC-AUC:   Model's ability to separate classes (1.0 = perfect)
    """
    print("\n" + "─" * 60)
    print("Classification Metrics (all models)")
    print("─" * 60)

    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

        results.append({
            'Model'    : name,
            'Accuracy' : round(acc, 4),
            'Precision': round(prec, 4),
            'Recall'   : round(rec, 4),
            'F1 Score' : round(f1, 4),
            'ROC-AUC'  : round(auc, 4),
        })

        print(f"\n   {name}")
        print(f"      Accuracy:  {acc:.4f}")
        print(f"      Precision: {prec:.4f}")
        print(f"      Recall:    {rec:.4f}  ← Most important in healthcare!")
        print(f"      F1 Score:  {f1:.4f}")
        print(f"      ROC-AUC:   {auc:.4f}")

        print(f"\n      Classification Report:")
        print(classification_report(y_test, y_pred,
              target_names=['Non-Adherent', 'Adherent']))

    metrics_df = pd.DataFrame(results)
    return metrics_df


def plot_confusion_matrices(models: dict, X_test, y_test):
    """
    Plot confusion matrices for all models side by side.

    Confusion Matrix:
    ┌──────────────────┬──────────────┬──────────────┐
    │                  │ Pred: 0      │ Pred: 1      │
    ├──────────────────┼──────────────┼──────────────┤
    │ Actual: 0        │ TN (good)    │ FP (okay)    │
    │ Actual: 1        │ FN (bad!)    │ TP (good)    │
    └──────────────────┴──────────────┴──────────────┘
    FN = False Negative = Patient is non-adherent but we
         predicted adherent → DANGEROUS in healthcare!
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt='d', ax=ax,
            cmap='Blues',
            xticklabels=['Non-Adherent', 'Adherent'],
            yticklabels=['Non-Adherent', 'Adherent'],
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontweight='bold')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # Highlight false negatives
        tn, fp, fn, tp = cm.ravel()
        ax.set_title(
            f"{name}\nConfusion Matrix\n"
            f"(FN={fn} — missed non-adherent patients)",
            fontsize=10, fontweight='bold'
        )

    plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/06_confusion_matrices.png", bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: 06_confusion_matrices.png")


def plot_roc_curves(models: dict, X_test, y_test):
    """
    Plot ROC (Receiver Operating Characteristic) curves.

    ROC Curve shows the trade-off between:
    - True Positive Rate (Recall) on Y-axis
    - False Positive Rate on X-axis

    A model with AUC=0.5 is no better than random guessing.
    A model with AUC=1.0 is perfect.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = sns.color_palette("tab10", n_colors=max(1, len(models)))

    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.500)')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/07_roc_curves.png", bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: 07_roc_curves.png")


def plot_metrics_comparison(metrics_df: pd.DataFrame):
    """
    Bar chart comparing all models across all metrics.
    Makes it easy to pick the best model for deployment.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    x = np.arange(len(metrics))
    n_models = max(1, len(metrics_df))
    width = 0.8 / n_models
    colors = sns.color_palette("tab10", n_colors=n_models)
    center_offset = (n_models - 1) / 2

    for i, (_, row) in enumerate(metrics_df.iterrows()):
        offset = (i - center_offset) * width
        bars = ax.bar(
            x + offset,
            [row[m] for m in metrics],
            width,
            label=row['Model'],
            color=colors[i],
            edgecolor='black',
            alpha=0.85
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='0.80 threshold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/08_model_comparison.png", bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: 08_model_comparison.png")


def plot_feature_importance(model, feature_names: list):
    """
    STEP 8: Model Interpretation — Feature Importance

    Random Forest gives us feature importance scores.
    These tell us WHICH features most influence the prediction.

    Higher importance = feature has more power in the decision.
    """
    print("\n" + "=" * 60)
    print("STEP 8: Feature Importance (Random Forest)")
    print("=" * 60)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Top 15 features
    top_n = min(15, len(feature_names))
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]

    print("\n   Top Feature Importances:")
    for rank, (feat, imp) in enumerate(zip(top_features, top_importances), 1):
        bar = "█" * int(imp * 100)
        print(f"   {rank:2d}. {feat:<30} {imp:.4f}  {bar}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]

    bars = ax.barh(range(top_n), top_importances[::-1], color=colors[::-1],
                   edgecolor='black', alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance Score", fontsize=12)
    ax.set_title("Random Forest — Feature Importance\n(Top Predictors of Medication Adherence)",
                 fontsize=13, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, top_importances[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/09_feature_importance.png", bbox_inches='tight')
    plt.close()
    print(f"\n   [OK] Saved: 09_feature_importance.png")


def export_for_powerbi(models: dict, X_test, y_test):
    """
    STEP 10: Export final dataset with predictions for Power BI.

    The CSV will contain:
    - All features
    - Actual adherence label
    - Predicted adherence (Random Forest)
    - Prediction probability (confidence score)

    This CSV is loaded directly into Power BI Desktop.
    """
    print("\n" + "=" * 60)
    print("STEP 10: Exporting for Power BI")
    print("=" * 60)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load raw data for richer Power BI context
    raw_df = pd.read_csv(RAW_PATH) if os.path.exists(RAW_PATH) else pd.DataFrame()

    # Get Random Forest predictions (best model)
    rf_model = models.get("Random Forest")
    if rf_model is None:
        print("   [WARN] Random Forest model not found. Using first available model.")
        rf_model = list(models.values())[0]

    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    # Build export DataFrame
    export_df = X_test.copy().reset_index(drop=True)
    export_df['actual_adherence']       = y_test.values
    export_df['predicted_adherence']    = y_pred
    export_df['adherence_probability']  = y_prob.round(4)
    export_df['correct_prediction']     = (y_test.values == y_pred).astype(int)

    # Add interpretable labels
    export_df['adherence_label']    = export_df['actual_adherence'].map({1: 'Adherent', 0: 'Non-Adherent'})
    export_df['prediction_label']   = export_df['predicted_adherence'].map({1: 'Adherent', 0: 'Non-Adherent'})
    export_df['confidence']         = export_df['adherence_probability'].apply(
        lambda p: 'High' if p > 0.75 or p < 0.25
                  else 'Medium' if p > 0.60 or p < 0.40
                  else 'Low'
    )

    # Save
    output_path = f"{REPORTS_DIR}/final_predictions.csv"
    export_df.to_csv(output_path, index=False)

    print(f"   [OK] Saved: {output_path}")
    print(f"   Rows: {len(export_df)} | Columns: {len(export_df.columns)}")
    print(f"\n   Power BI Dashboard Suggestions:")
    print("   ─────────────────────────────────────────")
    print("   Page 1 – Overview")
    print("     • KPI Card: Total Patients")
    print("     • KPI Card: Overall Adherence Rate (%)")
    print("     • KPI Card: Model Accuracy")
    print("     • Bar Chart: Adherent vs Non-Adherent count")
    print()
    print("   Page 2 – Demographics")
    print("     • Bar Chart: Adherence by Age Group")
    print("     • Donut: Gender distribution")
    print("     • Stacked Bar: Insurance Type vs Adherence")
    print()
    print("   Page 3 – Financial Analysis")
    print("     • Scatter: Financial Burden vs Adherence Probability")
    print("     • Box Plot: Claim Amount by Adherence")
    print("     • Line: Contribution vs Claim by Age Group")
    print()
    print("   Page 4 – Model Insights")
    print("     • Bar Chart: Feature Importance (top 10)")
    print("     • Gauge: Model Confidence distribution")
    print("     • Matrix: Confusion Matrix visualization")
    print("     • Slicer: Filter by adherence / prediction / confidence")

    return export_df


def main():
    """Run the full evaluation pipeline."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load
    models, X_test, y_test = load_models_and_data()

    if not models:
        print("[ERROR] No models found. Please run train.py first.")
        return

    # Compute metrics
    metrics_df = compute_metrics(models, X_test, y_test)

    print("\n" + "─" * 60)
    print("Summary Table:")
    print("─" * 60)
    print(metrics_df.to_string(index=False))

    # Best model
    best_model_name = metrics_df.loc[metrics_df['ROC-AUC'].idxmax(), 'Model']
    best_auc = metrics_df['ROC-AUC'].max()
    print(f"\nBest Model: {best_model_name} (ROC-AUC = {best_auc:.4f})")

    # Plots
    print("\n" + "─" * 60)
    print("Generating Evaluation Plots")
    print("─" * 60)
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_metrics_comparison(metrics_df)

    # Feature importance
    feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")
    feature_names = (
        joblib.load(feature_names_path)
        if os.path.exists(feature_names_path)
        else list(X_test.columns)
    )
    if "Random Forest" in models:
        plot_feature_importance(models["Random Forest"], feature_names)

    # Export for Power BI
    export_for_powerbi(models, X_test, y_test)

    # Save metrics summary
    metrics_df.to_csv(f"{REPORTS_DIR}/model_metrics.csv", index=False)

    print("\n" + "=" * 60)
    print("[OK] Evaluation Complete!")
    print("   Plots → outputs/figures/")
    print("   Power BI CSV → outputs/reports/final_predictions.csv")
    print("   Metrics → outputs/reports/model_metrics.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
