"""
Generate a static preview of the live demo's output for the README.

This renders the SAME model and the SAME local-ablation explanation that
app.py shows, for one example at-risk patient — saved as a PNG so the README
has a real, reproducible illustration of the demo (no browser needed).

Run:  python scripts/make_demo_screenshot.py
Out:  docs/screenshots/live_demo_preview.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
FEATURED = os.path.join(ROOT, "data", "processed", "featured_data.csv")
OUT_DIR = os.path.join(ROOT, "docs", "screenshots")

PRETTY = {
    "refill_ratio": "Refill ratio", "refill_gap": "Refill gap",
    "financial_burden": "Financial burden", "claim_amount": "Claim amount",
    "annual_contribution": "Annual contribution", "age": "Age",
    "num_medications": "Number of medications", "medication_complexity": "Med. complexity",
    "supply_category": "Days-supply category",
}


def train():
    df = pd.read_csv(FEATURED)
    X = df.drop(columns=["adherent"], errors="ignore")
    y = df["adherent"]
    drop = [c for c in X.columns if "id" in c.lower()]
    if "refill_ratio" in X.columns and "refill_gap" in X.columns:
        drop += ["expected_refills", "refills_received"]
    X = X.drop(columns=drop, errors="ignore").fillna(0)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=8,
                                   max_features="sqrt", random_state=42, n_jobs=-1).fit(X, y)
    return model, list(X.columns), X.median()


def drivers(model, cols, medians, row, top_n=6):
    rf = row[cols].astype(float)
    base = model.predict_proba(rf)[0][1]
    out = []
    for c in cols:
        if rf.iloc[0][c] == float(medians[c]):
            continue
        cf = rf.copy()
        cf.iloc[0, cf.columns.get_loc(c)] = float(medians[c])
        d = base - model.predict_proba(cf)[0][1]
        if abs(d) > 1e-4:
            out.append((PRETTY.get(c, c), d))
    out.sort(key=lambda t: abs(t[1]), reverse=True)
    return base, out[:top_n]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, cols, medians = train()

    # Example at-risk patient: elderly, low refills, high financial burden
    patient = {
        "age": 72, "annual_contribution": 1000, "claim_amount": 3200, "days_supply": 30,
        "num_medications": 6, "gender_encoded": 0, "insurance_type_Medicaid": 1,
        "insurance_type_Medicare": 0, "insurance_type_PPO": 0,
        "chronic_condition_Diabetes": 1, "chronic_condition_Heart Disease": 0,
        "chronic_condition_Hypertension": 0, "chronic_condition_None": 0,
        "refill_ratio": 2 / 10, "financial_burden": min(3200 / 1000, 5),
        "age_group": 2, "medication_complexity": 2, "supply_category": 0, "refill_gap": 8,
    }
    row = pd.DataFrame([{c: patient.get(c, 0) for c in cols}])
    prob_adherent, drv = drivers(model, cols, medians, row)
    risk = 1 - prob_adherent

    fig = plt.figure(figsize=(11, 4.6))
    fig.suptitle("Medication Adherence Risk Predictor  —  live demo (app.py)",
                 fontsize=14, fontweight="bold", x=0.5, y=0.99)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.35,
                          left=0.04, right=0.97, top=0.84, bottom=0.12)

    # ── Left: verdict + risk ──────────────────────────────────────────
    axl = fig.add_subplot(gs[0, 0]); axl.axis("off")
    axl.add_patch(FancyBboxPatch((0.05, 0.55), 0.9, 0.38, boxstyle="round,pad=0.02",
                                 linewidth=0, facecolor="#cb2d3e"))
    axl.text(0.5, 0.84, "!  AT RISK OF NON-ADHERENCE", ha="center", va="center",
             color="white", fontsize=12.5, fontweight="bold")
    axl.text(0.5, 0.66, f"Non-adherence risk: {risk:.0%}", ha="center", va="center",
             color="white", fontsize=11)
    # risk meter
    axl.add_patch(plt.Rectangle((0.05, 0.32), 0.9, 0.08, color="#eee"))
    axl.add_patch(plt.Rectangle((0.05, 0.32), 0.9 * risk, 0.08, color="#cb2d3e"))
    axl.text(0.05, 0.20, "Example patient: age 72 · Medicaid · 2/10 refills · "
             "claim 3.2× contribution · 6 meds", fontsize=8.5, color="#444")
    axl.set_xlim(0, 1); axl.set_ylim(0, 1)

    # ── Right: driver bars ────────────────────────────────────────────
    axr = fig.add_subplot(gs[0, 1])
    labels = [d[0] for d in drv][::-1]
    vals = [d[1] * 100 for d in drv][::-1]
    colors = ["#11998e" if v > 0 else "#cb2d3e" for v in vals]
    axr.barh(labels, vals, color=colors)
    axr.axvline(0, color="#999", lw=1)
    axr.set_title("What's driving this prediction?", fontsize=11.5, fontweight="bold")
    axr.set_xlabel("Effect on adherence probability (percentage points)", fontsize=9)
    for y, v in enumerate(vals):
        axr.text(v + (0.6 if v >= 0 else -0.6), y, f"{v:+.1f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=8.5)
    axr.margins(x=0.18)
    axr.tick_params(labelsize=9)

    out = os.path.join(OUT_DIR, "live_demo_preview.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved {out}  (risk={risk:.0%}, drivers={[d[0] for d in drv]})")


if __name__ == "__main__":
    main()
