"""
============================================================
app.py — Medication Adherence Risk Predictor (Live Demo)
============================================================
A single-screen, public-facing demo:

    Enter a patient's details  →  adherence-risk probability
                                →  the factors driving that risk

Deploy on Streamlit Community Cloud or Hugging Face Spaces and
share the link (see "Deploy" section in the README).

Run locally:
    streamlit run app.py

Design notes
------------
* The model is trained INLINE from the committed, feature-engineered
  dataset (`data/processed/featured_data.csv`) and cached. We do this
  instead of loading a pickled model so the demo never breaks from a
  scikit-learn version mismatch on a hosted runner.
* Random Forest is scale-invariant, so no scaler is needed here — the
  inline model matches the pipeline's Random Forest behaviour.
* "What's driving this prediction" uses a faithful local ablation: for
  each feature we reset it to the population median and measure how much
  the predicted probability moves. No SHAP dependency required.
* The dataset is SYNTHETIC (mirrors the Mendeley adherence dataset
  structure) — this is a methods demo, not medical advice.
============================================================
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# ── Paths ──────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
FEATURED    = os.path.join(ROOT, "data", "processed", "featured_data.csv")
METRICS_CSV = os.path.join(ROOT, "outputs", "reports", "model_metrics.csv")
TARGET_COL  = "adherent"

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Adherence Risk Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .hero-title { font-size: 2.1rem; font-weight: 800; color: #1a3a5c; margin-bottom: 0.1rem; }
    .hero-sub   { font-size: 1.02rem; color: #5a6b7b; margin-bottom: 1.0rem; }
    .pill { display:inline-block; background:#eef4f9; border:1px solid #cfe0ec;
            border-radius:999px; padding:3px 12px; font-size:0.8rem; color:#2f72a4; margin:2px 4px 2px 0;}
    .disclaimer { font-size:0.82rem; color:#8a8a8a; border-top:1px solid #eee; padding-top:10px; margin-top:24px;}
</style>
""", unsafe_allow_html=True)


# ── Train (or reuse) the model, cached across reruns ───────────
@st.cache_resource(show_spinner="Training the adherence model…")
def get_model_and_data():
    """
    Train a Random Forest inline from the feature-engineered dataset.

    Returns
    -------
    model        : fitted RandomForestClassifier
    feature_cols : list[str]  — exact column order the model expects
    medians      : pd.Series  — population medians (baseline for ablation)
    importances  : pd.Series  — global feature importances, sorted desc
    """
    df = pd.read_csv(FEATURED)

    # Mirror src/train.py::prepare_features so the demo matches the pipeline.
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    y = df[TARGET_COL]

    drop_cols = [c for c in X.columns if "id" in c.lower()]
    if "refill_ratio" in X.columns and "refill_gap" in X.columns:
        # Engineered features supersede the raw refill counts (avoids redundancy).
        drop_cols += ["expected_refills", "refills_received"]
    X = X.drop(columns=drop_cols, errors="ignore").fillna(0)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=15,
        min_samples_leaf=8, max_features="sqrt", random_state=42, n_jobs=-1,
    ).fit(X, y)

    medians = X.median()
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, list(X.columns), medians, importances


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_CSV):
        return pd.read_csv(METRICS_CSV)
    return pd.DataFrame()


def engineer_patient(age, gender, insurance, annual_contribution, claim_amount,
                     expected_refills, refills_received, days_supply, condition, num_meds):
    """Build the engineered feature dict for one patient (mirrors src/)."""
    refill_ratio          = min(refills_received / max(expected_refills, 1), 1.0)
    financial_burden      = min(claim_amount / max(annual_contribution, 1), 5.0)
    age_group             = 0 if age <= 35 else (1 if age <= 64 else 2)
    medication_complexity = 0 if num_meds <= 2 else (1 if num_meds <= 5 else 2)
    supply_category       = 0 if days_supply <= 30 else (1 if days_supply <= 60 else 2)
    refill_gap            = max(expected_refills - refills_received, 0)

    return {
        "age": age,
        "annual_contribution": annual_contribution,
        "claim_amount": claim_amount,
        "days_supply": days_supply,
        "num_medications": num_meds,
        "gender_encoded": 1 if gender == "Male" else 0,
        "insurance_type_Medicaid": 1 if insurance == "Medicaid" else 0,
        "insurance_type_Medicare": 1 if insurance == "Medicare" else 0,
        "insurance_type_PPO": 1 if insurance == "PPO" else 0,
        "chronic_condition_Diabetes": 1 if condition == "Diabetes" else 0,
        "chronic_condition_Heart Disease": 1 if condition == "Heart Disease" else 0,
        "chronic_condition_Hypertension": 1 if condition == "Hypertension" else 0,
        "chronic_condition_None": 1 if condition == "None" else 0,
        "refill_ratio": refill_ratio,
        "financial_burden": financial_burden,
        "age_group": age_group,
        "medication_complexity": medication_complexity,
        "supply_category": supply_category,
        "refill_gap": refill_gap,
    }


# Human-readable labels for the driver chart
PRETTY = {
    "refill_ratio": "Refill ratio",
    "refill_gap": "Refill gap (missed refills)",
    "financial_burden": "Financial burden",
    "claim_amount": "Claim amount",
    "annual_contribution": "Annual contribution",
    "age": "Age",
    "age_group": "Age group",
    "num_medications": "Number of medications",
    "medication_complexity": "Medication complexity",
    "supply_category": "Days-supply category",
    "days_supply": "Days supply",
    "gender_encoded": "Gender",
}


def local_drivers(model, feature_cols, medians, patient_row, top_n=6):
    """
    Faithful local explanation via single-feature ablation.

    For each feature we replace the patient's value with the population
    median and measure the change in predicted P(adherent). A large drop
    means that feature was pushing the patient TOWARD adherence; a large
    rise means it was pushing them toward NON-adherence (risk).

    Returns a DataFrame sorted by absolute impact (descending).
    """
    # Work in float so resetting an int column to a float median is clean.
    patient_f = patient_row[feature_cols].astype(float)
    base_prob = model.predict_proba(patient_f)[0][1]

    rows = []
    for col in feature_cols:
        if patient_f.iloc[0][col] == float(medians[col]):
            continue  # at baseline → no contribution to attribute
        counter = patient_f.copy()
        counter.iloc[0, counter.columns.get_loc(col)] = float(medians[col])
        cf_prob = model.predict_proba(counter)[0][1]
        delta = base_prob - cf_prob  # +ve → feature raised adherence prob
        if abs(delta) < 1e-4:
            continue
        rows.append({
            "feature": PRETTY.get(col, col),
            "delta": delta,
            "direction": "Supports adherence" if delta > 0 else "Raises risk",
        })

    drivers = pd.DataFrame(rows)
    if drivers.empty:
        return drivers, base_prob
    drivers["abs"] = drivers["delta"].abs()
    drivers = drivers.sort_values("abs", ascending=False).head(top_n)
    return drivers, base_prob


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">💊 Medication Adherence Risk Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Enter a patient profile to estimate their probability of '
    'adhering to medication — and see <b>which factors drive the risk</b>.</div>',
    unsafe_allow_html=True,
)

model, feature_cols, medians, importances = get_model_and_data()
metrics_df = load_metrics()

# Model performance strip (held-out test set)
if not metrics_df.empty and "Model" in metrics_df.columns:
    rf = metrics_df[metrics_df["Model"] == "Random Forest"]
    rf = rf.iloc[0] if not rf.empty else metrics_df.iloc[metrics_df["ROC-AUC"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{rf.get('ROC-AUC', float('nan')):.3f}", help="Held-out test set")
    c2.metric("PR-AUC (at-risk)", f"{rf.get('PR-AUC (At-Risk)', float('nan')):.3f}",
              help="Average precision for detecting non-adherent patients")
    c3.metric("At-risk recall", f"{rf.get('At-Risk Recall', float('nan')):.3f}",
              help="Share of truly non-adherent patients the model flags")
    c4.metric("Accuracy", f"{rf.get('Accuracy', float('nan')):.3f}", help="Held-out test set")

# ── Inputs (sidebar) ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧾 Patient profile")
    st.caption("Adjust the inputs, then read the prediction on the right.")

    st.markdown("**Demographics**")
    age       = st.slider("Age", 18, 85, 45)
    gender    = st.selectbox("Gender", ["Male", "Female"])
    insurance = st.selectbox("Insurance type", ["HMO", "PPO", "Medicare", "Medicaid"])

    st.markdown("**Finances**")
    annual_contribution = st.number_input("Annual contribution ($)", 500, 8000, 3000, step=100)
    claim_amount        = st.number_input("Claim amount ($)", 50, 7000, 1200, step=50)

    st.markdown("**Prescription**")
    expected_refills = st.slider("Expected refills", 3, 12, 8)
    refills_received = st.slider("Refills received", 0, 12, 6)
    days_supply      = st.selectbox("Days supply", [30, 60, 90], index=0)

    st.markdown("**Clinical**")
    condition = st.selectbox("Chronic condition", ["None", "Diabetes", "Hypertension", "Asthma", "Heart Disease"])
    num_meds  = st.slider("Number of medications", 1, 7, 3)

    st.markdown("---")
    st.caption("Tip: drop *refills received* well below *expected* to watch risk climb.")

# ── Predict ────────────────────────────────────────────────────
patient = engineer_patient(age, gender, insurance, annual_contribution, claim_amount,
                           expected_refills, refills_received, days_supply, condition, num_meds)
patient_row = pd.DataFrame([{f: patient.get(f, 0) for f in feature_cols}])

prob_adherent = model.predict_proba(patient_row[feature_cols])[0][1]
risk = 1 - prob_adherent
is_adherent = prob_adherent >= 0.5

left, right = st.columns([1, 1.25])

with left:
    st.markdown("#### Prediction")
    if is_adherent:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#11998e,#38ef7d);padding:26px;
                    border-radius:16px;text-align:center;color:white;">
            <div style="font-size:3rem;">✅</div>
            <div style="font-size:1.6rem;font-weight:800;">LIKELY ADHERENT</div>
            <div style="font-size:1.1rem;">Adherence probability: {prob_adherent:.0%}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#cb2d3e,#ef473a);padding:26px;
                    border-radius:16px;text-align:center;color:white;">
            <div style="font-size:3rem;">⚠️</div>
            <div style="font-size:1.6rem;font-weight:800;">AT RISK OF NON-ADHERENCE</div>
            <div style="font-size:1.1rem;">Non-adherence risk: {risk:.0%}</div>
        </div>""", unsafe_allow_html=True)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        number={"suffix": "%", "font": {"size": 30}},
        title={"text": "Non-adherence risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#cb2d3e" if not is_adherent else "#11998e"},
            "steps": [
                {"range": [0, 40], "color": "#e8f5e9"},
                {"range": [40, 60], "color": "#fff3cd"},
                {"range": [60, 100], "color": "#fdecea"},
            ],
            "threshold": {"line": {"color": "gray", "width": 3}, "thickness": 0.75, "value": 50},
        },
    ))
    gauge.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20))
    st.plotly_chart(gauge, use_container_width=True)

with right:
    st.markdown("#### What's driving this prediction?")
    drivers, _ = local_drivers(model, feature_cols, medians, patient_row)
    if drivers.empty:
        st.info("This patient sits near the population average on every feature — no single factor stands out.")
    else:
        # Green = pushes toward adherence, Red = pushes toward risk
        drivers = drivers.sort_values("delta")
        colors = ["#cb2d3e" if d < 0 else "#11998e" for d in drivers["delta"]]
        fig = go.Figure(go.Bar(
            x=drivers["delta"] * 100,
            y=drivers["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{d*100:+.1f} pts" for d in drivers["delta"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.1f} pts vs. typical patient<extra></extra>",
        ))
        fig.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=30),
            xaxis_title="Effect on adherence probability (percentage points)",
            plot_bgcolor="white",
        )
        fig.add_vline(x=0, line_width=1, line_color="#999")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🟢 pushes toward adherence • 🔴 raises non-adherence risk — "
                   "each bar = change vs. a median patient (local ablation).")

# ── Context: global drivers ────────────────────────────────────
with st.expander("📊 Which features matter most overall? (global importance)"):
    top = importances.head(8).iloc[::-1]
    fig2 = go.Figure(go.Bar(
        x=top.values, y=[PRETTY.get(i, i) for i in top.index],
        orientation="h", marker_color="#2f72a4",
        text=[f"{v:.3f}" for v in top.values], textposition="outside",
    ))
    fig2.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=30),
                       xaxis_title="Random Forest feature importance", plot_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        "**Refill ratio** is the strongest single predictor, followed by the "
        "**refill gap** and **financial-burden** signals — consistent with the "
        "clinical literature on adherence."
    )

st.markdown(
    '<div class="disclaimer">⚠️ <b>Demo only — not medical advice.</b> Trained on a '
    'synthetic dataset that mirrors the structure of the Mendeley medication-adherence '
    'dataset. Metrics shown are from a held-out test set. Built by '
    '<a href="https://github.com/hiemalrana22/Medicine-Adherence-Risk-Predictor">this repository</a>.'
    '</div>',
    unsafe_allow_html=True,
)
