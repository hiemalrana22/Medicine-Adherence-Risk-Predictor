"""
============================================================
dashboard.py — Medical Adherence Predictor
Interactive Streamlit Dashboard (replaces Grafana)
============================================================
Run inside Docker:  streamlit run app/dashboard.py --server.port 8501
Open browser at:    http://localhost:8501
============================================================
"""

import os
import io
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "outputs", "figures")
MODELS_DIR  = os.path.join(ROOT, "outputs", "models")
REPORTS_DIR = os.path.join(ROOT, "outputs", "reports")
DATA_DIR    = os.path.join(ROOT, "data", "processed")
RAW_DIR     = os.path.join(ROOT, "data", "raw")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Adherence Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a3a5c;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 18px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .kpi-green  { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .kpi-blue   { background: linear-gradient(135deg, #2980b9, #6dd5fa); }
    .kpi-orange { background: linear-gradient(135deg, #f7971e, #ffd200); }
    .kpi-red    { background: linear-gradient(135deg, #cb2d3e, #ef473a); }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a3a5c;
        border-left: 4px solid #667eea;
        padding-left: 10px;
        margin: 20px 0 12px 0;
    }
    .pipeline-badge {
        display: inline-block;
        background: #e8f4f8;
        border: 1px solid #b8d4e3;
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 0.82rem;
        color: #2980b9;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: safe load ──────────────────────────────────────────
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_image(path: str):
    if os.path.exists(path):
        return Image.open(path)
    return None


# ── Data loading ───────────────────────────────────────────────
metrics_df  = load_csv(os.path.join(REPORTS_DIR, "model_metrics.csv"))
preds_df    = load_csv(os.path.join(REPORTS_DIR, "final_predictions.csv"))
raw_df      = load_csv(os.path.join(RAW_DIR,     "medication_adherence.csv"))
featured_df = load_csv(os.path.join(DATA_DIR,    "featured_data.csv"))

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/pill.png", width=72)
    st.markdown("## 💊 Med Adherence Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📊 EDA & Data",
         "🤖 Model Results",
         "📈 ROC & Metrics",
         "🔍 Predictions Browser",
         "🔮 Live Predictor"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**Pipeline Steps**")
    for step in [
        "1 · Preprocessing",
        "2 · Feature Engineering",
        "3 · Model Training",
        "4 · Evaluation",
    ]:
        st.markdown(f'<span class="pipeline-badge">{step}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built with Streamlit + Plotly")


# ═══════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="main-header">💊 Medical Adherence Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">End-to-end ML pipeline for predicting medication adherence · Interactive Dashboard</div>', unsafe_allow_html=True)

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)

    total_patients = len(raw_df) if not raw_df.empty else "—"
    adherence_rate = f"{raw_df['adherent'].mean()*100:.1f}%" if not raw_df.empty and 'adherent' in raw_df.columns else "—"

    best_auc = "—"
    best_model = "—"
    best_acc = "—"
    if not metrics_df.empty and 'ROC-AUC' in metrics_df.columns:
        idx = metrics_df['ROC-AUC'].idxmax()
        best_auc   = f"{metrics_df.loc[idx, 'ROC-AUC']:.3f}"
        best_model = metrics_df.loc[idx, 'Model']
        best_acc   = f"{metrics_df.loc[idx, 'Accuracy']:.1%}"

    with col1:
        st.markdown(f"""
        <div class="metric-card kpi-blue">
            <div class="metric-value">{total_patients}</div>
            <div class="metric-label">Total Patients</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card kpi-green">
            <div class="metric-value">{adherence_rate}</div>
            <div class="metric-label">Adherence Rate</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card kpi-orange">
            <div class="metric-value">{best_auc}</div>
            <div class="metric-label">Best ROC-AUC ({best_model})</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card kpi-red">
            <div class="metric-value">{best_acc}</div>
            <div class="metric-label">Best Accuracy</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project Architecture
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-title">📋 Project Pipeline</div>', unsafe_allow_html=True)
        pipeline_data = {
            "Step": ["1. Preprocessing", "2. Feature Engineering", "3. Training", "4. Evaluation"],
            "Script": ["src/preprocessing.py", "src/feature_engineering.py", "src/train.py", "src/evaluate.py"],
            "Output": ["cleaned_data.csv", "featured_data.csv", "4 model .pkl files", "Charts + CSV"],
        }
        st.dataframe(pd.DataFrame(pipeline_data), hide_index=True, use_container_width=True)

        st.markdown('<div class="section-title">🗂️ Dataset Overview</div>', unsafe_allow_html=True)
        if not raw_df.empty:
            st.dataframe(raw_df.head(6), hide_index=True, use_container_width=True)
        else:
            st.info("Run the pipeline first to generate data.")

    with col_b:
        st.markdown('<div class="section-title">📊 Target Distribution</div>', unsafe_allow_html=True)
        if not raw_df.empty and 'adherent' in raw_df.columns:
            counts = raw_df['adherent'].value_counts().reset_index()
            counts.columns = ['Adherent', 'Count']
            counts['Label'] = counts['Adherent'].map({1: 'Adherent', 0: 'Non-Adherent'})

            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=counts['Label'],
                values=counts['Count'],
                hole=0.45,
                marker_colors=['#27AE60', '#E74C3C'],
                textinfo='label+percent+value',
                textfont_size=13,
            ))
            fig.update_layout(
                title="Medication Adherence Distribution",
                height=320,
                margin=dict(t=40, b=10, l=10, r=10),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feature count
        if not featured_df.empty:
            st.metric("Feature Columns", featured_df.shape[1] - 1)
            st.metric("Training Rows", featured_df.shape[0])

    # Model performance summary table
    st.markdown('<div class="section-title">🤖 Model Performance Summary</div>', unsafe_allow_html=True)
    if not metrics_df.empty:
        styled = metrics_df.copy()
        st.dataframe(styled, hide_index=True, use_container_width=True)
    else:
        st.info("Run the ML pipeline to generate model results.")


# ═══════════════════════════════════════════════════════════════
# PAGE 2: EDA & DATA
# ═══════════════════════════════════════════════════════════════
elif page == "📊 EDA & Data":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if raw_df.empty:
        st.error("No data found. Please run the pipeline first.")
        st.stop()

    # ── Tabs ───────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribution", "Age Analysis", "Financial", "Correlation", "Raw Data"
    ])

    with tab1:
        st.markdown('<div class="section-title">Refill Ratio Distribution</div>', unsafe_allow_html=True)
        if not featured_df.empty and 'refill_ratio' in featured_df.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    featured_df, x='refill_ratio',
                    color='adherent',
                    color_discrete_map={0: '#E74C3C', 1: '#27AE60'},
                    labels={'adherent': 'Adherent', 'refill_ratio': 'Refill Ratio'},
                    title="Refill Ratio by Adherence Status",
                    nbins=30,
                    barmode='overlay',
                    opacity=0.7,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = px.box(
                    featured_df, x='adherent', y='refill_ratio',
                    color='adherent',
                    color_discrete_map={0: '#E74C3C', 1: '#27AE60'},
                    labels={'adherent': 'Adherent (0=No, 1=Yes)', 'refill_ratio': 'Refill Ratio'},
                    title="Refill Ratio Box Plot",
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Insurance type
        if 'insurance_type' in raw_df.columns and 'adherent' in raw_df.columns:
            st.markdown('<div class="section-title">Adherence by Insurance Type</div>', unsafe_allow_html=True)
            ins_data = raw_df.groupby('insurance_type')['adherent'].agg(['mean', 'count']).reset_index()
            ins_data.columns = ['Insurance Type', 'Adherence Rate', 'Patient Count']
            ins_data['Adherence Rate (%)'] = (ins_data['Adherence Rate'] * 100).round(1)

            fig3 = px.bar(
                ins_data, x='Insurance Type', y='Adherence Rate (%)',
                color='Adherence Rate (%)',
                color_continuous_scale='RdYlGn',
                title="Adherence Rate by Insurance Type",
                text='Adherence Rate (%)',
            )
            fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">Age Group Analysis</div>', unsafe_allow_html=True)
        if 'age' in raw_df.columns and 'adherent' in raw_df.columns:
            raw_copy = raw_df.copy()
            raw_copy['age_group'] = pd.cut(
                raw_copy['age'],
                bins=[0, 35, 64, 120],
                labels=['Young (18-35)', 'Adult (36-64)', 'Elderly (65+)'],
            )

            col1, col2 = st.columns(2)
            with col1:
                age_adh = raw_copy.groupby('age_group', observed=True)['adherent'].mean().reset_index()
                age_adh.columns = ['Age Group', 'Adherence Rate']
                fig = px.bar(
                    age_adh, x='Age Group', y='Adherence Rate',
                    color='Age Group',
                    color_discrete_sequence=['#3498DB', '#F39C12', '#9B59B6'],
                    title="Adherence Rate by Age Group",
                    text=age_adh['Adherence Rate'].apply(lambda x: f"{x:.1%}"),
                )
                fig.update_traces(textposition='outside')
                fig.update_yaxes(range=[0, 1.1])
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = px.histogram(
                    raw_copy, x='age',
                    color='adherent',
                    color_discrete_map={0: '#E74C3C', 1: '#27AE60'},
                    title="Age Distribution by Adherence",
                    nbins=25,
                    barmode='overlay',
                    opacity=0.7,
                    labels={'adherent': 'Adherent'},
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Condition analysis
            if 'chronic_condition' in raw_df.columns:
                st.markdown('<div class="section-title">Adherence by Chronic Condition</div>', unsafe_allow_html=True)
                cond_data = raw_df.groupby('chronic_condition')['adherent'].agg(['mean', 'count']).reset_index()
                cond_data.columns = ['Condition', 'Adherence Rate', 'Count']
                cond_data['Adherence Rate (%)'] = (cond_data['Adherence Rate'] * 100).round(1)
                fig3 = px.bar(
                    cond_data.sort_values('Adherence Rate (%)', ascending=False),
                    x='Condition', y='Adherence Rate (%)',
                    color='Adherence Rate (%)',
                    color_continuous_scale='RdYlGn',
                    title="Adherence Rate by Chronic Condition",
                    text='Adherence Rate (%)',
                )
                fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-title">Financial Burden Analysis</div>', unsafe_allow_html=True)
        if 'claim_amount' in raw_df.columns and 'annual_contribution' in raw_df.columns and 'adherent' in raw_df.columns:
            raw_copy2 = raw_df.dropna(subset=['claim_amount', 'annual_contribution']).copy()
            raw_copy2['financial_burden'] = (raw_copy2['claim_amount'] / raw_copy2['annual_contribution']).clip(0, 5)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(
                    raw_copy2, x='adherent', y='financial_burden',
                    color='adherent',
                    color_discrete_map={0: '#E74C3C', 1: '#27AE60'},
                    title="Financial Burden by Adherence",
                    labels={'adherent': 'Adherent (0=No, 1=Yes)', 'financial_burden': 'Financial Burden'},
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = px.scatter(
                    raw_copy2.sample(min(500, len(raw_copy2))),
                    x='annual_contribution', y='claim_amount',
                    color='adherent',
                    color_discrete_map={0: '#E74C3C', 1: '#27AE60'},
                    title="Annual Contribution vs Claim Amount",
                    labels={'adherent': 'Adherent'},
                    opacity=0.6,
                )
                st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-title">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        if not featured_df.empty:
            key_cols = ['adherent', 'refill_ratio', 'financial_burden',
                        'age_group', 'refill_gap', 'medication_complexity', 'supply_category']
            key_cols = [c for c in key_cols if c in featured_df.columns]
            corr = featured_df[key_cols].corr()

            fig = px.imshow(
                corr,
                text_auto='.2f',
                color_continuous_scale='RdYlGn',
                title="Feature Correlation Heatmap",
                aspect='auto',
                zmin=-1, zmax=1,
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown('<div class="section-title">Raw Dataset</div>', unsafe_allow_html=True)
        st.dataframe(raw_df, hide_index=True, use_container_width=True)
        st.caption(f"Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")


# ═══════════════════════════════════════════════════════════════
# PAGE 3: MODEL RESULTS
# ═══════════════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.markdown('<div class="main-header">🤖 Machine Learning Results</div>', unsafe_allow_html=True)

    if metrics_df.empty:
        st.error("No model metrics found. Please run the pipeline first.")
        st.stop()

    # ── Top model highlight ──────────────────────────────────
    best_idx   = metrics_df['ROC-AUC'].idxmax()
    best_row   = metrics_df.loc[best_idx]

    st.success(f"🏆 **Best Model: {best_row['Model']}**  |  "
               f"ROC-AUC: **{best_row['ROC-AUC']:.4f}**  |  "
               f"Accuracy: **{best_row['Accuracy']:.1%}**  |  "
               f"F1: **{best_row['F1 Score']:.4f}**")

    # ── Metrics table ───────────────────────────────────────
    st.markdown('<div class="section-title">📋 All Model Metrics</div>', unsafe_allow_html=True)
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    # ── Grouped bar chart ───────────────────────────────────
    st.markdown('<div class="section-title">📊 Model Comparison Chart</div>', unsafe_allow_html=True)
    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    metric_cols = [c for c in metric_cols if c in metrics_df.columns]

    melted = metrics_df.melt(
        id_vars=['Model'], value_vars=metric_cols,
        var_name='Metric', value_name='Score'
    )
    fig = px.bar(
        melted, x='Metric', y='Score', color='Model',
        barmode='group',
        title="Model Performance — All Metrics",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text=melted['Score'].apply(lambda x: f"{x:.3f}"),
    )
    fig.update_traces(textposition='outside', textfont_size=10)
    fig.update_yaxes(range=[0, 1.15])
    fig.add_hline(y=0.8, line_dash='dash', line_color='orange',
                  annotation_text="0.80 threshold", annotation_position="right")
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature Importance ──────────────────────────────────
    st.markdown('<div class="section-title">🎯 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)

    rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    fn_path = os.path.join(MODELS_DIR, "feature_names.pkl")

    if os.path.exists(rf_path) and os.path.exists(fn_path):
        try:
            rf_model      = joblib.load(rf_path)
        except Exception as e:
            st.warning(f"⚠️ Could not load Random Forest model (stale .pkl): {e}\n\n"
                       "Delete `outputs/models/` and restart Docker to retrain.")
            rf_model = None
        feature_names = joblib.load(fn_path)
        if rf_model is not None:
            importances = rf_model.feature_importances_
            indices     = np.argsort(importances)[::-1]
            top_n       = min(15, len(feature_names))
            top_idx     = indices[:top_n]

            fi_df = pd.DataFrame({
                'Feature'   : [feature_names[i] for i in top_idx],
                'Importance': importances[top_idx],
            }).sort_values('Importance')

            fig2 = px.bar(
                fi_df, x='Importance', y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='RdYlGn',
                title="Top Features — Random Forest Importance",
                text=fi_df['Importance'].apply(lambda x: f"{x:.3f}"),
            )
            fig2.update_traces(textposition='outside')
            fig2.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run the pipeline to generate trained models.")

    # ── Confusion Matrices ──────────────────────────────────
    st.markdown('<div class="section-title">🟦 Confusion Matrices</div>', unsafe_allow_html=True)
    if not preds_df.empty and os.path.exists(os.path.join(FIGURES_DIR, "06_confusion_matrices.png")):
        img = load_image(os.path.join(FIGURES_DIR, "06_confusion_matrices.png"))
        if img:
            st.image(img, caption="Confusion Matrices — All Models", use_column_width=True)
    else:
        st.info("Confusion matrix image not found. Run the pipeline.")


# ═══════════════════════════════════════════════════════════════
# PAGE 4: ROC & METRICS
# ═══════════════════════════════════════════════════════════════
elif page == "📈 ROC & Metrics":
    st.markdown('<div class="main-header">📈 ROC Curves & Detailed Metrics</div>', unsafe_allow_html=True)

    if preds_df.empty:
        st.error("No predictions found. Please run the pipeline first.")
        st.stop()

    # ── Interactive ROC from predictions ────────────────────
    st.markdown('<div class="section-title">ROC Curves (Interactive)</div>', unsafe_allow_html=True)

    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree"      : "decision_tree.pkl",
        "Random Forest"      : "random_forest.pkl",
        "Gradient Boosting"  : "gradient_boosting.pkl",
    }

    test_path = os.path.join(DATA_DIR, "test_data.csv")
    if os.path.exists(test_path):
        from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
        test_df = pd.read_csv(test_path)
        X_test  = test_df.drop(columns=['adherent'])
        y_test  = test_df['adherent']

        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        stale_models = []

        for i, (name, fname) in enumerate(model_files.items()):
            mpath = os.path.join(MODELS_DIR, fname)
            if not os.path.exists(mpath):
                continue
            try:
                model = joblib.load(mpath)
                if not hasattr(model, 'predict_proba'):
                    continue
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc = roc_auc_score(y_test, y_prob)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{name} (AUC={auc:.3f})",
                    line=dict(color=colors[i % len(colors)], width=2.5),
                    mode='lines',
                    hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
                ))
            except (AttributeError, Exception):
                stale_models.append(name)

        # Warn user if old .pkl files are incompatible with current sklearn
        if stale_models:
            st.warning(
                f"⚠️ **Stale model files detected** — {', '.join(stale_models)} "
                f"could not be loaded because they were trained with an older version "
                f"of scikit-learn.\n\n"
                f"**Fix:** Delete your `outputs/models/` folder and restart Docker "
                f"so the pipeline retrains fresh models:\n"
                f"```\nrm -rf outputs/models outputs/reports outputs/figures\n"
                f"docker-compose restart\n```"
            )

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name="Random Classifier (AUC=0.500)",
            line=dict(color='gray', width=1.5, dash='dash'),
            mode='lines',
        ))
        fig.update_layout(
            title="ROC Curves — All Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            height=520,
            legend=dict(x=0.6, y=0.1),
            plot_bgcolor='white',
        )
        fig.update_xaxes(range=[0, 1], gridcolor='#f0f0f0')
        fig.update_yaxes(range=[0, 1.05], gridcolor='#f0f0f0')

        # Only render the chart if at least one model loaded successfully
        if len(fig.data) > 1:
            st.plotly_chart(fig, use_container_width=True)
        elif not stale_models:
            st.info("No models loaded. Run the pipeline first.")

        # ── Per-model confusion matrices (interactive) ───────
        st.markdown('<div class="section-title">Confusion Matrices (Interactive)</div>', unsafe_allow_html=True)
        loaded_models = {}
        for name, fname in model_files.items():
            mp = os.path.join(MODELS_DIR, fname)
            if not os.path.exists(mp):
                continue
            try:
                loaded_models[name] = joblib.load(mp)
            except (AttributeError, Exception):
                pass  # already warned above

        if loaded_models:
            cols = st.columns(min(2, len(loaded_models)))
            for idx, (name, model) in enumerate(loaded_models.items()):
                try:
                    y_pred = model.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    heat = go.Figure(go.Heatmap(
                        z=cm,
                        x=['Non-Adherent', 'Adherent'],
                        y=['Non-Adherent', 'Adherent'],
                        colorscale='Blues',
                        text=[[f"TN={tn}", f"FP={fp}"], [f"FN={fn}", f"TP={tp}"]],
                        texttemplate="%{text}",
                        textfont={"size": 16, "color": "black"},
                        showscale=False,
                    ))
                    heat.update_layout(
                        title=dict(text=f"{name}", font=dict(size=13)),
                        xaxis_title="Predicted",
                        yaxis_title="Actual",
                        height=280,
                        margin=dict(t=45, b=30, l=60, r=10),
                    )
                    with cols[idx % 2]:
                        st.plotly_chart(heat, use_container_width=True)
                except Exception:
                    pass

    else:
        img = load_image(os.path.join(FIGURES_DIR, "07_roc_curves.png"))
        if img:
            st.image(img, caption="ROC Curves", use_column_width=True)

    # ── Precision-Recall comparison ─────────────────────────
    st.markdown('<div class="section-title">Precision vs Recall Trade-off</div>', unsafe_allow_html=True)
    if not metrics_df.empty:
        fig2 = px.scatter(
            metrics_df, x='Recall', y='Precision',
            color='Model', size='F1 Score',
            size_max=30,
            text='Model',
            title="Precision vs Recall (bubble size = F1 Score)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.update_traces(textposition='top center')
        fig2.update_layout(height=420, showlegend=True)
        fig2.update_xaxes(range=[0, 1.05])
        fig2.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 5: PREDICTIONS BROWSER
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Predictions Browser":
    st.markdown('<div class="main-header">🔍 Predictions Browser</div>', unsafe_allow_html=True)

    if preds_df.empty:
        st.error("No predictions found. Please run the pipeline first.")
        st.stop()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_adherence = st.multiselect(
            "Filter by Actual Adherence",
            options=[0, 1], default=[0, 1],
            format_func=lambda x: "Adherent" if x == 1 else "Non-Adherent",
        )
    with col2:
        if 'confidence' in preds_df.columns:
            show_confidence = st.multiselect(
                "Filter by Confidence",
                options=preds_df['confidence'].unique().tolist(),
                default=preds_df['confidence'].unique().tolist(),
            )
        else:
            show_confidence = []
    with col3:
        show_correct = st.selectbox(
            "Filter by Prediction Correctness",
            options=["All", "Correct Only", "Wrong Only"],
        )

    filtered = preds_df.copy()
    if show_adherence:
        filtered = filtered[filtered['actual_adherence'].isin(show_adherence)]
    if show_confidence and 'confidence' in filtered.columns:
        filtered = filtered[filtered['confidence'].isin(show_confidence)]
    if show_correct == "Correct Only" and 'correct_prediction' in filtered.columns:
        filtered = filtered[filtered['correct_prediction'] == 1]
    elif show_correct == "Wrong Only" and 'correct_prediction' in filtered.columns:
        filtered = filtered[filtered['correct_prediction'] == 0]

    st.info(f"Showing {len(filtered):,} of {len(preds_df):,} records")

    # Summary KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records", f"{len(filtered):,}")
    if 'adherence_probability' in filtered.columns:
        k2.metric("Avg Adherence Probability", f"{filtered['adherence_probability'].mean():.1%}")
    if 'correct_prediction' in filtered.columns:
        k3.metric("Correct Predictions", f"{filtered['correct_prediction'].sum():,}")
        k4.metric("Accuracy on Filter", f"{filtered['correct_prediction'].mean():.1%}")

    # Probability distribution
    if 'adherence_probability' in filtered.columns:
        st.markdown('<div class="section-title">Adherence Probability Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            filtered,
            x='adherence_probability',
            color='actual_adherence' if 'actual_adherence' in filtered.columns else None,
            color_discrete_map={0: '#E74C3C', 1: '#27AE60'},
            nbins=30,
            barmode='overlay',
            opacity=0.75,
            title="Distribution of Predicted Adherence Probability",
            labels={'adherence_probability': 'Predicted Probability', 'actual_adherence': 'Actual'},
        )
        fig.add_vline(x=0.5, line_dash='dash', line_color='gray',
                      annotation_text="Decision Threshold 0.5")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Prediction Records</div>', unsafe_allow_html=True)
    st.dataframe(filtered.reset_index(drop=True), hide_index=True, use_container_width=True)

    # Download
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Filtered Predictions",
        data=csv_bytes,
        file_name="filtered_predictions.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════
# PAGE 6: LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════════
elif page == "🔮 Live Predictor":
    st.markdown('<div class="main-header">🔮 Live Patient Adherence Predictor</div>', unsafe_allow_html=True)
    st.markdown("Enter a patient's details below to get a real-time adherence prediction.")

    rf_path     = os.path.join(MODELS_DIR, "random_forest.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    fn_path     = os.path.join(MODELS_DIR, "feature_names.pkl")

    if not (os.path.exists(rf_path) and os.path.exists(fn_path)):
        st.error("Model files not found. Please run the ML pipeline first.")
        st.stop()

    rf_model      = joblib.load(rf_path)
    scaler        = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    feature_names = joblib.load(fn_path)

    with st.form("predictor_form"):
        st.markdown("#### Patient Demographics")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 85, 45)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col3:
            insurance = st.selectbox("Insurance Type", ["HMO", "PPO", "Medicare", "Medicaid"])

        st.markdown("#### Financial Information")
        col4, col5 = st.columns(2)
        with col4:
            annual_contribution = st.number_input("Annual Contribution ($)", 500, 8000, 3000, step=100)
        with col5:
            claim_amount = st.number_input("Claim Amount ($)", 50, 6000, 1200, step=50)

        st.markdown("#### Prescription Details")
        col6, col7, col8 = st.columns(3)
        with col6:
            expected_refills = st.slider("Expected Refills", 3, 12, 8)
        with col7:
            refills_received = st.slider("Refills Received", 0, 12, 6)
        with col8:
            days_supply = st.selectbox("Days Supply", [30, 60, 90], index=0)

        st.markdown("#### Clinical Details")
        col9, col10 = st.columns(2)
        with col9:
            condition = st.selectbox("Chronic Condition", ["None", "Diabetes", "Hypertension", "Asthma", "Heart Disease"])
        with col10:
            num_meds = st.slider("Number of Medications", 1, 7, 3)

        submitted = st.form_submit_button("🔮 Predict Adherence", use_container_width=True)

    if submitted:
        # ── Engineer features ──────────────────────────────
        refill_ratio         = min(refills_received / max(expected_refills, 1), 1.0)
        financial_burden     = min(claim_amount / max(annual_contribution, 1), 5.0)
        age_group            = 0 if age <= 35 else (1 if age <= 64 else 2)
        medication_complexity= 0 if num_meds <= 2 else (1 if num_meds <= 5 else 2)
        supply_category      = 0 if days_supply <= 30 else (1 if days_supply <= 60 else 2)
        refill_gap           = max(expected_refills - refills_received, 0)

        # ── Build input DataFrame ──────────────────────────
        input_dict = {
            'age'                        : age,
            'annual_contribution'        : annual_contribution,
            'claim_amount'               : claim_amount,
            'days_supply'                : days_supply,
            'num_medications'            : num_meds,
            'gender_encoded'             : 1 if gender == 'Male' else 0,
            'insurance_type_Medicaid'    : 1 if insurance == 'Medicaid' else 0,
            'insurance_type_Medicare'    : 1 if insurance == 'Medicare' else 0,
            'insurance_type_PPO'         : 1 if insurance == 'PPO' else 0,
            'chronic_condition_Diabetes' : 1 if condition == 'Diabetes' else 0,
            'chronic_condition_Heart Disease': 1 if condition == 'Heart Disease' else 0,
            'chronic_condition_Hypertension': 1 if condition == 'Hypertension' else 0,
            'chronic_condition_None'     : 1 if condition == 'None' else 0,
            'refill_ratio'               : refill_ratio,
            'financial_burden'           : financial_burden,
            'age_group'                  : age_group,
            'medication_complexity'      : medication_complexity,
            'supply_category'            : supply_category,
            'refill_gap'                 : refill_gap,
        }

        # Build DataFrame aligned to training features
        input_df = pd.DataFrame([{f: input_dict.get(f, 0) for f in feature_names}])

        # Apply scaler if available
        if scaler is not None:
            try:
                scale_cols = [c for c in feature_names if input_df[c].nunique() > 2 or input_df[c].iloc[0] > 2]
                # safer: just try transform on all features
                input_scaled = input_df.copy()
                input_scaled[feature_names] = scaler.transform(input_df[feature_names])
                input_df = input_scaled
            except Exception:
                pass  # fallback: use unscaled

        prob = rf_model.predict_proba(input_df)[0][1]
        pred = int(prob >= 0.5)

        # ── Result display ─────────────────────────────────
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if pred == 1:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#11998e,#38ef7d);
                             padding:30px;border-radius:16px;text-align:center;color:white;">
                    <div style="font-size:3.5rem;">✅</div>
                    <div style="font-size:1.8rem;font-weight:700;">ADHERENT</div>
                    <div style="font-size:1.2rem;">Probability: {prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#cb2d3e,#ef473a);
                             padding:30px;border-radius:16px;text-align:center;color:white;">
                    <div style="font-size:3.5rem;">⚠️</div>
                    <div style="font-size:1.8rem;font-weight:700;">NON-ADHERENT</div>
                    <div style="font-size:1.2rem;">Probability: {prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            # Gauge chart
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={'suffix': '%', 'font': {'size': 32}},
                delta={'reference': 50, 'increasing': {'color': '#27AE60'}, 'decreasing': {'color': '#E74C3C'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': '#27AE60' if pred == 1 else '#E74C3C'},
                    'steps': [
                        {'range': [0, 40], 'color': '#fdecea'},
                        {'range': [40, 60], 'color': '#fff3cd'},
                        {'range': [60, 100], 'color': '#e8f5e9'},
                    ],
                    'threshold': {'line': {'color': 'gray', 'width': 3}, 'thickness': 0.75, 'value': 50},
                },
                title={'text': "Adherence Probability"},
            ))
            gauge.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(gauge, use_container_width=True)

        # ── Key factors ────────────────────────────────────
        st.markdown("#### Key Factors for This Patient")
        factors_data = {
            'Factor': ['Refill Ratio', 'Financial Burden', 'Age Group', 'Medication Complexity', 'Supply Category', 'Refill Gap'],
            'Value': [f"{refill_ratio:.2f}", f"{financial_burden:.2f}", ['Young', 'Adult', 'Elderly'][age_group],
                      ['Low', 'Medium', 'High'][medication_complexity], ['Short(30d)', 'Medium(60d)', 'Long(90d)'][supply_category],
                      str(refill_gap)],
            'Impact': ['High ↑' if refill_ratio > 0.75 else 'Risk ↓',
                       'Risk ↑' if financial_burden > 0.6 else 'Low ↓',
                       'Moderate' if age_group == 1 else 'At Risk',
                       'Low' if medication_complexity == 0 else ('Medium' if medication_complexity == 1 else 'High Risk ↑'),
                       'Positive ↑' if supply_category == 2 else 'Neutral',
                       'Good ✓' if refill_gap == 0 else f'Gap of {refill_gap} ↓'],
        }
        st.dataframe(pd.DataFrame(factors_data), hide_index=True, use_container_width=True)
