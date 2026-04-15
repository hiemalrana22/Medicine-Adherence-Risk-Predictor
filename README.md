# 💊 Medical Adherence Predictor

> An end-to-end machine learning project to predict patient medication adherence using Python, R, Docker, and Power BI.

---

## 📌 Project Overview

Medication non-adherence is one of the leading causes of poor health outcomes globally. This project builds a machine learning pipeline to **predict whether a patient will adhere to their medication** based on demographic, financial, and prescription refill data.

This is an internship-level data science project demonstrating:
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Machine Learning modeling and evaluation
- Statistical analysis in R
- Containerization with Docker
- Dashboard-ready data export for Power BI

---

## 📂 Dataset

**Source:** [Mendeley Data – Medication Adherence Dataset](https://data.mendeley.com/datasets/zkp7sbbx64/2)

The dataset contains patient-level information including demographics, financial data (insurance claims, contributions), and prescription refill history. The target variable is **Adherence** (binary: 1 = Adherent, 0 = Non-Adherent).

---

## 🛠️ Tech Stack

| Tool       | Purpose                              |
|------------|--------------------------------------|
| Python 3.10 | ML pipeline, preprocessing, modeling |
| R 4.x      | Statistical analysis, ggplot2 viz    |
| Docker     | Containerization                     |
| Power BI   | Final dashboard (CSV export)         |
| scikit-learn | ML models                          |
| pandas / numpy | Data manipulation               |
| matplotlib / seaborn | EDA plots               |
| imbalanced-learn | SMOTE for class imbalance    |

---

## 📁 Project Structure

```
medical_adherence_predictor/
│
├── data/
│   ├── raw/                    # Original dataset (place downloaded CSV here)
│   └── processed/              # Cleaned & feature-engineered data
│
├── notebooks/
│   └── exploration.ipynb       # Jupyter notebook for step-by-step walkthrough
│
├── src/
│   ├── preprocessing.py        # Data cleaning, encoding, scaling
│   ├── feature_engineering.py  # New feature creation
│   ├── train.py                # Model training pipeline
│   └── evaluate.py             # Metrics, confusion matrix, ROC curve
│
├── r_scripts/
│   └── analysis.R              # Statistical analysis & ggplot2 visualizations
│
├── outputs/
│   ├── figures/                # Saved plots
│   ├── models/                 # Saved trained model (.pkl)
│   └── reports/                # Final CSV for Power BI
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container setup
└── README.md                   # This file
```

---

## 🚀 How to Run

### Option 1: Local Python Environment

```bash
# 1. Clone or download the project
cd medical_adherence_predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset in data/raw/ folder
# Download from: https://data.mendeley.com/datasets/zkp7sbbx64/2

# 4. Run the full pipeline
python src/preprocessing.py
python src/feature_engineering.py
python src/train.py
python src/evaluate.py
```

### Option 2: Docker (Recommended)

```bash
# 1. Build the Docker image
docker build -t medical-adherence-predictor .

# 2. Run the container
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs medical-adherence-predictor

# 3. Outputs (plots, models, CSVs) will appear in the outputs/ folder
```

### Option 3: R Script

```bash
# In R or RStudio, open and run:
Rscript r_scripts/analysis.R
```

---

## 📊 Power BI Dashboard

After running the pipeline, open Power BI Desktop and load:
```
outputs/reports/final_predictions.csv
```

**Suggested Dashboard Pages:**
1. **Overview** – Total patients, adherence rate, key KPIs
2. **Demographics** – Age group, gender distribution vs adherence
3. **Financial Analysis** – Claim amounts, financial burden vs adherence
4. **Model Insights** – Feature importance, prediction confidence

---

## 📈 Grafana Dashboard (Localhost)

This repo now includes a fully provisioned Grafana setup using SQLite as the data source for:
- `outputs/reports/final_predictions.csv`
- `outputs/reports/model_metrics.csv`
- `outputs/reports/tableau_output.csv`

### 1) Build Grafana data source DB

```bash
python scripts/build_grafana_data.py
```

This creates:
`outputs/reports/adherence_dashboard.db`

### 2) Start Grafana on localhost

```bash
docker compose -f docker-compose.grafana.yml up -d
```

Open:
`http://localhost:3001`

Login:
- Username: `admin`
- Password: `admin`

If you specifically want port 3000:
```bash
GRAFANA_PORT=3000 docker compose -f docker-compose.grafana.yml up -d
```

### 3) Stop Grafana

```bash
docker compose -f docker-compose.grafana.yml down
```

### Notes
- Dashboard file: `grafana_dashboard.json`
- Datasource provisioning: `grafana/provisioning/datasources/sqlite.yml`
- Dashboard provisioning: `grafana/provisioning/dashboards/dashboard.yml`
- If you rerun the ML pipeline and outputs change, rerun:
  `python scripts/build_grafana_data.py`

---

## 📈 Results

| Model               | Accuracy | F1 Score | ROC-AUC |
|--------------------|----------|----------|---------|
| Logistic Regression | ~78%    | ~0.76    | ~0.83   |
| Decision Tree       | ~80%    | ~0.79    | ~0.81   |
| Random Forest       | **~85%**| **~0.84**| **~0.89**|

> ⚠️ Recall is the most important metric in healthcare — we want to minimize missed non-adherent patients.

---

## 🧠 Key Insights

- **Refill ratio** is the strongest predictor of adherence
- **Financial burden** negatively correlates with adherence
- **Elderly patients** show lower adherence rates
- Class imbalance handled using **SMOTE** oversampling

---

## 👤 Author

Built as an internship-level ML project demonstrating end-to-end data science skills.

---

## 📄 License

For educational and research purposes only. Dataset credits: Mendeley Data.
