#!/usr/bin/env python3
"""Builds SQLite data source for Grafana from project report CSVs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    reports_dir = project_root / "outputs" / "reports"
    sqlite_path = reports_dir / "adherence_dashboard.db"

    final_predictions_path = reports_dir / "final_predictions.csv"
    model_metrics_path = reports_dir / "model_metrics.csv"
    tableau_output_path = reports_dir / "tableau_output.csv"

    if not final_predictions_path.exists():
        raise FileNotFoundError(f"Missing file: {final_predictions_path}")
    if not model_metrics_path.exists():
        raise FileNotFoundError(f"Missing file: {model_metrics_path}")
    if not tableau_output_path.exists():
        raise FileNotFoundError(f"Missing file: {tableau_output_path}")

    final_predictions = pd.read_csv(final_predictions_path)
    model_metrics = pd.read_csv(model_metrics_path)
    tableau_output = pd.read_csv(tableau_output_path)

    model_metrics.columns = [
        col.strip().lower().replace("-", "_").replace(" ", "_")
        for col in model_metrics.columns
    ]

    if sqlite_path.exists():
        sqlite_path.unlink()

    with sqlite3.connect(sqlite_path) as connection:
        final_predictions.to_sql("final_predictions", connection, if_exists="replace", index=False)
        model_metrics.to_sql("model_metrics", connection, if_exists="replace", index=False)
        tableau_output.to_sql("tableau_output", connection, if_exists="replace", index=False)

    print(f"Grafana SQLite database created at: {sqlite_path}")


if __name__ == "__main__":
    main()
