#!/usr/bin/env python3
"""Build a standalone HTML dashboard from project outputs."""

from __future__ import annotations

import csv
import html
import os
from pathlib import Path


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def as_int(value: str, default: int = 0) -> int:
    try:
        return int(float((value or "").strip()))
    except (TypeError, ValueError):
        return default


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float((value or "").strip())
    except (TypeError, ValueError):
        return default


def render_table(rows: list[dict[str, str]], title: str) -> str:
    if not rows:
        return f"<h3>{html.escape(title)}</h3><p>No data available.</p>"
    columns = list(rows[0].keys())
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows)
    return (
        f"<h3>{html.escape(title)}</h3>"
        f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reports_dir = root / "outputs" / "reports"
    figures_dir = root / "outputs" / "figures"
    output_html = reports_dir / "project_dashboard.html"

    required_files = {
        "final_predictions": reports_dir / "final_predictions.csv",
        "model_metrics": reports_dir / "model_metrics.csv",
        "tableau_output": reports_dir / "tableau_output.csv",
    }
    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    final_predictions = load_csv(required_files["final_predictions"])
    model_metrics = load_csv(required_files["model_metrics"])
    tableau_output = load_csv(required_files["tableau_output"])

    total_patients = len(final_predictions)
    adherence_rate = (
        100.0
        * sum(as_int(row.get("actual_adherence", "0")) for row in final_predictions)
        / max(1, total_patients)
    )
    accuracy = (
        100.0
        * sum(as_int(row.get("correct_prediction", "0")) for row in final_predictions)
        / max(1, total_patients)
    )
    best_auc = max((as_float(row.get("ROC-AUC", "0")) for row in model_metrics), default=0.0)

    sorted_metrics = sorted(
        model_metrics,
        key=lambda row: as_float(row.get("ROC-AUC", "0")),
        reverse=True,
    )

    images = [
        ("Confusion Matrices", figures_dir / "06_confusion_matrices.png"),
        ("ROC Curves", figures_dir / "07_roc_curves.png"),
        ("Model Comparison", figures_dir / "08_model_comparison.png"),
        ("Feature Importance", figures_dir / "09_feature_importance.png"),
    ]
    image_html_blocks: list[str] = []
    for title, path in images:
        if path.exists():
            rel_path = os.path.relpath(path, reports_dir).replace("\\", "/")
            image_html_blocks.append(
                f"<section><h3>{html.escape(title)}</h3><img src='{html.escape(rel_path)}' alt='{html.escape(title)}'></section>"
            )

    dashboard_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Medical Adherence Project Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f5f7fa; color: #1f2a37; margin: 0; padding: 24px; }}
    h1, h2, h3 {{ margin-top: 0; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }}
    .kpi {{ background: white; border-radius: 8px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
    .kpi .label {{ font-size: 12px; color: #6b7280; }}
    .kpi .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    .panel {{ background: white; border-radius: 8px; padding: 14px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    img {{ width: 100%; border-radius: 8px; border: 1px solid #e5e7eb; }}
    .small-note {{ font-size: 12px; color: #6b7280; }}
  </style>
</head>
<body>
  <h1>Medical Adherence Predictor - Full Project Dashboard</h1>
  <p class="small-note">Generated from CSV outputs. Refresh by rerunning this script after pipeline updates.</p>

  <div class="kpis">
    <div class="kpi"><div class="label">Total Test Patients</div><div class="value">{total_patients}</div></div>
    <div class="kpi"><div class="label">Actual Adherence Rate</div><div class="value">{adherence_rate:.2f}%</div></div>
    <div class="kpi"><div class="label">Prediction Accuracy</div><div class="value">{accuracy:.2f}%</div></div>
    <div class="kpi"><div class="label">Best ROC-AUC</div><div class="value">{best_auc:.4f}</div></div>
  </div>

  <section class="panel">
    {render_table(sorted_metrics, "Model Metrics (sorted by ROC-AUC)")}
  </section>

  <section class="panel">
    {render_table(tableau_output[:20], "Tableau Output (first 20 rows)")}
  </section>

  {"".join(f"<section class='panel'>{block}</section>" for block in image_html_blocks)}
</body>
</html>
"""

    reports_dir.mkdir(parents=True, exist_ok=True)
    output_html.write_text(dashboard_html, encoding="utf-8")
    print(f"Project dashboard generated at: {output_html}")


if __name__ == "__main__":
    main()
