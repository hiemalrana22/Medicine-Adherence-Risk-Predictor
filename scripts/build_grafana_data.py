#!/usr/bin/env python3
"""Builds SQLite data source for Grafana from project report CSVs."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Callable


def normalize_column(name: str) -> str:
    """Normalize CSV column names for SQLite compatibility."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def infer_sql_type(values: list[str]) -> str:
    """Infer SQLite column type from CSV string values."""
    has_float = False
    for raw_value in values:
        value = (raw_value or "").strip()
        if value == "":
            continue
        try:
            int(value)
            continue
        except ValueError:
            try:
                float(value)
                has_float = True
                continue
            except ValueError:
                return "TEXT"
    return "REAL" if has_float else "INTEGER"


def cast_value(raw_value: str, sql_type: str):
    """Cast CSV cell value to SQLite-compatible Python type."""
    value = (raw_value or "").strip()
    if value == "":
        return None
    if sql_type == "INTEGER":
        return int(value)
    if sql_type == "REAL":
        return float(value)
    return value


def load_csv_rows(csv_path: Path, column_transform: Callable[[str], str] | None = None) -> tuple[list[str], list[dict[str, str]]]:
    """Load CSV into memory as header + row dictionaries."""
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no headers: {csv_path}")

        original_headers = [header.strip() for header in reader.fieldnames]
        headers = (
            [column_transform(header) for header in original_headers]
            if column_transform
            else original_headers
        )

        rows: list[dict[str, str]] = []
        for raw_row in reader:
            if column_transform:
                row = {
                    column_transform(key): value
                    for key, value in raw_row.items()
                    if key is not None
                }
            else:
                row = {key.strip(): value for key, value in raw_row.items() if key is not None}
            rows.append(row)

    return headers, rows


def write_table(connection: sqlite3.Connection, table_name: str, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Create and populate a SQLite table from row dictionaries."""
    if not columns:
        raise ValueError(f"No columns provided for table '{table_name}'.")

    column_values = {column: [] for column in columns}
    for row in rows:
        for column in columns:
            column_values[column].append(row.get(column, ""))

    column_types = {column: infer_sql_type(values) for column, values in column_values.items()}
    quoted_columns = [f'"{column}"' for column in columns]
    create_columns = ", ".join(
        f'"{column}" {column_types[column]}'
        for column in columns
    )
    connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    connection.execute(f'CREATE TABLE "{table_name}" ({create_columns})')

    placeholders = ", ".join(["?"] * len(columns))
    insert_sql = (
        f'INSERT INTO "{table_name}" ({", ".join(quoted_columns)}) '
        f"VALUES ({placeholders})"
    )
    insert_values = [
        tuple(cast_value(row.get(column, ""), column_types[column]) for column in columns)
        for row in rows
    ]
    if insert_values:
        connection.executemany(insert_sql, insert_values)
    connection.commit()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    reports_dir = project_root / "outputs" / "reports"
    sqlite_path = reports_dir / "adherence_dashboard.db"

    csv_files = {
        "final_predictions": reports_dir / "final_predictions.csv",
        "model_metrics": reports_dir / "model_metrics.csv",
        "tableau_output": reports_dir / "tableau_output.csv",
    }
    for table_name, path in csv_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file for {table_name}: {path}")

    if sqlite_path.exists():
        sqlite_path.unlink()

    with sqlite3.connect(sqlite_path) as connection:
        for table_name, path in csv_files.items():
            transform = normalize_column if table_name == "model_metrics" else None
            columns, rows = load_csv_rows(path, transform)
            write_table(connection, table_name, columns, rows)
            print(f"Loaded table '{table_name}' ({len(rows)} rows)")

    print(f"Grafana SQLite database created at: {sqlite_path}")


if __name__ == "__main__":
    main()
