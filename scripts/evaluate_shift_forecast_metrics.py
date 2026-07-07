#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate demand forecast errors for shifted forecast CSV files.

The shift analysis outputs contain both the realized demand and the shifted
forecast in one CSV, for example:

    timestamp,demand_before_kw,charge_kw,discharge_kw,forecast_kw,...

This script computes conventional forecast metrics and peak-timing metrics for
one or more such files.
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd


ACTUAL_CANDIDATES = [
    "demand_before_kw",
    "actual_kw",
    "demand_mw",
    "demand_kw",
    "actual_mw",
]
FORECAST_CANDIDATES = [
    "forecast_kw",
    "demand_forecast_kw",
    "demand_forecast_mw",
    "demand_predict",
    "demand_predict_mw",
    "forecast",
]


def _resolve_col(df: pd.DataFrame, requested: str | None, candidates: list[str], kind: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"{kind} column '{requested}' not found")
        return requested
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"{kind} column not found. Tried: {', '.join(candidates)}")


def _read_timestamp(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, errors="coerce")
    if ts.isna().all():
        raise ValueError("timestamp column could not be parsed")
    return ts


def _infer_slot_minutes(ts: pd.Series) -> float:
    diffs = ts.sort_values().diff().dropna()
    if diffs.empty:
        return float("nan")
    return float(diffs.median().total_seconds() / 60.0)


def _infer_shift_slots(path: Path) -> int | None:
    m = re.search(r"shift_([+-]?\d+)slot", path.stem)
    if not m:
        return None
    return int(m.group(1))


def _format_time(ts: pd.Timestamp) -> str:
    return ts.strftime("%H:%M")


def evaluate_file(
    path: Path,
    timestamp_col: str,
    actual_col: str | None,
    forecast_col: str | None,
    date: str | None,
    nrmse_denominator: str,
) -> dict:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"timestamp column '{timestamp_col}' not found")

    actual_name = _resolve_col(df, actual_col, ACTUAL_CANDIDATES, "actual")
    forecast_name = _resolve_col(df, forecast_col, FORECAST_CANDIDATES, "forecast")

    df = df.copy()
    df[timestamp_col] = _read_timestamp(df[timestamp_col])
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

    if date:
        day = pd.Timestamp(date).date()
        df = df[df[timestamp_col].dt.date == day]

    df = df[[timestamp_col, actual_name, forecast_name]].dropna()
    if df.empty:
        raise ValueError("no valid rows after filtering")

    actual = df[actual_name].astype(float).to_numpy()
    forecast = df[forecast_name].astype(float).to_numpy()
    error = forecast - actual

    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    nonzero = actual != 0
    mape = float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0) if nonzero.any() else float("nan")

    actual_peak_idx = int(np.argmax(actual))
    forecast_peak_idx = int(np.argmax(forecast))
    actual_peak_time = df.iloc[actual_peak_idx][timestamp_col]
    forecast_peak_time = df.iloc[forecast_peak_idx][timestamp_col]
    actual_peak_value = float(actual[actual_peak_idx])
    forecast_peak_value = float(forecast[forecast_peak_idx])

    slot_minutes = _infer_slot_minutes(df[timestamp_col])
    peak_time_error_minutes = float((forecast_peak_time - actual_peak_time).total_seconds() / 60.0)
    peak_time_error_slots = (
        float(peak_time_error_minutes / slot_minutes)
        if slot_minutes and not np.isnan(slot_minutes)
        else float("nan")
    )
    peak_magnitude_error = forecast_peak_value - actual_peak_value

    if nrmse_denominator == "range":
        denom = float(actual.max() - actual.min())
    elif nrmse_denominator == "mean":
        denom = float(actual.mean())
    else:
        denom = actual_peak_value
    nrmse = float(rmse / denom) if denom != 0 else float("nan")

    return {
        "file": str(path),
        "shift_slots": _infer_shift_slots(path),
        "rows": int(len(df)),
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": mae,
        "mape_pct": mape,
        "actual_peak_time": _format_time(actual_peak_time),
        "forecast_peak_time": _format_time(forecast_peak_time),
        "peak_time_error_h": peak_time_error_minutes / 60.0,
        "peak_time_error_slots": peak_time_error_slots,
        "actual_peak": actual_peak_value,
        "forecast_peak": forecast_peak_value,
        "peak_magnitude_error": peak_magnitude_error,
        "actual_col": actual_name,
        "forecast_col": forecast_name,
    }


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    unique = sorted(set(paths), key=lambda p: str(p))
    return unique


def to_markdown_table(df: pd.DataFrame) -> str:
    """Render a small markdown table without requiring pandas[tabulate]."""
    headers = [str(c) for c in df.columns]
    rows = df.astype(object).where(pd.notna(df), "").astype(str).values.tolist()

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RMSE/NRMSE/MAPE and peak timing errors for shifted forecast CSVs."
    )
    parser.add_argument("--csv", nargs="+", required=True, help="Input CSV paths or glob patterns")
    parser.add_argument("--date", help="Optional date filter, e.g. 2024-08-05")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--actual-col", help="Actual demand column. Auto-detected if omitted.")
    parser.add_argument("--forecast-col", help="Forecast demand column. Auto-detected if omitted.")
    parser.add_argument(
        "--nrmse-denominator",
        choices=["range", "mean", "peak"],
        default="range",
        help="Denominator used for NRMSE. Default: range.",
    )
    parser.add_argument("--out-csv", help="Optional output CSV path for the metrics table")
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown table")
    args = parser.parse_args()

    rows = []
    for path in expand_inputs(args.csv):
        if not path.exists():
            raise SystemExit(f"input CSV not found: {path}")
        rows.append(
            evaluate_file(
                path=path,
                timestamp_col=args.timestamp_col,
                actual_col=args.actual_col,
                forecast_col=args.forecast_col,
                date=args.date,
                nrmse_denominator=args.nrmse_denominator,
            )
        )

    metrics = pd.DataFrame(rows)
    if "shift_slots" in metrics:
        metrics = metrics.sort_values(["shift_slots", "file"], na_position="last")

    display_cols = [
        "shift_slots",
        "rmse",
        "nrmse",
        "mape_pct",
        "actual_peak_time",
        "forecast_peak_time",
        "peak_time_error_h",
        "peak_time_error_slots",
        "peak_magnitude_error",
        "actual_peak",
        "forecast_peak",
        "file",
    ]
    display = metrics[display_cols].copy()
    numeric_cols = display.select_dtypes(include=[np.number]).columns
    display[numeric_cols] = display[numeric_cols].round(4)

    if args.markdown:
        print(to_markdown_table(display))
    else:
        print(display.to_string(index=False))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out_path, index=False)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
