"""Rebuild the revised Experiment 2 from TEPCO inputs."""

import argparse
import csv
from datetime import timedelta
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from prepare_tepco import ROOT, prepare

from experiments.eneres_bess import load_design, run_baseline, write_csv
from experiments.eneres_design import battery_configs, load_forecasts, select_days, selected_error_metrics
from experiments.eneres_oracle import run_diagnostics
from experiments.eneres_sensitivity import run_sensitivity
from experiments.eneres_shift import run_shifts
from forecast.eneres_day_ahead import load_prepared, sha256

INPUTS = ROOT / "data/inputs.json"
FROZEN_FORECAST = ROOT / "data/lstm_evaluation.csv.gz"
REFERENCE_PEAK_KW = 57_006_000.0


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_frozen_forecasts(dataset, periods, inputs):
    if sha256(FROZEN_FORECAST) != inputs["forecast_sha256"]:
        raise ValueError("Frozen LSTM forecast checksum mismatch")
    truth = {record.timestamp: record.demand_kw for record in dataset.records}
    evaluation = [record for record in dataset.records
                  if periods.evaluation_start <= record.timestamp.date() < periods.end_exclusive]
    by_day = {}
    with gzip.open(FROZEN_FORECAST, "rt", encoding="utf-8", newline="") as stream:
        forecasts = csv.DictReader(stream)
        if forecasts.fieldnames != ["timestamp", "lstm_forecast_kw"]:
            raise ValueError("Unexpected frozen forecast columns")
        for record, row in zip(evaluation, forecasts, strict=True):
            stamp = record.timestamp
            forecast = float(row["lstm_forecast_kw"])
            if row["timestamp"] != stamp.isoformat() or not math.isfinite(forecast) or forecast < 0:
                raise ValueError(f"Invalid frozen forecast interval at {stamp.isoformat()}")
            by_day.setdefault(stamp.date().isoformat(), []).append({
                "origin_timestamp": stamp.replace(hour=0, minute=0).isoformat(),
                "timestamp": stamp.isoformat(),
                "actual_demand_kw": record.demand_kw,
                "lstm_forecast_kw": forecast,
                "seasonal_naive_kw": truth[stamp - timedelta(days=7)],
            })
    return by_day


def prepare_design(prepared, output, inputs, retrained=None):
    dataset, periods, _ = load_prepared(prepared)
    _, days, selection = select_days(dataset, periods)
    if [day["date"] for day in days] != inputs["selected_dates"]:
        raise ValueError("Seasonal high-demand selection differs from the manuscript")
    if retrained:
        forecasts, _ = load_forecasts(retrained, prepared, dataset, periods)
    else:
        forecasts = load_frozen_forecasts(dataset, periods, inputs)
    daily_errors, selected_errors = selected_error_metrics(days, forecasts)
    _, annual_errors = selected_error_metrics([{"date": day, "season": "all", "season_rank": 0}
                                               for day in forecasts], forecasts)

    # Preserve the executed historical BESS setting. Do not resize from the expanded demand history.
    configs = battery_configs(REFERENCE_PEAK_KW)
    output.mkdir()
    write_csv(output / "selected_days.csv", days)
    write_csv(output / "selected_profiles.csv", [
        {**{key: day[key] for key in ("date", "season", "season_rank")}, **row}
        for day in days for row in forecasts[day["date"]]
    ])
    write_csv(output / "battery_configs.csv", configs)
    write_csv(output / "selected_forecast_daily_metrics.csv", daily_errors)
    save_json(output / "selected_forecast_metrics.json", selected_errors)
    save_json(output / "annual_forecast_metrics.json", annual_errors)
    save_json(output / "design.json", {
        "status": "design_only_not_simulated", "selected_days": len(days),
        "reference_peak_kw": REFERENCE_PEAK_KW, "battery_configs": configs,
        "selection_summary": selection,
    })
    save_json(output / "manifest.json", {
        "schema_version": 1, "status": "design_only_not_simulated",
        "dataset_id": inputs["dataset_id"], "prepared_manifest_sha256": sha256(prepared / "manifest.json"),
        "forecast_input_sha256": sha256(retrained / "evaluation_predictions.csv") if retrained else inputs["forecast_sha256"],
        "outputs": {path.name: {"sha256": sha256(path)} for path in sorted(output.iterdir())},
    })
    load_design(output)


def execute(script, *args, env=None):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                   check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="TEPCO input archive directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="New run directory")
    parser.add_argument("--download", action="store_true", help="Download any missing official archives")
    parser.add_argument("--retrain", action="store_true", help="Retrain LSTM instead of using manuscript predictions")
    parser.add_argument("--main-only", action="store_true", help="Omit the additional lambda and BESS configurations")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4, help="Parallel workers for lambda sweep")
    args = parser.parse_args()
    inputs = json.loads(INPUTS.read_text(encoding="utf-8"))
    root = args.out_dir.resolve()
    root.mkdir(parents=True)
    print("Preparing four years of Tokyo-area demand", flush=True)
    prepared = prepare(args.raw_dir.resolve(), root, inputs, args.download)
    retrained = root / "forecast" if args.retrain else None
    if retrained:
        execute("train_eneres_forecast.py", "--prepared-dir", prepared, "--out-dir", retrained,
                "--seed", 42, "--epochs", 100, "--patience", 12, "--threads", 2)
    design, baseline, shifts, corrections = (root / stage for stage in ("design", "baseline", "shifts", "corrections"))
    prepare_design(prepared, design, inputs, retrained)
    command = [sys.executable, *sys.argv]
    print("Solving baseline, all 17 shifts, and the ex post corrections", flush=True)
    run_baseline(design, baseline, command=command)
    run_shifts(design, baseline, shifts, command=command, reuse_zero_baseline=True)
    run_diagnostics(design, baseline, shifts, corrections, command=command)
    if not args.main_only:
        print("Solving the nine BESS configurations and the ten lambda values", flush=True)
        run_sensitivity(design, baseline, root / "bess", command=command)
        execute("run_tepco_lambda_sensitivity.py", "--design-dir", design,
                "--out-dir", root / "lambda", "--workers", args.workers)
    execute("make_figures_and_tables.py", root, env={
        **os.environ, "MPLCONFIGDIR": str(root / ".matplotlib"),
        "XDG_CACHE_HOME": str(root / ".cache"),
    })
    print(f"Figures and tables: {root / 'figures'} and {root / 'tables'}")


if __name__ == "__main__":
    main()
