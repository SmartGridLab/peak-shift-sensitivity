#!/usr/bin/env python3
"""R1-7: fresh lambda sweep, keeping frozen TEPCO forecasts and dispatch rules."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import pulp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.eneres_bess import (
    BatterySpec, CHARGE_TIEBREAK, OPTIMIZER_PATH, REPO, SolverSettings,
    digest, evaluate_schedule, load_design, read_csv, solve_forecast, write_csv,
)
from experiments.eneres_oracle import corrected_profile
from forecast.demand_data import _code_record

LAMBDAS = (0., 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2, 1e-1, 1.)
EXTRAS = ("seasonal_naive", "perfect_foresight_oracle", "timing_oracle", "magnitude_oracle", "both_oracle")


@dataclass(frozen=True)
class LambdaBatterySpec(BatterySpec):
    """Relax only the fixed-lambda experiment guard; retain physical validation."""

    def __post_init__(self):
        if not math.isfinite(self.action_penalty_lambda) or self.action_penalty_lambda < 0:
            raise ValueError("Lambda must be finite and nonnegative")
        base = asdict(self)
        base["action_penalty_lambda"] = 1e-4
        BatterySpec(**base)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_one(design_dir, directory, penalty, settings):
    """Each independent lambda uses the identical frozen solver in both stages."""
    directory = Path(directory)
    directory.mkdir()
    (directory / "solver_logs").mkdir()
    days, groups, base, _, _ = load_design(design_dir)
    spec = asdict(base)
    spec["action_penalty_lambda"] = penalty
    battery = LambdaBatterySpec(**spec)
    metrics, schedules, validations = [], [], []
    started = perf_counter()
    for day in days:
        group = groups[day["date"]]
        actual = np.asarray([float(r["actual_demand_kw"]) for r in group])
        original = np.asarray([float(r["lstm_forecast_kw"]) for r in group])
        cases = [("lstm_shift", s, np.roll(original, s), 1.) for s in range(-8, 9)]
        for condition in EXTRAS:
            if condition == "seasonal_naive":
                forecast = np.asarray([float(r["seasonal_naive_kw"]) for r in group])
                shift, scale = 0, 1.
            else:
                forecast, shift, scale = corrected_profile(actual, original, condition)
            cases.append((condition, shift, forecast, scale))
        for condition, shift, forecast, scale in cases:
            case_id = f"{day['date']}_{condition}_{shift:+d}"
            result = solve_forecast(forecast, battery, settings, directory / "solver_logs" / f"{case_id}.log")
            metric, predicted_net, actual_net = evaluate_schedule(actual, forecast, result, battery)
            primary = result["charge_tiebreak"]["primary"]
            actions = float(np.sum(result["c_t"] + result["d_t"]))
            meta = {"case_id": case_id, "lambda": penalty, "date": day["date"],
                    "season": day["season"], "season_rank": int(day["season_rank"]),
                    "config_id": "baseline", "condition": condition, "shift_slots": shift,
                    "magnitude_scale": scale}
            metrics.append({**meta, **metric, "primary_peak_kw": primary["peak_kw"],
                            "primary_action_sum_kw": primary["action_sum_kw"],
                            "action_sum_kw": actions,
                            "action_penalty_kw": penalty * actions,
                            "action_penalty_to_forecast_peak_percent": 100 * penalty * actions / float(predicted_net.max()),
                            "stage2_peak_increase_kw": float(predicted_net.max()) - primary["peak_kw"],
                            "stage2_action_increase_kw": actions - primary["action_sum_kw"]})
            validations.append({**meta, **result["independent_validation"]})
            for i, row in enumerate(group):
                schedules.append({**meta, "timestamp": row["timestamp"],
                                  "forecast_origin_timestamp": row["origin_timestamp"],
                                  "actual_demand_kw": float(actual[i]), "forecast_demand_kw": float(forecast[i]),
                                  "charge_kw": float(result["c_t"][i]), "discharge_kw": float(result["d_t"][i]),
                                  "is_charging": float(result["is_charging_t"][i]),
                                  "soc_start_kwh": float(result["soc_t"][i]), "soc_end_kwh": float(result["soc_t"][i + 1]),
                                  "controlled_forecast_kw": float(predicted_net[i]), "controlled_actual_kw": float(actual_net[i])})
        if len(metrics) % 110 == 0:
            print(f"lambda={penalty:g}: {len(metrics)}/440 verified", flush=True)
    write_csv(directory / "daily_metrics.csv", metrics)
    write_csv(directory / "schedules.csv", schedules)
    dump(directory / "validation.json", {"all_valid": True, "cases": validations})
    return {"lambda": penalty, "directory": directory.name, "cases": len(metrics), "wall_s": perf_counter() - started}


def summarize(metrics):
    records = []
    for penalty in sorted({r["lambda"] for r in metrics}):
        subset = [r for r in metrics if r["lambda"] == penalty]
        for season in ("all", "spring", "summer", "autumn", "winter"):
            for condition, shift in [("lstm_shift", s) for s in range(-8, 9)] + [(c, None) for c in EXTRAS]:
                part = [r for r in subset if r["condition"] == condition
                        and (shift is None or r["shift_slots"] == shift)
                        and (season == "all" or r["season"] == season)]
                v = np.asarray([r["actual_peak_reduction_percent"] for r in part])
                records.append({"lambda": penalty, "season": season, "condition": condition,
                                "shift_slots": shift if shift is not None else "", "days": len(part),
                                "mean_reduction_percent": float(v.mean()), "median_reduction_percent": float(np.median(v)),
                                "min_reduction_percent": float(v.min()), "max_reduction_percent": float(v.max()),
                                "negative_days": int(np.count_nonzero(v < -1e-6)), "zero_days": int(np.count_nonzero(np.abs(v) <= 1e-6)),
                                "mean_charge_energy_kwh": float(np.mean([r["charge_energy_kwh"] for r in part])),
                                "mean_discharge_energy_kwh": float(np.mean([r["discharge_energy_kwh"] for r in part])),
                                "mean_primary_peak_kw": float(np.mean([r["primary_peak_kw"] for r in part])),
                                "mean_primary_action_sum_kw": float(np.mean([r["primary_action_sum_kw"] for r in part])),
                                "mean_action_penalty_kw": float(np.mean([r["action_penalty_kw"] for r in part])),
                                "mean_solver_elapsed_wall_s": float(np.mean([r["solver_elapsed_wall_s"] for r in part]))})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS))
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    penalties = sorted(set(args.lambdas))
    if len(penalties) != len(args.lambdas) or any(not math.isfinite(v) or v < 0 for v in penalties):
        raise ValueError("Require distinct finite nonnegative lambdas")
    if not 1 <= args.workers <= 8:
        raise ValueError("Workers must lie in [1,8]")
    _, _, battery, upstream, input_hashes = load_design(args.design_dir)
    paths = [Path(__file__), OPTIMIZER_PATH, REPO / "src/experiments/eneres_bess.py",
             REPO / "src/experiments/eneres_oracle.py", REPO / "src/experiments/eneres_shift.py",
             REPO / "src/forecast/demand_data.py", REPO / "requirements-eneres-bess.txt"]
    code = _code_record(REPO, paths)
    settings = SolverSettings()
    solver = settings.make_solver()
    solver_path = Path(solver.path).resolve()
    solver_hash = digest(solver_path)
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "environment.txt").write_text("\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()})) + "\n")
    manifest = {"schema_version": 1, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": [sys.executable, *sys.argv], "code": code, "input_hashes": input_hashes,
                "upstream_design_manifest": upstream, "battery_nominal": asdict(battery),
                "lambdas": penalties, "workers": args.workers,
                "solver": {"pulp_version": pulp.__version__, "binary": str(solver_path), "binary_sha256": solver_hash,
                           "version_banner": subprocess.run([str(solver_path), "-stop"], check=True, capture_output=True, text=True).stdout,
                           "settings": asdict(settings), "configuration": solver.toDict()},
                "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
                "protocol": {"days": 20, "cases_per_lambda": 440, "fresh_cases": 440 * len(penalties),
                             "stages_per_case": 2, "shift_slots": list(range(-8, 9)), "extra_conditions": list(EXTRAS),
                             "objective": "peak_forecast_kw + lambda * sum(charge_kw + discharge_kw); lambda dimensionless; no dt in penalty",
                             "charge_tiebreak": CHARGE_TIEBREAK,
                             "tiebreak_peak_tolerance_kw": 10., "tiebreak_action_tolerance_kw": 1000.,
                             "physical_tolerances": "Battery-scaled; recorded per case in validation.json; mode=1e-6",
                             "evaluation": "Actual + fixed charge - fixed discharge; same 20 days; no retraining or dispatch correction",
                             "scope": "Baseline BESS only; fixed stage-2 selection rule; ex-post corrections and shift comparisons are diagnostics, not operational tuning"}}
    dump(args.out_dir / "manifest.json", manifest)
    started = perf_counter()
    runs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(run_one, str(args.design_dir), str(args.out_dir / f"lambda_{i:02d}"), penalty, settings)
                 for i, penalty in enumerate(penalties)]
        for task in as_completed(tasks):
            record = task.result()
            runs.append(record)
            print(f"Completed lambda={record['lambda']:g}: {record['cases']} cases, {record['wall_s']:.1f}s", flush=True)
    metrics = []
    for record in sorted(runs, key=lambda r: r["lambda"]):
        for row in read_csv(args.out_dir / record["directory"] / "daily_metrics.csv"):
            for key in ("lambda", "actual_peak_reduction_percent", "charge_energy_kwh", "discharge_energy_kwh",
                        "primary_peak_kw", "primary_action_sum_kw", "action_penalty_kw", "solver_elapsed_wall_s"):
                row[key] = float(row[key])
            row["shift_slots"] = int(row["shift_slots"])
            metrics.append(row)
    if len(metrics) != 440 * len(penalties):
        raise ValueError("Incomplete lambda sweep")
    write_csv(args.out_dir / "summary.csv", summarize(metrics))
    if _code_record(REPO, paths) != code or any(digest(p) != h for p, h in input_hashes.items()) or digest(solver_path) != solver_hash:
        raise ValueError("Code, inputs or CBC changed during simulation")
    manifest.update(status="lambda_sensitivity_simulated", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                    wall_s=perf_counter() - started, runs=sorted(runs, key=lambda r: r["lambda"]))
    manifest["outputs"] = {str(p.relative_to(args.out_dir)): {"sha256": digest(p), "size_bytes": p.stat().st_size}
                           for p in sorted(args.out_dir.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    dump(args.out_dir / "manifest.json", manifest)
    print(f"Saved {len(metrics)} fresh verified cases in {manifest['wall_s']:.1f}s: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
