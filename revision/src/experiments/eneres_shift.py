"""Frozen whole-profile cyclic shifts; actual demand never selects a planning shift."""

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pulp

from forecast.demand_data import _code_record
from experiments.eneres_bess import (
    CHARGE_TIEBREAK, ENERGY_TOLERANCE_KWH, OPTIMIZER_PATH, POWER_TOLERANCE_KW,
    REPO, SEASONS, SolverSettings, TIEBREAK_ACTION_TOLERANCE_KW,
    TIEBREAK_PEAK_TOLERANCE_KW, digest, evaluate_schedule, finite_profile,
    load_design, read_csv, solve_forecast, write_csv,
)

SHIFT_SLOTS = tuple(range(-8, 9))


def cyclic_shift(profile, shift_slots):
    """Positive delays, negative advances: shifted[t] = original[(t - shift) % 48]."""
    values = finite_profile(profile, 48)
    if isinstance(shift_slots, bool) or not isinstance(shift_slots, int) or shift_slots not in SHIFT_SLOTS:
        raise ValueError("Require an integer shift between -8 and +8 slots")
    return np.roll(values, shift_slots)


def load_baseline(directory, battery, design_hashes, settings):
    """Require the same v2 planning code, inputs, battery and CBC, before comparison."""
    directory = Path(directory)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    protocol = manifest["protocol"]
    if (manifest.get("status") != "baseline_simulated" or protocol["cases"] != 60
            or protocol["shift_slots"] != [0] or protocol.get("charge_tiebreak") != CHARGE_TIEBREAK):
        raise ValueError("Require the two-stage v2 baseline, not the superseded v1")
    expected_battery = {**asdict(battery), "final_soc_equality": True, "soc_min_fraction": 0, "soc_max_fraction": 1}
    if manifest["battery"] != expected_battery or manifest["solver"]["settings"] != asdict(settings):
        raise ValueError("Baseline battery or solver settings differ")
    solver = settings.make_solver()
    if not solver.available() or digest(Path(solver.path)) != manifest["solver"]["binary_sha256"]:
        raise ValueError("Baseline CBC binary differs")
    if manifest["solver"]["pulp_version"] != pulp.__version__:
        raise ValueError("Baseline PuLP version differs")
    for name, record in manifest["code"]["files"].items():
        if digest(REPO / name) != record["sha256"]:
            raise ValueError(f"Baseline planning code changed: {name}")
    if manifest["input_hashes"] != design_hashes:
        raise ValueError("Baseline and shifts must use exactly the same frozen design")
    hashes = {str(path.resolve()): digest(path)}
    for name, record in manifest["outputs"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unexpected baseline output path")
        p = directory / name
        if digest(p) != record["sha256"]:
            raise ValueError(f"Baseline hash mismatch: {name}")
        hashes[str(p.resolve())] = record["sha256"]
    metrics = {r["date"]: r for r in read_csv(directory / "daily_metrics.csv") if r["forecast_model"] == "lstm"}
    schedules = {}
    for row in read_csv(directory / "schedules.csv"):
        if row["forecast_model"] == "lstm":
            schedules.setdefault(row["date"], []).append(row)
    if len(metrics) != 20 or set(metrics) != set(schedules) or any(len(v) != 48 for v in schedules.values()):
        raise ValueError("Incomplete LSTM baseline")
    for group in schedules.values():
        group.sort(key=lambda row: row["timestamp"])
    return metrics, schedules, manifest, hashes


def compare_zero(day, metric, schedules, baseline_metrics, baseline_schedules):
    """Fresh zero-shift solves must reproduce v2 dispatch as well as peak metrics."""
    reference = baseline_metrics[day]
    maximum = 0.
    for key in ("actual_peak_after_kw", "actual_peak_reduction_kw", "solver_objective_kw",
                "primary_solver_objective_kw", "charge_tiebreak_objective"):
        error = abs(float(metric[key]) - float(reference[key]))
        if error > POWER_TOLERANCE_KW:
            raise ValueError(f"Zero-shift baseline metric differs: {day}, {key}")
        maximum = max(maximum, error)
    for row, old in zip(schedules, baseline_schedules[day]):
        if row["timestamp"] != old["timestamp"]:
            raise ValueError("Zero-shift baseline timestamp mismatch")
        for key in ("actual_demand_kw", "forecast_demand_kw", "charge_kw", "discharge_kw",
                    "soc_start_kwh", "soc_end_kwh", "controlled_actual_kw", "controlled_forecast_kw"):
            error = abs(float(row[key]) - float(old[key]))
            tolerance = ENERGY_TOLERANCE_KWH if key.endswith("kwh") else POWER_TOLERANCE_KW
            if error > tolerance:
                raise ValueError(f"Zero-shift baseline schedule differs: {day}, {key}")
            maximum = max(maximum, error)
    return {"date": day, "all_valid": True, "max_difference_kw_or_kwh": maximum}


def summarize_shifts(metrics):
    """Fixed-shift equal-day statistics, including harm and gain versus zero shift."""
    zeros = {r["date"]: r for r in metrics if r["shift_slots"] == 0}
    summary = []
    for season in ("all", *SEASONS):
        for shift in SHIFT_SLOTS:
            part = [r for r in metrics if r["shift_slots"] == shift and (season == "all" or r["season"] == season)]
            reductions = [r["actual_peak_reduction_percent"] for r in part]
            changes = [r["actual_peak_reduction_percent"] - zeros[r["date"]]["actual_peak_reduction_percent"] for r in part]
            differences_kw = [r["actual_peak_reduction_kw"] - zeros[r["date"]]["actual_peak_reduction_kw"] for r in part]
            summary.append({
                "season": season, "forecast_model": "lstm", "config_id": "baseline",
                "shift_slots": shift, "shift_hours": shift * .5, "days": len(part),
                "mean_actual_peak_reduction_kw": float(np.mean([r["actual_peak_reduction_kw"] for r in part])),
                "mean_actual_peak_reduction_percent": float(np.mean(reductions)),
                "median_actual_peak_reduction_percent": float(np.median(reductions)),
                "min_actual_peak_reduction_percent": float(np.min(reductions)),
                "max_actual_peak_reduction_percent": float(np.max(reductions)),
                "mean_change_vs_zero_percentage_points": float(np.mean(changes)),
                "worsened_vs_uncontrolled_days": sum(r["actual_peak_reduction_kw"] < -POWER_TOLERANCE_KW for r in part),
                "better_than_zero_days": sum(x > POWER_TOLERANCE_KW for x in differences_kw),
                "worse_than_zero_days": sum(x < -POWER_TOLERANCE_KW for x in differences_kw),
                "equivalent_to_zero_days": sum(abs(x) <= POWER_TOLERANCE_KW for x in differences_kw),
            })
    return summary


def retrospective_extrema(metrics):
    """Describe ex-post maxima/minima; these are not prospectively chosen forecasts."""
    extrema = []
    for day in sorted({r["date"] for r in metrics}):
        part = [r for r in metrics if r["date"] == day]
        # Numerically equivalent peaks prefer less shifting, then the negative shift.
        best_value = max(r["actual_peak_reduction_kw"] for r in part)
        worst_value = min(r["actual_peak_reduction_kw"] for r in part)
        best = min((r for r in part if best_value - r["actual_peak_reduction_kw"] <= POWER_TOLERANCE_KW),
                   key=lambda r: (abs(r["shift_slots"]), r["shift_slots"]))
        worst = min((r for r in part if r["actual_peak_reduction_kw"] - worst_value <= POWER_TOLERANCE_KW),
                    key=lambda r: (abs(r["shift_slots"]), r["shift_slots"]))
        zero = next(r for r in part if r["shift_slots"] == 0)
        extrema.append({"date": day, "season": zero["season"], "season_rank": zero["season_rank"],
                        "selection": "retrospective_using_actual_demand_not_operational",
                        "zero_reduction_percent": zero["actual_peak_reduction_percent"],
                        "best_shift_slots": best["shift_slots"], "best_reduction_percent": best["actual_peak_reduction_percent"],
                        "worst_shift_slots": worst["shift_slots"], "worst_reduction_percent": worst["actual_peak_reduction_percent"],
                        "best_gain_vs_zero_percentage_points": best["actual_peak_reduction_percent"] - zero["actual_peak_reduction_percent"]})
    return extrema


def run_shifts(design_dir, baseline_dir, output_dir, settings=SolverSettings(), command=None, progress=None,
               *, reuse_zero_baseline=False):
    """Prepare340 fixed cases, optionally reusing20 unchanged zero-shift cases."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    days, groups, battery, upstream, input_hashes = load_design(design_dir)
    baseline_metrics, baseline_schedules, baseline_manifest, baseline_hashes = load_baseline(
        baseline_dir, battery, input_hashes, settings)
    baseline_validations = {r["date"]: r for r in json.loads(
        (Path(baseline_dir) / "validation.json").read_text())["cases"] if r["forecast_model"] == "lstm"}
    code_paths = [Path(__file__), REPO / "src/experiments/eneres_bess.py", OPTIMIZER_PATH,
                  REPO / "scripts/run_eneres_shift.py", REPO / "src/forecast/demand_data.py",
                  REPO / "requirements-eneres-bess.txt"]
    code = _code_record(REPO, code_paths)
    solver = settings.make_solver()
    solver_binary = Path(solver.path).resolve()
    solver_hash = digest(solver_binary)
    version = subprocess.run([str(solver_binary), "-stop"], check=True, capture_output=True, text=True).stdout
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        (staging / "solver_logs").mkdir()
        metrics, schedules, validations, zero_checks = [], [], [], []
        for day in days:
            group = groups[day["date"]]
            actual = np.asarray([float(r["actual_demand_kw"]) for r in group])
            original = np.asarray([float(r["lstm_forecast_kw"]) for r in group])
            for shift in SHIFT_SLOTS:
                forecast = cyclic_shift(original, shift)
                case_id = f"{day['date']}_lstm_shift_{shift:+d}"
                reused = shift == 0 and reuse_zero_baseline
                if reused:
                    old = baseline_metrics[day["date"]]
                    old_rows = baseline_schedules[day["date"]]
                    meta_keys = {"case_id", "date", "season", "season_rank", "config_id", "forecast_model", "shift_slots"}
                    values = {k: (v if k.endswith("solver_status") or k.endswith("solver_solution_status")
                                  else v == "True" if k == "constraints_valid"
                                  else int(v) if k in ("actual_peak_after_slot", "actual_export_slots")
                                  else float(v)) for k, v in old.items() if k not in meta_keys}
                    result = {"c_t": [float(r["charge_kw"]) for r in old_rows],
                              "d_t": [float(r["discharge_kw"]) for r in old_rows],
                              "is_charging_t": [float(r["is_charging"]) for r in old_rows],
                              "soc_t": [float(old_rows[0]["soc_start_kwh"])] + [float(r["soc_end_kwh"]) for r in old_rows],
                              "independent_validation": baseline_validations[day["date"]]}
                    forecast_net = [float(r["controlled_forecast_kw"]) for r in old_rows]
                    actual_net = [float(r["controlled_actual_kw"]) for r in old_rows]
                    for suffix in (".log", ".tiebreak.log"):
                        shutil.copyfile(Path(baseline_dir) / "solver_logs" / f"{day['date']}_lstm{suffix}",
                                        staging / "solver_logs" / f"{case_id}{suffix}")
                else:
                    result = solve_forecast(forecast, battery, settings, staging / "solver_logs" / f"{case_id}.log")
                    values, forecast_net, actual_net = evaluate_schedule(actual, forecast, result, battery)
                meta = {"case_id": case_id, "date": day["date"], "season": day["season"],
                        "season_rank": int(day["season_rank"]), "config_id": "baseline", "forecast_model": "lstm",
                        "shift_slots": shift, "shift_hours": shift * .5,
                        "case_provenance": "reused_baseline" if reused else "new_optimization"}
                metric = {**meta, **values, "original_forecast_peak_slot": int(np.argmax(original)),
                          "shifted_forecast_peak_slot": int(np.argmax(forecast)), "actual_peak_slot": int(np.argmax(actual))}
                metrics.append(metric)
                validations.append({**result["independent_validation"], **meta})
                saved = [{**meta, "slot": i, "source_forecast_slot": (i - shift) % 48,
                          "timestamp": row["timestamp"], "forecast_origin_timestamp": row["origin_timestamp"],
                          "actual_demand_kw": actual[i], "original_forecast_kw": original[i], "forecast_demand_kw": forecast[i],
                          "charge_kw": result["c_t"][i], "discharge_kw": result["d_t"][i],
                          "is_charging": result["is_charging_t"][i], "soc_start_kwh": result["soc_t"][i],
                          "soc_end_kwh": result["soc_t"][i + 1], "controlled_forecast_kw": forecast_net[i],
                          "controlled_actual_kw": actual_net[i]} for i, row in enumerate(group)]
                schedules.extend(saved)
                if shift == 0:
                    zero_checks.append(compare_zero(day["date"], metric, saved, baseline_metrics, baseline_schedules))
                if progress is not None:
                    progress(metric, len(metrics), 340)
        if len(metrics) != 340 or len(schedules) != 16320 or len(zero_checks) != 20:
            raise ValueError("Incomplete shift experiment")
        summary = summarize_shifts(metrics)
        extrema = retrospective_extrema(metrics)
        write_csv(staging / "daily_metrics.csv", metrics)
        write_csv(staging / "schedules.csv", schedules)
        write_csv(staging / "summary.csv", summary)
        write_csv(staging / "retrospective_extrema.csv", extrema)
        def write_json(name, value):
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_json("validation.json", {"all_valid": True, "cases": validations, "zero_shift_baseline_checks": zero_checks,
                                       "tolerances": "Case-specific; recorded in cases[].tolerances; mode=1e-6"})
        (staging / "environment.txt").write_text("\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()})) + "\n")
        if (_code_record(REPO, code_paths) != code or digest(solver_binary) != solver_hash
                or any(digest(p) != h for p, h in {**input_hashes, **baseline_hashes}.items())):
            raise ValueError("Code, inputs, baseline or CBC changed during simulation")
        manifest = {
            "schema_version": 1, "status": "shift_sensitivity_simulated",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": list(command) if command is not None else None, "code": code,
            "input_hashes": input_hashes, "baseline_hashes": baseline_hashes,
            "upstream_design_manifest": upstream, "baseline_manifest_sha256": digest(Path(baseline_dir) / "manifest.json"),
            "battery": baseline_manifest["battery"],
            "solver": {"name": "PULP_CBC_CMD", "pulp_version": pulp.__version__, "binary": str(solver_binary),
                       "binary_sha256": solver_hash, "version_banner": version, "settings": asdict(settings),
                       "configuration": solver.toDict(), "acceptance": baseline_manifest["solver"]["acceptance"]},
            "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
            "protocol": {"cases": 340, "days": 20, "forecast_models": ["lstm"], "config_ids": ["baseline"],
                         "newly_optimized_cases": 320 if reuse_zero_baseline else 340,
                         "reused_zero_cases": 20 if reuse_zero_baseline else 0,
                         "solver_logs": "640 new and40 copied baseline logs" if reuse_zero_baseline else "680 new logs",
                         "shift_slots": list(SHIFT_SLOTS), "shift_formula": "shifted[t] = original[(t - shift_slots) % 48]",
                         "shift_sign": "positive delays; negative advances; cyclic within the same 48-slot day",
                         "planning": "Only the prespecified shifted LSTM forecast enters both MILP stages; no actual-demand selection or feedback",
                         "evaluation": "actual_demand_kw + charge_kw - discharge_kw; actual series and actions are not shifted or corrected",
                         "objective": baseline_manifest["protocol"]["objective"], "charge_tiebreak": CHARGE_TIEBREAK,
                         "tiebreak_tolerances": {"policy": "battery-scaled; see validation.json cases[].tolerances"},
                         "daily_soc": baseline_manifest["protocol"]["daily_soc"],
                         "grid_constraint": baseline_manifest["protocol"]["grid_constraint"],
                         "forecast_availability": baseline_manifest["protocol"]["forecast_availability"],
                         "statistics": "Equal-day means for each fixed shift; 20 selected high-demand days, not independent random annual samples",
                         "retrospective_extrema": "Selected using actual results after all runs; diagnostic only, not an operational strategy; ties within0.05kW prefer smallest absolute shift then negative",
                         "interpretation": "Whole-profile timing perturbation moves peaks, shoulders, valleys and the daily boundary; not an isolated peak-time causal effect"},
            "outputs": {str(p.relative_to(staging)): {"sha256": digest(p), "size_bytes": p.stat().st_size}
                        for p in sorted(staging.rglob("*")) if p.is_file()},
        }
        write_json("manifest.json", manifest)
        output_dir.mkdir()
        try:
            for path in staging.iterdir():
                path.rename(output_dir / path.name)
        except BaseException:
            shutil.rmtree(output_dir)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest, summary
