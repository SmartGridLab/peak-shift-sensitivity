"""Ex-post peak-time/peak-magnitude diagnostics, explicitly using target-day truth."""

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
    CHARGE_TIEBREAK, OPTIMIZER_PATH, POWER_TOLERANCE_KW, REPO, SEASONS,
    SolverSettings, digest, evaluate_schedule, finite_profile, load_design,
    read_csv, solve_forecast, validate_solution, write_csv,
)
from experiments.eneres_shift import compare_zero, load_baseline

CONDITIONS = ("original_lstm", "timing_oracle", "magnitude_oracle", "both_oracle", "perfect_foresight_oracle")
CORRECTIONS = CONDITIONS[1:4]


def correction_parameters(actual, original):
    """First maxima; shortest signed cyclic shift, with a half-day tie advanced."""
    actual, original = finite_profile(actual, 48), finite_profile(original, 48)
    ap, fp = float(actual.max()), float(original.max())
    if ap <= 0 or fp <= 0:
        raise ValueError("Positive actual and forecast peaks required for oracle scaling")
    ai, fi = int(np.argmax(actual)), int(np.argmax(original))
    return {"actual_peak_slot": ai, "original_peak_slot": fi,
            "actual_peak_kw": ap, "original_peak_kw": fp,
            "actual_peak_ties": int(np.count_nonzero(actual == ap)),
            "original_peak_ties": int(np.count_nonzero(original == fp)),
            "unwrapped_timing_shift_slots": ai - fi,
            "oracle_timing_shift_slots": (ai - fi + 24) % 48 - 24,
            "oracle_magnitude_scale": ap / fp}


def corrected_profile(actual, original, condition):
    """Correct by peak values, never by searching control outcomes; do not mutate inputs."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown diagnostic condition")
    p = correction_parameters(actual, original)
    shift = p["oracle_timing_shift_slots"] if condition in ("timing_oracle", "both_oracle") else 0
    scale = p["oracle_magnitude_scale"] if condition in ("magnitude_oracle", "both_oracle") else 1.
    forecast = finite_profile(actual, 48).copy() if condition == "perfect_foresight_oracle" else np.roll(finite_profile(original, 48), shift) * scale
    return forecast, shift, scale


def prediction_metrics(actual, forecast):
    """Pointwise errors and first-maximum errors, separate from the aligned source peak."""
    actual, forecast = finite_profile(actual, 48), finite_profile(forecast, 48)
    if actual.max() <= 0:
        raise ValueError("Positive actual peak required")
    error = forecast - actual
    positive = actual > 0
    ai, fi = int(np.argmax(actual)), int(np.argmax(forecast))
    signed = fi - ai
    return {"rmse_kw": float(np.sqrt(np.mean(error ** 2))), "mae_kw": float(np.mean(np.abs(error))),
            "wape_percent": float(np.sum(np.abs(error)) / np.sum(actual) * 100),
            "mape_percent": float(np.mean(np.abs(error[positive] / actual[positive])) * 100),
            "mape_valid_slots": int(np.count_nonzero(positive)),
            "first_peak_time_error_slots": signed, "absolute_first_peak_time_error_minutes": abs(signed) * 30,
            "circular_first_peak_time_error_slots": (signed + 24) % 48 - 24,
            "peak_magnitude_error_kw": float(forecast.max() - actual.max()),
            "absolute_peak_magnitude_error_kw": float(abs(forecast.max() - actual.max())),
            "forecast_daily_energy_kwh": float(np.sum(forecast) * .5),
            "actual_daily_energy_kwh": float(np.sum(actual) * .5)}


def load_shift_reference(directory, design_hashes, baseline_manifest, baseline_dir):
    """Read-only comparison with the frozen step-6 experiment, not shift selection."""
    directory = Path(directory)
    path = directory / "manifest.json"
    m = json.loads(path.read_text())
    if (m.get("status") != "shift_sensitivity_simulated" or m["protocol"]["cases"] != 340
            or m["protocol"]["shift_slots"] != list(range(-8, 9)) or m["input_hashes"] != design_hashes
            or m["battery"] != baseline_manifest["battery"]
            or any(m["solver"][key] != baseline_manifest["solver"][key]
                   for key in ("name", "pulp_version", "binary_sha256", "settings", "configuration"))
            or m["baseline_manifest_sha256"] != digest(Path(baseline_dir) / "manifest.json")):
        raise ValueError("Step-6 input, baseline, solver or protocol differs")
    for name, record in m["code"]["files"].items():
        if digest(REPO / name) != record["sha256"]:
            raise ValueError(f"Step-6 planning code changed: {name}")
    hashes = {str(path.resolve()): digest(path)}
    for name, record in m["outputs"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unexpected step-6 output path")
        p = directory / name
        if digest(p) != record["sha256"]:
            raise ValueError(f"Step-6 output changed: {name}")
        hashes[str(p.resolve())] = record["sha256"]
    metrics = {(r["date"], int(r["shift_slots"])): r for r in read_csv(directory / "daily_metrics.csv")}
    groups = {}
    for r in read_csv(directory / "schedules.csv"):
        groups.setdefault((r["date"], int(r["shift_slots"])), []).append(r)
    if len(metrics) != 340 or set(metrics) != set(groups) or any(len(g) != 48 for g in groups.values()):
        raise ValueError("Incomplete step-6 reference")
    for g in groups.values():
        g.sort(key=lambda r: r["timestamp"])
    return metrics, groups, hashes


def baseline_result(metric, group, forecast, battery):
    """Reconstruct and validate recorded reference actions; do not rerun their solver."""
    def values(name):
        return np.asarray([float(r[name]) for r in group])
    result = {"c_t": values("charge_kw"), "d_t": values("discharge_kw"),
              "soc_t": np.r_[float(group[0]["soc_start_kwh"]), values("soc_end_kwh")],
              "demand_t": values("controlled_forecast_kw"), "is_charging_t": values("is_charging"),
              "demand_peak_forecast": float(metric["forecast_peak_after_kw"]),
              "objective_value": float(metric["solver_objective_kw"]),
              "problem_status": metric["solver_status"], "solution_status": metric["solver_solution_status"],
              "charge_tiebreak": {"primary": {"objective_value": float(metric["primary_solver_objective_kw"]),
                                               "peak_kw": float(metric["primary_peak_kw"]),
                                               "action_sum_kw": float(metric["primary_action_sum_kw"]),
                                               "problem_status": metric["primary_solver_status"], "solution_status": metric["primary_solver_solution_status"]},
                                  "weights": forecast / (float(np.max(forecast)) or 1.),
                                  "objective_value": float(metric["charge_tiebreak_objective"])},
              "elapsed_wall_s": 0.}
    if any(stage["problem_status"] != "Optimal" or stage["solution_status"] != "Optimal Solution Found"
           for stage in (result, result["charge_tiebreak"]["primary"])):
        raise ValueError("Unverified baseline reference")
    result["independent_validation"] = validate_solution(forecast, result, battery)
    return result


def paired_effects(metrics):
    """Conditional contrasts and nonadditivity, not isolated causal contributions."""
    output = []
    for day in sorted({r["date"] for r in metrics}):
        r = {row["condition"]: row for row in metrics if row["date"] == day}
        values = {c: r[c]["actual_peak_reduction_percent"] for c in CONDITIONS}
        original, time, magnitude, both, perfect = (values[c] for c in CONDITIONS)
        output.append({"date": day, "season": r["original_lstm"]["season"],
                       "original_reduction_percent": original, "timing_reduction_percent": time,
                       "magnitude_reduction_percent": magnitude, "both_reduction_percent": both,
                       "perfect_reduction_percent": perfect, "timing_change_pp": time - original,
                       "magnitude_change_pp": magnitude - original, "both_change_pp": both - original,
                       "timing_change_after_magnitude_pp": both - magnitude,
                       "magnitude_change_after_timing_pp": both - time,
                       "nonadditivity_pp": both - time - magnitude + original})
    return output


def summarize_diagnostics(metrics):
    """Equal-day paired summaries, retaining negative changes and peak worsening."""
    originals = {r["date"]: r for r in metrics if r["condition"] == "original_lstm"}
    summary = []
    for season in ("all", *SEASONS):
        for condition in CONDITIONS:
            part = [r for r in metrics if r["condition"] == condition and (season == "all" or r["season"] == season)]
            pct = [r["actual_peak_reduction_percent"] for r in part]
            differences = [r["actual_peak_reduction_kw"] - originals[r["date"]]["actual_peak_reduction_kw"] for r in part]
            record = {"season": season, "condition": condition, "days": len(part),
                      "mean_actual_peak_reduction_kw": float(np.mean([r["actual_peak_reduction_kw"] for r in part])),
                      "mean_actual_peak_reduction_percent": float(np.mean(pct)),
                      "median_actual_peak_reduction_percent": float(np.median(pct)),
                      "min_actual_peak_reduction_percent": float(np.min(pct)), "max_actual_peak_reduction_percent": float(np.max(pct)),
                      "mean_change_vs_original_pp": float(np.mean([r["actual_peak_reduction_percent"] - originals[r["date"]]["actual_peak_reduction_percent"] for r in part])),
                      "better_than_original_days": sum(x > POWER_TOLERANCE_KW for x in differences),
                      "worse_than_original_days": sum(x < -POWER_TOLERANCE_KW for x in differences),
                      "equivalent_to_original_days": sum(abs(x) <= POWER_TOLERANCE_KW for x in differences),
                      "worsened_vs_uncontrolled_days": sum(r["actual_peak_reduction_kw"] < -POWER_TOLERANCE_KW for r in part)}
            for key in ("rmse_kw", "mae_kw", "wape_percent", "mape_percent", "absolute_first_peak_time_error_minutes", "absolute_peak_magnitude_error_kw"):
                record[f"mean_{key}"] = float(np.mean([r[key] for r in part]))
            summary.append(record)
    return summary


def run_diagnostics(design_dir, baseline_dir, shift_dir, output_dir, settings=SolverSettings(), command=None, progress=None):
    """Compute 60 correction cases and revalidate 40 inherited baseline references."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    days, profiles, battery, upstream, input_hashes = load_design(design_dir)
    _, _, baseline, baseline_hashes = load_baseline(baseline_dir, battery, input_hashes, settings)
    shift_metrics, shift_schedules, shift_hashes = load_shift_reference(shift_dir, input_hashes, baseline, baseline_dir)
    baseline_metrics = {(r["date"], r["forecast_model"]): r for r in read_csv(Path(baseline_dir) / "daily_metrics.csv")}
    baseline_groups = {}
    for row in read_csv(Path(baseline_dir) / "schedules.csv"):
        baseline_groups.setdefault((row["date"], row["forecast_model"]), []).append(row)
    for g in baseline_groups.values():
        g.sort(key=lambda r: r["timestamp"])
    code_paths = [Path(__file__), REPO / "src/experiments/eneres_shift.py", REPO / "src/experiments/eneres_bess.py",
                  OPTIMIZER_PATH, REPO / "scripts/run_eneres_oracle.py", REPO / "src/forecast/demand_data.py", REPO / "requirements-eneres-bess.txt"]
    code = _code_record(REPO, code_paths)
    solver = settings.make_solver()
    binary = Path(solver.path).resolve()
    solver_hash = digest(binary)
    version = subprocess.run([str(binary), "-stop"], check=True, capture_output=True, text=True).stdout
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        (staging / "solver_logs").mkdir()
        metrics, schedules, parameters, validations, reference_checks, timing_checks = [], [], [], [], [], []
        solved = 0
        for day in days:
            group = profiles[day["date"]]
            actual = np.asarray([float(r["actual_demand_kw"]) for r in group])
            original = np.asarray([float(r["lstm_forecast_kw"]) for r in group])
            p = correction_parameters(actual, original)
            parameters.append({"date": day["date"], "season": day["season"], "season_rank": int(day["season_rank"]), **p})
            for condition in CONDITIONS:
                forecast, shift, scale = corrected_profile(actual, original, condition)
                case_id = f"{day['date']}_{condition}"
                model = "perfect_foresight_oracle" if condition == "perfect_foresight_oracle" else "lstm"
                inherited = condition in ("original_lstm", "perfect_foresight_oracle")
                if inherited:
                    old = baseline_metrics[day["date"], model]
                    result = baseline_result(old, baseline_groups[day["date"], model], forecast, battery)
                    for suffix in (".log", ".tiebreak.log"):
                        source = Path(baseline_dir) / "solver_logs" / f"{old['case_id']}{suffix}"
                        if "Result - Optimal solution found" not in source.read_text():
                            raise ValueError("Missing baseline optimality log")
                        shutil.copyfile(source, staging / "solver_logs" / f"{case_id}{suffix}")
                else:
                    result = solve_forecast(forecast, battery, settings, staging / "solver_logs" / f"{case_id}.log")
                    solved += 1
                values, predicted_net, actual_net = evaluate_schedule(actual, forecast, result, battery)
                meta = {"case_id": case_id, "date": day["date"], "season": day["season"], "season_rank": int(day["season_rank"]),
                        "config_id": "baseline", "forecast_model": model, "condition": condition,
                        "shift_slots": shift, "magnitude_scale": scale,
                        "uses_target_day_actual_in_planning": condition != "original_lstm",
                        "execution": "reused_baseline_v2" if inherited else "new_two_stage_solve"}
                record = {**meta, **values, **prediction_metrics(actual, forecast),
                          "mapped_original_peak_slot": (p["original_peak_slot"] + shift) % 48,
                          "forecast_peak_slot": int(np.argmax(forecast)),
                          "source_case_id": old["case_id"] if inherited else case_id}
                metrics.append(record)
                validations.append({**meta, **result["independent_validation"]})
                saved = [{**meta, "slot": i, "source_forecast_slot": (i - shift) % 48,
                          "timestamp": row["timestamp"], "forecast_origin_timestamp": row["origin_timestamp"],
                          "actual_demand_kw": actual[i], "original_forecast_kw": original[i], "forecast_demand_kw": forecast[i],
                          "charge_kw": result["c_t"][i], "discharge_kw": result["d_t"][i], "is_charging": result["is_charging_t"][i],
                          "soc_start_kwh": result["soc_t"][i], "soc_end_kwh": result["soc_t"][i + 1],
                          "controlled_forecast_kw": predicted_net[i], "controlled_actual_kw": actual_net[i]} for i, row in enumerate(group)]
                schedules.extend(saved)
                if inherited:
                    check = compare_zero(day["date"], record, saved, {day["date"]: old}, {day["date"]: baseline_groups[day["date"], model]})
                    reference_checks.append({"condition": condition, **check})
                elif condition == "timing_oracle" and (day["date"], shift) in shift_metrics:
                    key = day["date"], shift
                    check = compare_zero(day["date"], record, saved, {day["date"]: shift_metrics[key]}, {day["date"]: shift_schedules[key]})
                    timing_checks.append({"shift_slots": shift, **check})
                if progress is not None:
                    progress(record, len(metrics), 100)
        if (len(metrics), len(schedules), solved, len(reference_checks)) != (100, 4800, 60, 40):
            raise ValueError("Incomplete oracle diagnostic")
        summary = summarize_diagnostics(metrics)
        for name, rows in (("daily_metrics.csv", metrics), ("schedules.csv", schedules), ("summary.csv", summary),
                           ("correction_parameters.csv", parameters), ("paired_effects.csv", paired_effects(metrics))):
            write_csv(staging / name, rows)
        def write_json(name, value):
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_json("validation.json", {"all_valid": True, "cases": validations, "baseline_reference_checks": reference_checks,
                                       "timing_matches_step6": timing_checks,
                                       "tolerances": "Case-specific; recorded in cases[].tolerances; mode=1e-6"})
        (staging / "environment.txt").write_text("\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()})) + "\n")
        if (_code_record(REPO, code_paths) != code or digest(binary) != solver_hash
                or any(digest(p) != h for p, h in {**input_hashes, **baseline_hashes, **shift_hashes}.items())):
            raise ValueError("Code, input, reference or solver changed during diagnostics")
        manifest = {"schema_version": 1, "status": "oracle_diagnostics_completed",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "command": list(command) if command is not None else None, "code": code,
                    "input_hashes": input_hashes, "baseline_hashes": baseline_hashes, "shift_hashes": shift_hashes,
                    "upstream_design_manifest": upstream, "battery": baseline["battery"],
                    "solver": {"name": "PULP_CBC_CMD", "pulp_version": pulp.__version__, "binary": str(binary),
                               "binary_sha256": solver_hash, "version_banner": version, "settings": asdict(settings),
                               "configuration": solver.toDict(), "acceptance": baseline["solver"]["acceptance"]},
                    "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
                    "protocol": {"cases": 100, "newly_solved_cases": 60, "reused_baseline_cases": 40, "days": 20,
                                 "conditions": list(CONDITIONS), "config_ids": ["baseline"],
                                 "timing": "s = (first_argmax(actual) - first_argmax(original) + 24) % 48 - 24; corrected[t] = original[(t-s)%48]; half-day tie advances",
                                 "magnitude": "a = max(actual) / max(original); multiply the whole forecast by a; no clipping or energy preservation",
                                 "both": "Apply the same time shift and positive magnitude scale; operations commute",
                                 "ties": "Choose first maxima of original and actual before shifting; selected source peak aligns, but an earlier tied maximum may remain after wrapping",
                                 "planning": "Oracle corrections deliberately use target-day actual peak time/value; optimizer receives only the resulting profile; no search over control outcomes or corrective dispatch",
                                 "evaluation": "Fixed actual_demand_kw + charge_kw - discharge_kw for every condition",
                                 "objective": baseline["protocol"]["objective"], "charge_tiebreak": CHARGE_TIEBREAK,
                                 "daily_soc": baseline["protocol"]["daily_soc"], "grid_constraint": baseline["protocol"]["grid_constraint"],
                                 "forecast_availability": baseline["protocol"]["forecast_availability"],
                                 "statistics": "Equal-day means and paired contrasts on 20 selected high-demand days; not independent random annual samples",
                                 "interpretation": "Ex-post diagnostics, not deployable forecasts; whole-profile shifts and scaling change valleys, shoulders, charging, boundary and daily energy; no independent causal contribution fractions",
                                 "references": "Original LSTM and perfect foresight actions/optimality logs reused from v2, revalidated and reevaluated; 3 corrections newly solved"},
                    "outputs": {str(p.relative_to(staging)): {"sha256": digest(p), "size_bytes": p.stat().st_size}
                                for p in sorted(staging.rglob("*")) if p.is_file()}}
        write_json("manifest.json", manifest)
        output_dir.mkdir()
        try:
            for p in staging.iterdir():
                p.rename(output_dir / p.name)
        except BaseException:
            shutil.rmtree(output_dir)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest, summary
