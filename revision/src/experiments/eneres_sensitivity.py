"""Frozen one-factor BESS scenarios, with unmodified forecasts and zero shift."""

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import math
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
    BatterySpec, FORECASTS, OPTIMIZER_PATH, POWER_TOLERANCE_KW, REPO,
    SolverSettings, digest, evaluate_schedule, load_design, read_csv,
    solve_forecast, summarize, write_csv,
)
from experiments.eneres_shift import compare_zero, load_baseline
from experiments.eneres_oracle import baseline_result

CONFIG_IDS = ("baseline", "power_10pct", "power_30pct", "duration_2h", "duration_6h",
              "soc_25pct", "soc_75pct", "roundtrip_80pct", "roundtrip_90pct")


def validate_configs(configs):
    """Verify every frozen physical and derived field, without changing the design."""
    if len(configs) != 9 or {c["config_id"] for c in configs} != set(CONFIG_IDS):
        raise ValueError("Exactly the nine planned BESS configurations are required")
    indexed = {c["config_id"]: c for c in configs}
    reference = float(indexed["baseline"]["reference_peak_kw"])
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Positive finite reference peak required")
    specs = {}
    for name in CONFIG_IDS:
        c = indexed[name]
        power, capacity, soc, rte, axis = reference * .2, reference * .2 * 4, .5, .85, "baseline"
        if name.startswith("power_"):
            power, axis = reference * (.1 if name == "power_10pct" else .3), "power"
        elif name.startswith("duration_"):
            capacity, axis = power * (2 if name == "duration_2h" else 6), "capacity"
        elif name.startswith("soc_"):
            soc, axis = (.25 if name == "soc_25pct" else .75), "initial_soc"
        elif name.startswith("roundtrip_"):
            rte, axis = (.8 if name == "roundtrip_80pct" else .9), "efficiency"
        expected = {"config_id": name, "varied_axis": axis, "reference_peak_kw": reference,
                    "max_power_kw": power, "battery_capacity_kwh": capacity,
                    "power_to_reference_peak_ratio": power / reference, "duration_hours": capacity / power,
                    "initial_soc_fraction": soc, "initial_soc_kwh": capacity * soc,
                    "round_trip_efficiency": rte, "charge_efficiency": math.sqrt(rte),
                    "discharge_efficiency": math.sqrt(rte), "soc_min_fraction": 0., "soc_max_fraction": 1.,
                    "final_soc_equality": True, "action_penalty_lambda": 1e-4, "timestep_h": .5}
        if set(c) != set(expected):
            raise ValueError(f"Missing or extra configuration fields: {name}")
        for key, value in expected.items():
            saved = c[key]
            matches = saved is value if isinstance(value, bool) else saved == value if isinstance(value, str) else (
                not isinstance(saved, bool) and math.isclose(float(saved), value, rel_tol=1e-12, abs_tol=1e-12))
            if not matches:
                raise ValueError(f"Frozen one-factor configuration mismatch: {name}, {key}")
        specs[name] = BatterySpec(**{k: float(c[k]) for k in BatterySpec.__dataclass_fields__})
    return specs, [indexed[n] for n in CONFIG_IDS]


def load_configs(design_dir):
    directory = Path(design_dir)
    specs, configs = validate_configs(json.loads((directory / "design.json").read_text())["battery_configs"])
    csv_rows = read_csv(directory / "battery_configs.csv")
    if len(csv_rows) != 9 or {r["config_id"] for r in csv_rows} != set(CONFIG_IDS):
        raise ValueError("Battery CSV configurations incomplete or duplicated")
    saved = {r["config_id"]: r for r in csv_rows}
    for c in configs:
        if set(saved[c["config_id"]]) != set(c):
            raise ValueError("Battery CSV fields differ")
        for key, value in c.items():
            s = saved[c["config_id"]][key]
            matches = s == str(value) if isinstance(value, (str, bool)) else float(s) == value
            if not matches:
                raise ValueError(f"Battery CSV/JSON mismatch: {c['config_id']}, {key}")
    return specs, configs


def summarize_sensitivity(metrics):
    """Same-day, same-model contrasts against baseline; never clip worsening."""
    baseline = {(r["date"], r["forecast_model"]): r for r in metrics if r["config_id"] == "baseline"}
    result = []
    for config in CONFIG_IDS:
        subset = [r for r in metrics if r["config_id"] == config]
        for saved in summarize(subset):
            part = [r for r in subset if r["forecast_model"] == saved["forecast_model"]
                    and (saved["season"] == "all" or r["season"] == saved["season"])]
            changes = [r["actual_peak_reduction_kw"] - baseline[r["date"], r["forecast_model"]]["actual_peak_reduction_kw"] for r in part]
            result.append({"config_id": config, "shift_slots": 0, **saved,
                           "mean_change_vs_baseline_pp": float(np.mean([r["actual_peak_reduction_percent"] - baseline[r["date"], r["forecast_model"]]["actual_peak_reduction_percent"] for r in part])),
                           "better_than_baseline_days": sum(x > POWER_TOLERANCE_KW for x in changes),
                           "worse_than_baseline_days": sum(x < -POWER_TOLERANCE_KW for x in changes),
                           "equivalent_to_baseline_days": sum(abs(x) <= POWER_TOLERANCE_KW for x in changes)})
    return result


def model_comparisons(metrics):
    lookup = {(r["config_id"], r["date"], r["forecast_model"]): r for r in metrics}
    output = []
    for config in CONFIG_IDS:
        for day in sorted({r["date"] for r in metrics}):
            lstm, naive, oracle = (lookup[config, day, m] for m in FORECASTS)
            output.append({"config_id": config, "date": day, "season": lstm["season"], "shift_slots": 0,
                           "lstm_reduction_percent": lstm["actual_peak_reduction_percent"],
                           "seasonal_naive_reduction_percent": naive["actual_peak_reduction_percent"],
                           "oracle_reduction_percent": oracle["actual_peak_reduction_percent"],
                           "lstm_minus_naive_pp": lstm["actual_peak_reduction_percent"] - naive["actual_peak_reduction_percent"],
                           "lstm_minus_naive_kw": lstm["actual_peak_reduction_kw"] - naive["actual_peak_reduction_kw"],
                           "oracle_minus_lstm_pp": oracle["actual_peak_reduction_percent"] - lstm["actual_peak_reduction_percent"],
                           "oracle_minus_naive_pp": oracle["actual_peak_reduction_percent"] - naive["actual_peak_reduction_percent"]})
    return output


def run_sensitivity(design_dir, baseline_dir, output_dir, settings=SolverSettings(), command=None, progress=None,
                    *, config_dir=None):
    """480 new two-stage solves plus 60 revalidated baseline references."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    days, profiles, base_battery, upstream, input_hashes = load_design(design_dir)
    config_dir = Path(config_dir) if config_dir is not None else Path(design_dir)
    batteries, configs = load_configs(config_dir)
    configuration_hashes = {str((config_dir / n).resolve()): digest(config_dir / n)
                            for n in ("design.json", "battery_configs.csv")}
    if batteries["baseline"] != base_battery:
        raise ValueError("Baseline specification differs")
    _, _, baseline, baseline_hashes = load_baseline(baseline_dir, base_battery, input_hashes, settings)
    old_rows = read_csv(Path(baseline_dir) / "daily_metrics.csv")
    old_metrics = {(r["date"], r["forecast_model"]): r for r in old_rows}
    old_groups = {}
    old_schedules = read_csv(Path(baseline_dir) / "schedules.csv")
    for row in old_schedules:
        old_groups.setdefault((row["date"], row["forecast_model"]), []).append(row)
    expected = {(d["date"], m) for d in days for m in FORECASTS}
    if (len(old_rows) != 60 or set(old_metrics) != expected or set(old_groups) != expected
            or len(old_schedules) != 2880 or any(len(g) != 48 for g in old_groups.values())):
        raise ValueError("Incomplete or duplicated baseline reference")
    for g in old_groups.values():
        g.sort(key=lambda r: r["timestamp"])
    code_paths = [Path(__file__), REPO / "scripts/run_eneres_sensitivity.py", REPO / "scripts/run_eneres_bess.py",
                  REPO / "src/experiments/eneres_bess.py", REPO / "src/experiments/eneres_shift.py",
                  REPO / "src/experiments/eneres_oracle.py", OPTIMIZER_PATH,
                  REPO / "src/forecast/demand_data.py", REPO / "requirements-eneres-bess.txt"]
    code = _code_record(REPO, code_paths)
    solver = settings.make_solver()
    binary = Path(solver.path).resolve()
    solver_hash = digest(binary)
    version = subprocess.run([str(binary), "-stop"], check=True, capture_output=True, text=True).stdout
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        (staging / "solver_logs").mkdir()
        metrics, schedules, validations, references = [], [], [], []
        solved = 0
        for config_id in CONFIG_IDS:
            battery = batteries[config_id]
            inherited = config_id == "baseline"
            for day in days:
                group = profiles[day["date"]]
                actual = np.asarray([float(r["actual_demand_kw"]) for r in group])
                for model, column in FORECASTS.items():
                    forecast = np.asarray([float(r[column]) for r in group])
                    case_id = f"{day['date']}_{config_id}_{model}"
                    old = old_metrics[day["date"], model]
                    if inherited:
                        result = baseline_result(old, old_groups[day["date"], model], forecast, battery)
                        for suffix in (".log", ".tiebreak.log"):
                            source = Path(baseline_dir) / "solver_logs" / f"{old['case_id']}{suffix}"
                            if "Result - Optimal solution found" not in source.read_text():
                                raise ValueError("Missing baseline optimality log")
                            shutil.copyfile(source, staging / "solver_logs" / f"{case_id}{suffix}")
                    else:
                        result = solve_forecast(forecast, battery, settings, staging / "solver_logs" / f"{case_id}.log")
                        solved += 1
                    values, predicted_net, actual_net = evaluate_schedule(actual, forecast, result, battery)
                    meta = {"case_id": case_id, "date": day["date"], "season": day["season"],
                            "season_rank": int(day["season_rank"]), "config_id": config_id, "forecast_model": model,
                            "shift_slots": 0, "uses_target_day_actual_in_planning": model == "perfect_foresight_oracle",
                            "execution": "reused_baseline_v2" if inherited else "new_two_stage_solve"}
                    record = {**meta, **values, "source_case_id": old["case_id"] if inherited else case_id,
                              "source_solver_elapsed_wall_s": float(old["solver_elapsed_wall_s"]) if inherited else 0.}
                    metrics.append(record)
                    validations.append({**meta, **result["independent_validation"]})
                    saved = [{**meta, "slot": i, "timestamp": row["timestamp"],
                              "forecast_origin_timestamp": row["origin_timestamp"],
                              "actual_demand_kw": actual[i], "forecast_demand_kw": forecast[i],
                              "charge_kw": result["c_t"][i], "discharge_kw": result["d_t"][i],
                              "is_charging": result["is_charging_t"][i], "soc_start_kwh": result["soc_t"][i],
                              "soc_end_kwh": result["soc_t"][i + 1], "controlled_forecast_kw": predicted_net[i],
                              "controlled_actual_kw": actual_net[i]} for i, row in enumerate(group)]
                    schedules.extend(saved)
                    if inherited:
                        references.append({"forecast_model": model, **compare_zero(day["date"], record, saved,
                                          {day["date"]: old}, {day["date"]: old_groups[day["date"], model]})})
                    if progress is not None:
                        progress(record, len(metrics), 540)
        if (len(metrics), len(schedules), solved, len(references)) != (540, 25920, 480, 60):
            raise ValueError("Incomplete BESS sensitivity comparison")
        summary = summarize_sensitivity(metrics)
        for name, values in (("daily_metrics.csv", metrics), ("schedules.csv", schedules),
                             ("summary.csv", summary), ("battery_configs.csv", configs),
                             ("model_comparisons.csv", model_comparisons(metrics))):
            write_csv(staging / name, values)
        def write_json(name, value):
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_json("validation.json", {"all_valid": True, "cases": validations, "baseline_reference_checks": references,
                                       "tolerances": "Case-specific; recorded in cases[].tolerances; mode=1e-6"})
        (staging / "environment.txt").write_text("\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()})) + "\n")
        if (_code_record(REPO, code_paths) != code or digest(binary) != solver_hash
                or any(digest(p) != h for p, h in {**input_hashes, **baseline_hashes, **configuration_hashes}.items())):
            raise ValueError("Code, input, baseline or solver changed during comparison")
        manifest = {"schema_version": 1, "status": "bess_sensitivity_completed",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "command": list(command) if command is not None else None, "code": code,
                    "input_hashes": input_hashes, "baseline_hashes": baseline_hashes,
                    "configuration_hashes": configuration_hashes,
                    "upstream_design_manifest": upstream, "battery_configs": configs,
                    "batteries": {n: {**asdict(b), "final_soc_equality": True, "soc_min_fraction": 0., "soc_max_fraction": 1.}
                                  for n, b in batteries.items()},
                    "solver": {**baseline["solver"], "binary": str(binary), "binary_sha256": solver_hash,
                               "version_banner": version, "settings": asdict(settings), "configuration": solver.toDict()},
                    "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
                    "protocol": {**baseline["protocol"], "cases": 540, "newly_solved_cases": 480,
                                 "reused_baseline_cases": 60, "config_ids": list(CONFIG_IDS),
                                 "configuration_design": "Frozen step-4 one-factor scenarios; same 20 days and three unchanged forecasts, shift 0 only; no oracle corrections or model retraining",
                                 "sizing_interpretation": "Study scenarios, not ENERES equipment ratings or economically optimal sizing; no uncertainty intervals or annual operation claims",
                                 "references": "Baseline actions and 120 CBC logs reused, revalidated and reevaluated; remaining eight configurations newly solved"},
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
