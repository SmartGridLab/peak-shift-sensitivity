"""Verified baseline MILP cases: forecasts plan actions; actual demand evaluates them."""

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import importlib.util
from importlib.metadata import distributions
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import pulp

from forecast.demand_data import JST, _code_record

REPO = Path(__file__).resolve().parents[2]
OPTIMIZER_PATH = REPO / "src/battery_control/scheduler/lp_peak_shaving_optimizer.py"
FORECASTS = {
    "lstm": "lstm_forecast_kw",
    "seasonal_naive": "seasonal_naive_kw",
    "perfect_foresight_oracle": "actual_demand_kw",
}
SEASONS = ("spring", "summer", "autumn", "winter")
# Legacy ENERES comparison thresholds; physical acceptance is case-specific below.
POWER_TOLERANCE_KW = 0.05
ENERGY_TOLERANCE_KWH = 0.05
TIEBREAK_PEAK_TOLERANCE_KW = 0.01
TIEBREAK_ACTION_TOLERANCE_KW = 1.0
CHARGE_TIEBREAK = ("stage 2 bounds forecast peak and total action using battery-scaled numerical slacks; "
                   "minimises 1000 * stage-1 objective + sum(forecast_kw / max(forecast_kw) * charge_kw); "
                   "forecast only; tolerances recorded for each case")


def numerical_tolerances(forecast, battery):
    """Budget finite-digit CBC output separately from MILP stage-2 slacks.

    1e-7 relative error covers CBC's ~8 significant digit solution output.
    Power/energy constraints depend on BESS size, never regional demand size.
    The legacy 0.05 floor shrinks below 50 kW/kWh; 1e-6 is the absolute floor.
    Stage-2 slacks retain ENERES values up to 50 MW / 200 MWh and scale by
    powers of ten only when BOTH limits exceed that range. This preserves
    the saved regional-scale protocol (10 kW / 1000 kW), not a unit conversion.
    """
    def physical(size):
        return max(1e-6, min(0.05, 1e-3 * size), 1e-7 * size)
    power = physical(battery.max_power_kw)
    energy = physical(battery.battery_capacity_kwh)
    demand = max(power, 1e-7 * float(np.max(np.abs(forecast))))
    ratio = min(battery.max_power_kw / 50000., battery.battery_capacity_kwh / 200000.)
    stage_scale = 10. ** max(0, math.ceil(math.log10(ratio)))
    peak_slack = min(0.01, 1e-3 * battery.max_power_kw) * stage_scale
    action_slack = min(1., 0.1 * battery.max_power_kw) * stage_scale
    return {"power_kw": power, "energy_kwh": energy, "demand_kw": demand,
            "peak_slack_kw": peak_slack, "action_slack_kw": action_slack,
            "objective_kw": demand + battery.action_penalty_lambda * len(forecast) * power,
            "position_objective_kw": len(forecast) * power}


def digest(path):
    """Hash an input or output without changing it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    """Read CSV rows with explicit UTF-8 encoding."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    """Write uniform records; refuse an empty result table."""
    if not rows:
        raise ValueError("Cannot save an empty result table")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class BatterySpec:
    """Grid-side kW and modeled usable kWh; no hidden MW conversion."""

    battery_capacity_kwh: float
    max_power_kw: float
    initial_soc_kwh: float
    charge_efficiency: float
    discharge_efficiency: float
    action_penalty_lambda: float = 1e-4
    timestep_h: float = 0.5

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Battery conditions must be finite")
        if self.battery_capacity_kwh <= 0 or self.max_power_kw <= 0:
            raise ValueError("Battery capacity and power must be positive")
        if not 0 <= self.initial_soc_kwh <= self.battery_capacity_kwh:
            raise ValueError("Initial SoC outside capacity")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("Efficiencies must lie in (0, 1]")
        if self.timestep_h != 0.5 or self.action_penalty_lambda != 1e-4:
            raise ValueError("This step requires 30-minute intervals and lambda=1e-4")

    @classmethod
    def from_design(cls, config):
        """Verify that the design maps directly to the legacy model's constraints."""
        if config["config_id"] != "baseline" or config["final_soc_equality"] is not True:
            raise ValueError("Only the baseline with equal terminal SoC is implemented")
        if config["soc_min_fraction"] != 0 or config["soc_max_fraction"] != 1:
            raise ValueError("This optimizer models usable SoC from zero to full")
        spec = cls(**{key: float(config[key]) for key in cls.__dataclass_fields__})
        expected = {
            "round_trip_efficiency": spec.charge_efficiency * spec.discharge_efficiency,
            "initial_soc_fraction": spec.initial_soc_kwh / spec.battery_capacity_kwh,
            "duration_hours": spec.battery_capacity_kwh / spec.max_power_kw,
            "power_to_reference_peak_ratio": spec.max_power_kw / float(config["reference_peak_kw"]),
        }
        if any(not math.isclose(float(config[k]), v, rel_tol=1e-12, abs_tol=1e-12) for k, v in expected.items()):
            raise ValueError("Inconsistent derived battery conditions")
        return spec


@dataclass(frozen=True)
class SolverSettings:
    """Explicit CBC parameters; an incumbent at the time limit is rejected."""

    time_limit_s: float = 60.0
    threads: int = 1
    seed: int = 42

    def __post_init__(self):
        if not math.isfinite(self.time_limit_s) or self.time_limit_s <= 0:
            raise ValueError("Positive solver time limit required")
        if self.threads != 1 or not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed <= 0:
            raise ValueError("One thread and a positive integer seed are required")

    def make_solver(self, log_path=None):
        """Use bundled CBC with no solver fallback or warm-start information."""
        return pulp.PULP_CBC_CMD(
            msg=False, mip=True, timeLimit=self.time_limit_s, gapRel=0, gapAbs=0,
            threads=self.threads, presolve=True, cuts=True, warmStart=False,
            options=[f"randomSeed {self.seed}", f"randomCbcSeed {self.seed}",
                     "primalTolerance 1e-7", "integerTolerance 1e-8"],
            logPath=str(log_path) if log_path is not None else None,
        )


@lru_cache(maxsize=1)
def optimizer_class():
    """Load the existing optimizer without importing unrelated RL/API packages."""
    spec = importlib.util.spec_from_file_location("eneres_peak_optimizer", OPTIMIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LPPeakShavingOptimizer


def finite_profile(values, length=None):
    """Demand profiles must be finite, nonnegative and one dimensional."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or not arr.size or not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError("Expected a finite nonnegative demand profile")
    if length is not None and len(arr) != length:
        raise ValueError("Demand profile length mismatch")
    return arr


def tiebreak_log_path(log_path):
    """Keep the stage-2 CBC log next to the stage-1 log of the same case."""
    return None if log_path is None else Path(log_path).with_suffix(".tiebreak.log")


def validate_solution(forecast, result, battery, *, require_stage2_bounds=True):
    """Recompute all physical constraints and objective from raw solver outputs."""
    n = len(forecast)
    tolerances = numerical_tolerances(forecast, battery)
    arrays = {}
    for key, length in (("c_t", n), ("d_t", n), ("soc_t", n + 1), ("demand_t", n), ("is_charging_t", n)):
        arr = np.asarray(result[key], dtype=np.float64)
        if arr.shape != (length,) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid solver values: {key}")
        arrays[key] = arr
    c, d, soc, net, mode = (arrays[k] for k in ("c_t", "d_t", "soc_t", "demand_t", "is_charging_t"))
    reconstructed = np.r_[battery.initial_soc_kwh, battery.initial_soc_kwh + np.cumsum(
        (battery.charge_efficiency * c - d / battery.discharge_efficiency) * battery.timestep_h
    )]
    controlled = np.asarray(forecast) + c - d
    objective = float(np.max(controlled) + battery.action_penalty_lambda * np.sum(c + d))
    peak = float(result["demand_peak_forecast"])
    if not math.isfinite(peak) or not math.isfinite(float(result["objective_value"])):
        raise ValueError("Nonfinite solver objective")
    tiebreak = result["charge_tiebreak"]
    primary_objective = float(tiebreak["primary"]["objective_value"])
    bounds_recorded = all(k in tiebreak["primary"] for k in ("peak_kw", "action_sum_kw"))
    if require_stage2_bounds and not bounds_recorded:
        raise ValueError("Missing primary peak/action records")
    primary_peak = float(tiebreak["primary"].get("peak_kw", 0.))
    primary_action = float(tiebreak["primary"].get("action_sum_kw", 0.))
    if not all(math.isfinite(v) for v in (primary_peak, primary_action, float(tiebreak["objective_value"]))):
        raise ValueError("Nonfinite charge tie-break values")
    weights = np.asarray(tiebreak["weights"], dtype=np.float64)
    expected_weights = np.asarray(forecast, dtype=np.float64) / (float(np.max(np.abs(forecast))) or 1.0)
    if not math.isfinite(primary_objective) or weights.shape != (n,) or not np.isfinite(weights).all():
        raise ValueError("Invalid charge tie-break record")
    power_violation = max(0., float(-c.min()), float(-d.min()), float(c.max() - battery.max_power_kw), float(d.max() - battery.max_power_kw))
    energy_violation = max(0., float(-soc.min()), float(soc.max() - battery.battery_capacity_kwh),
                           float(-reconstructed.min()), float(reconstructed.max() - battery.battery_capacity_kwh))
    details = {
        "max_power_bound_violation_kw": power_violation,
        "max_soc_bound_violation_kwh": energy_violation,
        "max_soc_update_error_kwh": float(np.max(np.abs(np.diff(soc) - (battery.charge_efficiency * c - d / battery.discharge_efficiency) * battery.timestep_h))),
        "max_reconstructed_soc_error_kwh": float(np.max(np.abs(soc - reconstructed))),
        "initial_soc_error_kwh": float(abs(soc[0] - battery.initial_soc_kwh)),
        "terminal_soc_error_kwh": float(abs(soc[-1] - battery.initial_soc_kwh)),
        "reconstructed_terminal_soc_error_kwh": float(abs(reconstructed[-1] - battery.initial_soc_kwh)),
        "max_simultaneous_power_kw": float(np.max(np.minimum(c, d))),
        "max_mode_integrality_error": float(np.max(np.abs(mode - np.rint(mode)))),
        "max_mode_bound_violation": max(0., float(-mode.min()), float(mode.max() - 1)),
        "max_mode_power_violation_kw": max(0., float(np.max(c - battery.max_power_kw * mode)), float(np.max(d - battery.max_power_kw * (1 - mode)))),
        "max_demand_definition_error_kw": float(np.max(np.abs(net - controlled))),
        "peak_definition_error_kw": float(abs(peak - np.max(controlled))),
        "objective_error_kw": float(abs(float(result["objective_value"]) - objective)),
        "primary_objective_increase_kw": max(0., objective - primary_objective),
        "primary_record_error_kw": abs(primary_objective - primary_peak - battery.action_penalty_lambda * primary_action),
        "stage2_peak_increase_kw": max(0., float(np.max(controlled)) - primary_peak),
        "stage2_action_increase_kw": max(0., float(np.sum(c + d)) - primary_action),
        "max_tiebreak_weight_error": float(np.max(np.abs(weights - expected_weights))),
        "tiebreak_objective_error_kw": float(abs(float(tiebreak["objective_value"]) - float(np.dot(expected_weights, c)))),
    }
    if not bounds_recorded:
        for key in ("primary_record_error_kw", "stage2_peak_increase_kw", "stage2_action_increase_kw"):
            del details[key]
    strict = ("max_mode_integrality_error", "max_mode_bound_violation", "max_tiebreak_weight_error")
    for key, value in details.items():
        if key in strict:
            tolerance = 1e-6
        elif key.endswith("kwh"):
            tolerance = tolerances["energy_kwh"]
        elif key == "stage2_peak_increase_kw":
            tolerance = tolerances["peak_slack_kw"] + tolerances["demand_kw"]
        elif key == "stage2_action_increase_kw":
            tolerance = tolerances["action_slack_kw"] + n * tolerances["power_kw"]
        elif key == "primary_objective_increase_kw":
            tolerance = (tolerances["peak_slack_kw"] + battery.action_penalty_lambda * tolerances["action_slack_kw"]
                         + tolerances["objective_kw"])
        elif key in ("objective_error_kw", "primary_record_error_kw"):
            tolerance = tolerances["objective_kw"]
        elif key == "tiebreak_objective_error_kw":
            tolerance = tolerances["position_objective_kw"]
        elif key in ("max_demand_definition_error_kw", "peak_definition_error_kw"):
            tolerance = tolerances["demand_kw"]
        else:
            tolerance = tolerances["power_kw"]
        if value > tolerance:
            raise ValueError(f"Physical validation failed: {key}={value} > {tolerance}")
    return {"all_valid": True, "stage2_bounds_checked": bounds_recorded, "tolerances": tolerances, **details}


def solve_forecast(forecast, battery, settings=SolverSettings(), log_path=None):
    """Only the forecast enters planning; actual demand is not an argument.

    Stage 1 minimises the forecast peak plus action penalty. Many charge placements
    reach the same value, so stage 2 deterministically charges in the lowest-forecast
    intervals without relaxing the stage-1 optimum beyond the documented slack.
    """
    forecast = finite_profile(forecast)
    solver = settings.make_solver(log_path)
    tiebreak_solver = settings.make_solver(tiebreak_log_path(log_path))
    if not solver.available() or not tiebreak_solver.available():
        raise RuntimeError("Bundled CBC is unavailable; no solver fallback is allowed")
    optimizer = optimizer_class()(**asdict(battery), final_soc_equality=True)
    started = perf_counter()
    tolerances = numerical_tolerances(forecast, battery)
    result = optimizer.optimize_peak_shaving(
        forecast, solver=solver, charge_tiebreak_solver=tiebreak_solver,
        tiebreak_peak_tolerance_kw=tolerances["peak_slack_kw"],
        tiebreak_action_tolerance_kw=tolerances["action_slack_kw"],
    )
    elapsed = perf_counter() - started
    optimal = pulp.LpSolution[pulp.LpSolutionOptimal]
    stages = [(result.get("charge_tiebreak", {}).get("primary", {}), log_path),
              (result, tiebreak_log_path(log_path))]
    for stage, path in stages:
        if stage.get("problem_status") != "Optimal" or stage.get("solution_status") != optimal:
            raise RuntimeError("A proven optimal integer solution is required in both stages")
        if path is not None and "Result - Optimal solution found" not in Path(path).read_text():
            raise RuntimeError("CBC log does not confirm an optimal solution")
    result["independent_validation"] = validate_solution(forecast, result, battery)
    result["elapsed_wall_s"] = elapsed
    return result


def evaluate_schedule(actual, forecast, result, battery):
    """Apply the same grid-side actions to actual demand, without corrective dispatch."""
    forecast = finite_profile(forecast)
    actual = finite_profile(actual, len(forecast))
    controlled_forecast = forecast + result["c_t"] - result["d_t"]
    controlled_actual = actual + result["c_t"] - result["d_t"]
    before, after = float(actual.max()), float(controlled_actual.max())
    predicted_before, predicted_after = float(forecast.max()), float(controlled_forecast.max())
    charge = float(np.sum(result["c_t"]) * battery.timestep_h)
    discharge = float(np.sum(result["d_t"]) * battery.timestep_h)
    metrics = {
        "actual_peak_before_kw": before, "actual_peak_after_kw": after,
        "actual_peak_reduction_kw": before - after,
        "actual_peak_reduction_percent": (before - after) / before * 100 if before > 0 else None,
        "forecast_peak_before_kw": predicted_before, "forecast_peak_after_kw": predicted_after,
        "forecast_peak_reduction_kw": predicted_before - predicted_after,
        "actual_peak_after_slot": int(np.argmax(controlled_actual)),
        "actual_net_demand_min_kw": float(controlled_actual.min()),
        "actual_export_slots": int(np.count_nonzero(controlled_actual < -POWER_TOLERANCE_KW)),
        "charge_energy_kwh": charge, "discharge_energy_kwh": discharge,
        "grid_energy_loss_kwh": charge - discharge,
        "soc_start_kwh": float(result["soc_t"][0]), "soc_end_kwh": float(result["soc_t"][-1]),
        "solver_objective_kw": float(result["objective_value"]),
        "primary_solver_objective_kw": float(result["charge_tiebreak"]["primary"]["objective_value"]),
        "primary_peak_kw": float(result["charge_tiebreak"]["primary"]["peak_kw"]),
        "primary_action_sum_kw": float(result["charge_tiebreak"]["primary"]["action_sum_kw"]),
        "charge_tiebreak_objective": float(result["charge_tiebreak"]["objective_value"]),
        "primary_solver_status": result["charge_tiebreak"]["primary"]["problem_status"],
        "primary_solver_solution_status": result["charge_tiebreak"]["primary"]["solution_status"],
        "solver_status": result["problem_status"], "solver_solution_status": result["solution_status"],
        "constraints_valid": result["independent_validation"]["all_valid"],
        "solver_elapsed_wall_s": result["elapsed_wall_s"],
    }
    return metrics, controlled_forecast, controlled_actual


def load_design(directory):
    """Check frozen upstream artifacts and exact day/slot/condition alignment."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "design_only_not_simulated":
        raise ValueError("Expected the step-4 design manifest")
    hashes = {str(manifest_path.resolve()): digest(manifest_path)}
    required = {"design.json", "selected_days.csv", "selected_profiles.csv", "battery_configs.csv"}
    if not required <= manifest["outputs"].keys():
        raise ValueError("Incomplete design manifest")
    for name, info in manifest["outputs"].items():
        if Path(name).name != name:
            raise ValueError("Unexpected design filename")
        path = directory / name
        if digest(path) != info["sha256"]:
            raise ValueError(f"Design hash mismatch: {name}")
        hashes[str(path.resolve())] = info["sha256"]
    design = json.loads((directory / "design.json").read_text())
    configs = [r for r in design["battery_configs"] if r["config_id"] == "baseline"]
    if len(configs) != 1:
        raise ValueError("Exactly one baseline is required")
    battery = BatterySpec.from_design(configs[0])
    config_rows = [r for r in read_csv(directory / "battery_configs.csv") if r["config_id"] == "baseline"]
    if len(config_rows) != 1:
        raise ValueError("Battery CSV baseline mismatch")
    for key, value in configs[0].items():
        saved = config_rows[0][key]
        if isinstance(value, bool):
            matches = saved == str(value)
        elif isinstance(value, (int, float)):
            matches = float(saved) == value
        else:
            matches = saved == value
        if not matches:
            raise ValueError(f"Battery CSV/JSON mismatch: {key}")
    days = read_csv(directory / "selected_days.csv")
    dates = {r["date"] for r in days}
    if len(days) != 20 or len(dates) != 20 or any(sum(r["season"] == s for r in days) != 5 for s in SEASONS):
        raise ValueError("Expected five distinct days per season")
    groups = {day: [] for day in dates}
    profiles = read_csv(directory / "selected_profiles.csv")
    if len(profiles) != 960:
        raise ValueError("Expected 960 profile rows")
    for row in profiles:
        if row["date"] not in groups:
            raise ValueError("Unexpected profile day")
        groups[row["date"]].append(row)
    for day in days:
        group = groups[day["date"]]
        group.sort(key=lambda r: r["timestamp"])
        origin = datetime.fromisoformat(day["date"]).replace(tzinfo=JST)
        if len(group) != 48:
            raise ValueError("Each case needs 48 intervals")
        for i, row in enumerate(group):
            if datetime.fromisoformat(row["timestamp"]) != origin + timedelta(minutes=30 * i) or datetime.fromisoformat(row["origin_timestamp"]) != origin:
                raise ValueError("Invalid profile timestamp or forecast origin")
            if row["season"] != day["season"] or row["season_rank"] != day["season_rank"]:
                raise ValueError("Profile selection metadata mismatch")
        for key in FORECASTS.values():
            finite_profile([float(r[key]) for r in group], 48)
        if max(float(r["actual_demand_kw"]) for r in group) != float(day["actual_peak_kw"]):
            raise ValueError("Actual peak differs from selected day")
    return days, groups, battery, manifest, hashes


def summarize(rows):
    """Report equal-day summaries, including worsening days rather than hiding them."""
    summaries = []
    for season in ("all", *SEASONS):
        for model in FORECASTS:
            part = [r for r in rows if r["forecast_model"] == model and (season == "all" or r["season"] == season)]
            reductions = np.asarray([r["actual_peak_reduction_kw"] for r in part])
            percentages = [r["actual_peak_reduction_percent"] for r in part]
            summaries.append({
                "season": season, "forecast_model": model, "days": len(part),
                "mean_actual_peak_reduction_kw": float(np.mean(reductions)),
                "mean_actual_peak_reduction_percent": float(np.mean(percentages)),
                "median_actual_peak_reduction_percent": float(np.median(percentages)),
                "min_actual_peak_reduction_percent": float(np.min(percentages)),
                "max_actual_peak_reduction_percent": float(np.max(percentages)),
                "worsened_days": int(np.count_nonzero(reductions < -POWER_TOLERANCE_KW)),
                "total_charge_energy_kwh": float(sum(r["charge_energy_kwh"] for r in part)),
                "total_discharge_energy_kwh": float(sum(r["discharge_energy_kwh"] for r in part)),
            })
    return summaries


def run_baseline(design_dir, output_dir, settings=SolverSettings(), command=None, progress=None):
    """Run exactly 20 x 3 unshifted cases and publish only a complete verified run."""
    design_dir, output_dir = Path(design_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    days, groups, battery, upstream, input_hashes = load_design(design_dir)
    code_paths = [Path(__file__), OPTIMIZER_PATH, REPO / "scripts/run_eneres_bess.py", REPO / "src/forecast/demand_data.py", REPO / "requirements-eneres-bess.txt"]
    code = _code_record(REPO, code_paths)
    solver = settings.make_solver()
    if not solver.available():
        raise RuntimeError("CBC unavailable")
    solver_binary = Path(solver.path).resolve()
    solver_hash = digest(solver_binary)
    version = subprocess.run([str(solver_binary), "-stop"], check=True, capture_output=True, text=True).stdout
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        (staging / "solver_logs").mkdir()
        metrics, schedules, validations = [], [], []
        for day in days:
            group = groups[day["date"]]
            actual = np.asarray([float(r["actual_demand_kw"]) for r in group])
            for model, column in FORECASTS.items():
                forecast = np.asarray([float(r[column]) for r in group])
                case_id = f"{day['date']}_{model}"
                result = solve_forecast(forecast, battery, settings, staging / "solver_logs" / f"{case_id}.log")
                case_metrics, predicted_net, actual_net = evaluate_schedule(actual, forecast, result, battery)
                meta = {"case_id": case_id, "date": day["date"], "season": day["season"],
                        "season_rank": int(day["season_rank"]), "config_id": "baseline", "forecast_model": model, "shift_slots": 0}
                metrics.append({**meta, **case_metrics})
                validations.append({**meta, **result["independent_validation"]})
                for i, row in enumerate(group):
                    schedules.append({**meta, "timestamp": row["timestamp"], "forecast_origin_timestamp": row["origin_timestamp"],
                                      "actual_demand_kw": actual[i], "forecast_demand_kw": forecast[i],
                                      "charge_kw": result["c_t"][i], "discharge_kw": result["d_t"][i],
                                      "is_charging": result["is_charging_t"][i],
                                      "soc_start_kwh": result["soc_t"][i], "soc_end_kwh": result["soc_t"][i + 1],
                                      "controlled_forecast_kw": predicted_net[i], "controlled_actual_kw": actual_net[i]})
                if progress is not None:
                    progress(metrics[-1], len(metrics), 60)
        if len(metrics) != 60 or len(schedules) != 2880:
            raise ValueError("Incomplete simulation")
        summaries = summarize(metrics)
        write_csv(staging / "daily_metrics.csv", metrics)
        write_csv(staging / "schedules.csv", schedules)
        write_csv(staging / "summary.csv", summaries)
        def write_json(name, obj):
            (staging / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_json("validation.json", {"all_valid": True, "cases": validations,
                                       "tolerances": "Case-specific; recorded in cases[].tolerances; mode=1e-6"})
        (staging / "environment.txt").write_text("\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()})) + "\n")
        if _code_record(REPO, code_paths) != code or any(digest(p) != h for p, h in input_hashes.items()) or digest(solver_binary) != solver_hash:
            raise ValueError("Code, inputs or solver changed during simulation")
        manifest = {
            "schema_version": 1, "status": "baseline_simulated", "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": list(command) if command is not None else None, "code": code, "input_hashes": input_hashes,
            "upstream_design_manifest": upstream, "battery": {**asdict(battery), "final_soc_equality": True, "soc_min_fraction": 0, "soc_max_fraction": 1},
            "solver": {"name": "PULP_CBC_CMD", "pulp_version": pulp.__version__, "binary": str(solver_binary), "binary_sha256": solver_hash,
                       "version_banner": version, "settings": asdict(settings), "configuration": solver.toDict(),
                       "acceptance": "Both stages: Optimal status, Optimal Solution Found solution status and matching CBC log; no time-limit incumbent accepted"},
            "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
            "protocol": {"cases": 60, "days": 20, "forecast_models": list(FORECASTS), "config_ids": ["baseline"], "shift_slots": [0],
                         "planning": "Only specified forecast enters MILP; actual demand enters planning only for perfect_foresight_oracle",
                         "evaluation": "actual_demand_kw + charge_kw - discharge_kw; fixed actions, no actual-demand feedback or correction",
                         "objective": "peak_forecast_kw + lambda * sum(charge_kw + discharge_kw); no dt multiplier on penalty",
                         "charge_tiebreak": CHARGE_TIEBREAK,
                         "tiebreak_tolerances": {"policy": "battery-scaled; see validation.json cases[].tolerances"},
                         "daily_soc": "Each day independently starts and ends at the same specified SoC; not consecutive operation",
                         "grid_constraint": "Net demand can be negative, as in the existing model; export occurrences recorded",
                         "oracle": "Perfect demand knowledge for a reference feasible schedule; not an operational forecast and not a pure peak-only optimum due to action penalty",
                         "statistics": "Equal-weight mean of daily reductions; selected high-demand cases, not independent random annual samples",
                         "forecast_availability": "Upstream 00:00 JST forecast assumes previous 23:30 interval available without reporting delay"},
            "outputs": {str(p.relative_to(staging)): {"sha256": digest(p), "size_bytes": p.stat().st_size} for p in sorted(staging.rglob("*")) if p.is_file()},
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
    return manifest, summaries
