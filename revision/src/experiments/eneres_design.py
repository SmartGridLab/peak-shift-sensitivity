"""Freeze actual-demand-only evaluation days and an OFAT BESS design.

This module prepares experiment inputs. It never optimizes a battery schedule.
Forecast errors and controlled demand do not enter the day-selection function.
"""

import csv
from dataclasses import asdict
from datetime import date, datetime, timedelta
import json
import math

import numpy as np

from forecast.demand_data import JST
from forecast.eneres_day_ahead import error_metrics, sha256


SEASONS = {
    "spring": (3, 4, 5), "summer": (6, 7, 8),
    "autumn": (9, 10, 11), "winter": (12, 1, 2),
}
DAYS_PER_SEASON = 5
BASE_POWER_RATIO = 0.20
BASE_DURATION_HOURS = 4.0
BASE_SOC_FRACTION = 0.50
BASE_ROUND_TRIP_EFFICIENCY = 0.85


def season_of(day: date) -> str:
    return next(name for name, months in SEASONS.items() if day.month in months)


def consecutive_groups(days):
    groups = []
    for day in sorted(set(days)):
        if groups and day == groups[-1][-1] + timedelta(days=1):
            groups[-1].append(day)
        else:
            groups.append([day])
    return groups


def _longest_true_run(mask):
    longest = current = 0
    for present in mask:
        current = current + 1 if present else 0
        longest = max(longest, current)
    return longest


def select_days(dataset, periods):
    """Select five largest daily ACTUAL peaks per season; earliest date breaks ties."""
    days = {}
    for record in dataset.records:
        day = record.timestamp.date()
        if periods.evaluation_start <= day < periods.end_exclusive:
            days.setdefault(day, []).append(record)
    if len(days) != (periods.end_exclusive - periods.evaluation_start).days:
        raise ValueError("Incomplete evaluation-day coverage")
    candidates = []
    for day, records in sorted(days.items()):
        if len(records) != 48:
            raise ValueError(f"Expected 48 intervals on {day}")
        for slot, record in enumerate(records):
            expected = datetime.combine(day, datetime.min.time(), tzinfo=JST) + timedelta(minutes=30 * slot)
            if record.timestamp != expected:
                raise ValueError(f"Incorrect interval order on {day}")
        values = np.asarray([r.demand_kw for r in records], dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid actual demand on {day}")
        peak = float(values.max())
        if peak <= 0:
            raise ValueError(f"Daily peak must be positive on {day}")
        high = values >= 0.90 * peak
        candidates.append({
            "date": day.isoformat(), "season": season_of(day),
            "actual_peak_kw": peak, "actual_peak_timestamp": records[int(np.argmax(values))].timestamp.isoformat(),
            "peak_tied_slots": int(np.count_nonzero(values == peak)),
            "actual_min_kw": float(values.min()), "actual_mean_kw": float(values.mean()),
            "actual_energy_kwh": float(values.sum() * 0.5), "load_factor": float(values.mean() / peak),
            "hours_at_or_above_90pct_peak": float(high.sum() * 0.5),
            "longest_contiguous_hours_at_or_above_90pct_peak": float(_longest_true_run(high) * 0.5),
        })
    selected, counts = [], {}
    for season in SEASONS:
        ranked = sorted((r for r in candidates if r["season"] == season),
                        key=lambda r: (-r["actual_peak_kw"], r["date"]))
        counts[season] = len(ranked)
        if len(ranked) < DAYS_PER_SEASON:
            raise ValueError(f"Fewer than five candidate days in {season}")
        for rank, row in enumerate(ranked, start=1):
            row["season_rank"] = rank
            row["selected"] = rank <= DAYS_PER_SEASON
        selected.extend(dict(row) for row in ranked[:DAYS_PER_SEASON])
    groups = consecutive_groups(date.fromisoformat(row["date"]) for row in selected)
    summary = {
        "candidate_days_by_season": counts,
        "selected_days_by_season": {season: DAYS_PER_SEASON for season in SEASONS},
        "adjacent_selected_pairs": sum(len(group) - 1 for group in groups),
        "longest_consecutive_selected_run_days": max(map(len, groups)),
        "consecutive_selected_groups": [[day.isoformat() for day in group] for group in groups if len(group) > 1],
        "selected_month_counts": {
            month: sum(row["date"].startswith(month) for row in selected)
            for month in sorted({row["date"][:7] for row in selected})
        },
    }
    return candidates, selected, summary


def battery_configs(reference_peak_kw):
    if not math.isfinite(reference_peak_kw) or reference_peak_kw <= 0:
        raise ValueError("Reference peak must be finite and positive")
    base_power = reference_peak_kw * BASE_POWER_RATIO
    base_energy = base_power * BASE_DURATION_HOURS

    def make(name, axis, power=base_power, energy=base_energy,
             soc=BASE_SOC_FRACTION, efficiency=BASE_ROUND_TRIP_EFFICIENCY):
        eta = math.sqrt(efficiency)
        return {
            "config_id": name, "varied_axis": axis,
            "reference_peak_kw": reference_peak_kw, "power_to_reference_peak_ratio": power / reference_peak_kw,
            "max_power_kw": power, "battery_capacity_kwh": energy, "duration_hours": energy / power,
            "initial_soc_fraction": soc, "initial_soc_kwh": energy * soc,
            "round_trip_efficiency": efficiency, "charge_efficiency": eta, "discharge_efficiency": eta,
            "soc_min_fraction": 0.0, "soc_max_fraction": 1.0,
            "final_soc_equality": True, "action_penalty_lambda": 1e-4, "timestep_h": 0.5,
        }

    return [
        make("baseline", "baseline"),
        make("power_10pct", "power", power=reference_peak_kw * 0.10),
        make("power_30pct", "power", power=reference_peak_kw * 0.30),
        make("duration_2h", "capacity", energy=base_power * 2.0),
        make("duration_6h", "capacity", energy=base_power * 6.0),
        make("soc_25pct", "initial_soc", soc=0.25),
        make("soc_75pct", "initial_soc", soc=0.75),
        make("roundtrip_80pct", "efficiency", efficiency=0.80),
        make("roundtrip_90pct", "efficiency", efficiency=0.90),
    ]


def load_forecasts(directory, prepared_dir, dataset, periods):
    """Verify the upstream run and align all 365 days before taking selected rows."""
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unexpected forecast manifest schema")
    if manifest["prepared_manifest"]["sha256"] != sha256(prepared_dir / "manifest.json"):
        raise ValueError("Forecast and prepared manifest do not match")
    if manifest["periods"] != {key: value.isoformat() for key, value in asdict(periods).items()}:
        raise ValueError("Forecast periods do not match prepared periods")
    if manifest["source"]["sha256"] != dataset.source_sha256:
        raise ValueError("Forecast source does not match prepared source")
    if "evaluation_predictions.csv" not in manifest["outputs"]:
        raise ValueError("Forecast manifest omits evaluation predictions")
    for name, info in manifest["outputs"].items():
        if Path(name).name != name or sha256(directory / name) != info["sha256"]:
            raise ValueError(f"Forecast output hash mismatch: {name}")
    with (directory / "evaluation_predictions.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["origin_timestamp", "timestamp", "actual_demand_kw", "lstm_forecast_kw", "seasonal_naive_kw"]:
            raise ValueError("Unexpected forecast columns")
        rows = list(reader)
    truth = {record.timestamp: record.demand_kw for record in dataset.records}
    evaluation = [r for r in dataset.records if periods.evaluation_start <= r.timestamp.date() < periods.end_exclusive]
    if len(rows) != len(evaluation):
        raise ValueError("Forecast row count mismatch")
    by_day = {}
    for row, record in zip(rows, evaluation):
        timestamp = datetime.fromisoformat(row["timestamp"])
        origin = datetime.fromisoformat(row["origin_timestamp"])
        if timestamp != record.timestamp or origin != record.timestamp.replace(hour=0, minute=0):
            raise ValueError("Forecast interval or origin mismatch")
        for key in ("actual_demand_kw", "lstm_forecast_kw", "seasonal_naive_kw"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid forecast demand: {key}")
        if row["actual_demand_kw"] != record.demand_kw:
            raise ValueError("Forecast actual demand differs from prepared data")
        if row["seasonal_naive_kw"] != truth[timestamp - timedelta(days=7)]:
            raise ValueError("Seasonal naive differs from actual demand seven days earlier")
        by_day.setdefault(timestamp.date().isoformat(), []).append(row)
    return by_day, manifest


def selected_error_metrics(selected, forecasts):
    daily, summary = [], {}
    actual = np.asarray([[r["actual_demand_kw"] for r in forecasts[day["date"]]] for day in selected])
    for model, key in (("lstm", "lstm_forecast_kw"), ("seasonal_naive", "seasonal_naive_kw")):
        predicted = np.asarray([[r[key] for r in forecasts[day["date"]]] for day in selected])
        time_error = (np.argmax(predicted, axis=1) - np.argmax(actual, axis=1)) * 30
        magnitude_error = predicted.max(axis=1) - actual.max(axis=1)
        summary[model] = {
            **error_metrics(actual, predicted), "days": len(selected), "slots": int(actual.size),
            "mean_absolute_peak_time_error_minutes": float(np.mean(abs(time_error))),
            "mean_absolute_peak_magnitude_error_kw": float(np.mean(abs(magnitude_error))),
            "mean_signed_peak_magnitude_error_kw": float(np.mean(magnitude_error)),
        }
        for i, day in enumerate(selected):
            daily.append({"date": day["date"], "season": day["season"], "season_rank": day["season_rank"],
                          "model": model, **error_metrics(actual[i], predicted[i]),
                          "peak_time_error_minutes": int(time_error[i]),
                          "peak_magnitude_error_kw": float(magnitude_error[i])})
    return daily, summary
