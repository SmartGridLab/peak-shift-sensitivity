"""Create Figures 5–8 and the numerical cells of Tables 3, 5 and S1–S3."""

import argparse
import json
from pathlib import Path
from statistics import mean

from paper_figures import plot_baseline, plot_corrections, plot_shift, plot_waveform
import prepare_tepco  # Set the import path for the archived experiment modules.

import numpy as np
from experiments.eneres_bess import FORECASTS, POWER_TOLERANCE_KW, SEASONS, read_csv, write_csv

BASE_MODELS = tuple(FORECASTS)
CORRECTIONS = ("timing_oracle", "magnitude_oracle", "both_oracle")
RATE = "actual_peak_reduction_percent"
POWER = "actual_peak_reduction_kw"
SLOTS = tuple(range(-8, 9))


def load_cases(path, fields, expected):
    rows = read_csv(path)
    cases = {tuple(row[field] for field in fields): row for row in rows}
    if len(cases) != len(rows) or cases.keys() != expected:
        raise ValueError(f"Missing, duplicated or unexpected cases in {path}")
    if any(row["constraints_valid"] != "True" or row["solver_status"] != "Optimal" for row in rows):
        raise ValueError(f"Invalid or nonoptimal case in {path}")
    return cases


def forecast_table(design):
    profiles = read_csv(design / "selected_profiles.csv")
    stats = json.loads((design / "selected_forecast_metrics.json").read_text(encoding="utf-8"))
    actual = np.array([float(row["actual_demand_kw"]) for row in profiles]).reshape(-1, 48)
    center = actual.argmax(axis=1)
    window = center[:, None] + np.arange(-4, 5)
    if window.min() < 0 or window.max() >= 48:
        raise ValueError("All selected days must have a full nine-slot peak window")
    models = ("lstm", "seasonal_naive")
    summaries = {}
    for model in models:
        forecast = np.array([float(row[FORECASTS[model]]) for row in profiles]).reshape(-1, 48)
        actual_peak = actual.max(axis=1)
        forecast_peak = forecast.max(axis=1)
        i = np.arange(len(actual))[:, None]
        normalized_error = abs(
            actual[i, window] / actual_peak[:, None] - forecast[i, window] / forecast_peak[:, None]
        )
        summaries[model] = (
            stats[model]["rmse_kw"] / 1e6,
            stats[model]["mape_nonzero_percent"],
            mean(abs(forecast_peak - actual_peak) / actual_peak) * 100,
            stats[model]["mean_absolute_peak_time_error_minutes"],
            normalized_error.mean() * 100,
        )
    labels = ("RMSE [GW]", "MAPE [%]", "Peak MAPE [%]",
              "Peak-time MAE [min]", "Normalized peak-window MAE [pt]")
    return [{"metric": label, **{model: f"{summaries[model][i]:.{1 if i == 3 else 2}f}" for model in models}}
            for i, label in enumerate(labels)]


def reduction_summary(rows):
    series = np.asarray([float(row[RATE]) for row in rows])
    q1, median, q3 = np.quantile(series, (.25, .50, .75), method="linear")
    return {
        "mean_percent": f"{series.mean():.2f}", "median_percent": f"{median:.2f}",
        "q1_percent": f"{q1:.2f}", "q3_percent": f"{q3:.2f}",
        "min_percent": f"{series.min():.2f}", "max_percent": f"{series.max():.2f}",
        "peak_amplification_days": sum(float(row[POWER]) < -POWER_TOLERANCE_KW for row in rows),
    }


def seasonal_table(selected, baseline, shifts):
    output = []
    for season in SEASONS:
        days = [day for day, row in selected.items() if row["season"] == season]
        results = reduction_summary([baseline[day, "lstm"] for day in days])
        peak_amplification = sum(float(shifts[day, str(slot)][POWER]) < -POWER_TOLERANCE_KW
                                 for day in days for slot in SLOTS if slot)
        output.append({
            "season": season, "days": len(days),
            **{field: results[field] for field in ("mean_percent", "median_percent", "q1_percent", "q3_percent")},
            "peak_amplification_nonzero_cases": peak_amplification,
            "nonzero_cases": len(days) * (len(SLOTS) - 1),
        })
    return output


def lambda_table(path):
    means = {(float(row["lambda"]), row["condition"], int(row["shift_slots"] or 0)):
             float(row["mean_reduction_percent"]) for row in read_csv(path) if row["season"] == "all"}
    output = []
    for penalty in sorted({key[0] for key in means}):
        maximum = max(means[penalty, "lstm_shift", slot] for slot in SLOTS)
        best = [slot for slot in SLOTS if abs(means[penalty, "lstm_shift", slot] - maximum) <= 1e-9]
        output.append({
            "lambda": f"{penalty:g}",
            **{condition: f"{means[penalty, condition, 0]:.2f}"
               for condition in ("lstm_shift", *BASE_MODELS[1:], *CORRECTIONS)},
            "best_lstm_shift_h": "All 17 tied" if len(best) == len(SLOTS) else ", ".join(f"{s / 2:g}" for s in best),
        })
    return output


def bess_table(configs, summary):
    means = {(row["config_id"], row["forecast_model"]): float(row["mean_actual_peak_reduction_percent"])
             for row in read_csv(summary) if row["season"] == "all"}
    return [{
        "configuration": config["config_id"],
        "power_gw": f"{float(config['max_power_kw']) / 1e6:.4f}",
        "capacity_gwh": f"{float(config['battery_capacity_kwh']) / 1e6:.4f}",
        "duration_hours": f"{float(config['duration_hours']):.2f}",
        "start_end_soc_percent": f"{100 * float(config['initial_soc_fraction']):g}",
        "round_trip_efficiency_percent": f"{100 * float(config['round_trip_efficiency']):g}",
        **{model: f"{means[config['config_id'], model]:.2f}" for model in BASE_MODELS},
    } for config in read_csv(configs)]


def waveform(corrections, schedules, capacity, output):
    day = "2026-03-03"
    rows = sorted((row for row in read_csv(schedules) if row["date"] == day and row["condition"] == "timing_oracle"),
                  key=lambda row: int(row["slot"]))
    if [int(row["slot"]) for row in rows] != list(range(48)):
        raise ValueError("Figure 8 requires all 48 half-hour slots for 3 March")
    values = {field: [float(row[field]) for row in rows] for field in (
        "actual_demand_kw", "forecast_demand_kw", "controlled_actual_kw",
        "charge_kw", "discharge_kw", "soc_start_kwh", "soc_end_kwh",
    )}
    actual, controlled = (values[key] for key in ("actual_demand_kw", "controlled_actual_kw"))
    evening, morning = int(np.argmax(actual)), int(np.argmax(controlled))
    case = {
        "date": day, "shift_slots": int(corrections[day, "timing_oracle"]["shift_slots"]),
        "morning_slot": morning, "evening_slot": evening,
        "uncontrolled_peak_gw": actual[evening] / 1e6, "controlled_peak_gw": controlled[morning] / 1e6,
        "morning_charge_gw": values["charge_kw"][morning] / 1e6,
        "evening_discharge_gw": values["discharge_kw"][evening] / 1e6,
        "amplification_gw": (controlled[morning] - actual[evening]) / 1e6,
    }
    plot_waveform(values, case, capacity, output)
    return case


def render(run):
    design = run / "design"
    selected = {row["date"]: row for row in read_csv(design / "selected_days.csv")}
    day_order = list(selected)
    days = sorted(selected)
    baseline = load_cases(run / "baseline/daily_metrics.csv", ("date", "forecast_model"),
                          {(day, model) for day in days for model in BASE_MODELS})
    shifts = load_cases(run / "shifts/daily_metrics.csv", ("date", "shift_slots"),
                        {(day, str(slot)) for day in days for slot in SLOTS})
    corrections = load_cases(run / "corrections/daily_metrics.csv", ("date", "condition"),
                              {(day, model) for day in days
                               for model in ("original_lstm", "perfect_foresight_oracle", *CORRECTIONS)})
    figures, tables = run / "figures", run / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    write_csv(tables / "table3.csv", forecast_table(design))

    row_groups = {model: [baseline[day, model] for day in days] for model in BASE_MODELS}
    row_groups.update({model: [corrections[day, model] for day in days] for model in CORRECTIONS})
    write_csv(tables / "table5.csv", [{"condition": model, **reduction_summary(rows)}
                                    for model, rows in row_groups.items()])
    write_csv(tables / "tableS3.csv", seasonal_table(selected, baseline, shifts))
    if (run / "lambda/summary.csv").is_file():
        write_csv(tables / "tableS1.csv", lambda_table(run / "lambda/summary.csv"))
    if (run / "bess/summary.csv").is_file():
        write_csv(tables / "tableS2.csv", bess_table(design / "battery_configs.csv", run / "bess/summary.csv"))

    plot_shift([[float(shifts[day, str(slot)][RATE]) for day in days] for slot in SLOTS], SLOTS, figures)
    plot_baseline([[float(baseline[day, model][RATE]) for day in days] for model in BASE_MODELS], figures)
    effects = {model: {day: float(corrections[day, model][RATE]) - float(baseline[day, "lstm"][RATE])
                       for day in day_order} for model in CORRECTIONS}
    plot_corrections(effects, figures)
    capacity = float(read_csv(design / "battery_configs.csv")[0]["battery_capacity_kwh"])
    figure8 = waveform(corrections, run / "corrections/schedules.csv", capacity, figures)
    tol = POWER_TOLERANCE_KW
    comparison = [float(shifts[day, str(slot)][POWER]) for day in days for slot in SLOTS if slot]
    lags = [float(shifts[day, str(slot)][POWER]) - float(shifts[day, "0"][POWER])
            for day in days for slot in SLOTS if slot]
    (tables / "figure_summary.json").write_text(json.dumps({
        "figure5": {"nonzero_cases": len(comparison), "worse_than_zero_shift": sum(v < -tol for v in lags),
                    "peak_amplification": sum(v < -tol for v in comparison),
                    "near_zero": sum(abs(v) <= tol for v in comparison)},
        "figure7_mean_change_pp": {model: mean(effects[model].values()) for model in CORRECTIONS},
        "figure8": figure8,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    render(parser.parse_args().run_dir)
