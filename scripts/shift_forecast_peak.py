#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shift forecast series peak timing by circular rolling.
- Supports relative shift by hours (positive = later, negative = earlier)
- Or align peak to a target hour (e.g., 17.0)
- Works for single-day or multi-day CSVs (use --by-day for per-day independent shift)

Examples:
  python scripts/shift_forecast_peak.py \
    --input data/fixtures/lstm_august_2024_08_05.csv \
    --forecast-col demand_forecast_mw \
    --hours 1 \
    --output data/fixtures/lstm_august_2024_08_05__shift_+1h.csv

  python scripts/shift_forecast_peak.py \
    --input data/fixtures/lstm_august_2024_08_05.csv \
    --forecast-col demand_forecast_mw \
    --target-hour 17 \
    --output data/fixtures/lstm_august_2024_08_05__shift_to_17h.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def infer_slot_minutes(df: pd.DataFrame, ts_col: str) -> int:
    dt = df[ts_col].diff().dropna().median()
    return int(round(dt.total_seconds() / 60))


def roll_series_by_slots(s: pd.Series, slots: int) -> pd.Series:
    if slots == 0:
        return s.copy()
    arr = s.to_numpy()
    arr2 = np.roll(arr, slots)  # +slots -> later timestamps
    return pd.Series(arr2, index=s.index, name=s.name)


def compute_slots_to_target_hour(df: pd.DataFrame, ts_col: str, col: str, target_hour: float) -> int:
    peak_idx = df[col].idxmax()
    peak_ts = df.loc[peak_idx, ts_col]
    slot_min = infer_slot_minutes(df, ts_col)
    day0 = pd.Timestamp(peak_ts.date())
    target_ts = day0 + pd.Timedelta(hours=target_hour)
    delta_min = int(round((target_ts - peak_ts).total_seconds() / 60))
    slots = int(round(delta_min / slot_min))
    return slots


def shift_group(df: pd.DataFrame, ts_col: str, col: str, slots: int) -> pd.DataFrame:
    before_peak_ts = df.loc[df[col].idxmax(), ts_col]
    df[col] = roll_series_by_slots(df[col], slots)
    after_peak_ts = df.loc[df[col].idxmax(), ts_col]
    print(f"[{before_peak_ts.date()}] peak {before_peak_ts.time()} -> {after_peak_ts.time()} (slots {slots:+d})")
    return df


def main():
    ap = argparse.ArgumentParser(description="Shift forecast peak by hours or target hour (circular roll)")
    ap.add_argument("--input", required=True, help="Input CSV path (must contain timestamp column)")
    ap.add_argument("--output", help="Output CSV path (auto-named if omitted)")
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--forecast-col", default="demand_forecast_mw", help="Forecast column to shift")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--hours", type=float, default=None, help="Relative shift in hours (+later, -earlier)")
    group.add_argument("--target-hour", type=float, default=None, help="Align peak to this hour in day (e.g., 17 or 15.5)")
    group.add_argument("--slots", type=int, default=None, help="Relative shift by slots (time steps), +later/-earlier")
    ap.add_argument("--by-day", action="store_true", help="Shift per day independently (for multi-day CSV)")
    args = ap.parse_args()

    in_path = Path(args.input)
    ts_col = args.timestamp_col
    col = args.forecast_col

    df = pd.read_csv(in_path)
    if ts_col not in df.columns:
        raise SystemExit(f"timestamp column '{ts_col}' not found")
    if col not in df.columns:
        raise SystemExit(f"forecast column '{col}' not found")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    # one of hours/target-hour/slots is guaranteed by mutually exclusive group

    if args.by_day:
        out_chunks = []
        for day, g in df.groupby(df[ts_col].dt.date, sort=True):
            g = g.copy()
            if args.target_hour is not None:
                slots = compute_slots_to_target_hour(g, ts_col, col, args.target_hour)
            elif args.slots is not None:
                slots = int(args.slots)
            else:
                slot_min = infer_slot_minutes(g, ts_col)
                slots = int(round((args.hours * 60) / slot_min))
            out_chunks.append(shift_group(g, ts_col, col, slots))
        out_df = pd.concat(out_chunks).sort_values(ts_col).reset_index(drop=True)
    else:
        if args.target_hour is not None:
            slots = compute_slots_to_target_hour(df.copy(), ts_col, col, args.target_hour)
        elif args.slots is not None:
            slots = int(args.slots)
        else:
            slot_min = infer_slot_minutes(df, ts_col)
            slots = int(round((args.hours * 60) / slot_min))
        out_df = shift_group(df.copy(), ts_col, col, slots)

    if args.output:
        out_path = Path(args.output)
    else:
        if args.slots is not None:
            tag = f"shift_{args.slots:+d}slot"
        else:
            tag = f"shift_{args.hours:+.2f}h" if args.hours is not None else f"shift_to_{args.target_hour:g}h"
        out_path = in_path.with_name(in_path.stem + f"__{tag}" + in_path.suffix)

    out_df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
