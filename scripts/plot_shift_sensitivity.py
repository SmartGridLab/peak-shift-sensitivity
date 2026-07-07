#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot peak reduction sensitivity vs forecast peak shift (in slots).

X-axis: shift in slots (-3, -2, -1, 0, 1, 2, 3)
Y-axis: peak reduction percentage vs actual peak before control

Uses LP optimizer with forecast as input, applies schedule to actual demand.
Assumes kW/kWh units.

Example:
  python scripts/plot_shift_sensitivity.py \
    --csv data/fixtures/lstm_august_2024_08_05.csv \
    --date 2024-08-05 \
    --actual-col demand_mw --forecast-col demand_forecast_mw \
    --eff 0.92 --capacity-kwh 40000 --power-kw 10000 \
    --out-dir data/output/peak_shift --version eff092
"""

import argparse
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LP_OPTIMIZER_PATH = Path(__file__).resolve().parents[1] / 'src' / 'peakshift' / 'lp_peak_shaving_optimizer.py'
try:
    spec = importlib.util.spec_from_file_location('lp_peak_shaving_optimizer', LP_OPTIMIZER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load module spec from {LP_OPTIMIZER_PATH}')
    lp_optimizer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lp_optimizer_module)
    LPPeakShavingOptimizer = lp_optimizer_module.LPPeakShavingOptimizer
except Exception as e:
    raise SystemExit(f"Failed to import LPPeakShavingOptimizer: {e}")

JST = 'Asia/Tokyo'


def read_day(csv_path: str, date_str: str, ts_col: str, actual_col: str, forecast_col: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    if ts_col not in df.columns or actual_col not in df.columns or forecast_col not in df.columns:
        raise SystemExit(f"Required columns not found. Have: {list(df.columns)}")
    ts = pd.to_datetime(df[ts_col], errors='coerce')
    ts = ts.dt.tz_localize(JST) if ts.dt.tz is None else ts.dt.tz_convert(JST)
    df = df.assign(timestamp=ts).dropna(subset=['timestamp']).sort_values('timestamp')
    target_day = pd.Timestamp(date_str).tz_localize(JST)
    mask = (df['timestamp'] >= target_day) & (df['timestamp'] < target_day + pd.Timedelta(days=1))
    day = df.loc[mask].copy()
    if day.empty:
        raise SystemExit(f"No data for {date_str} in {csv_path}")
    day = day.set_index('timestamp')
    actual_kw = day[actual_col].astype(float)
    forecast_kw = day[forecast_col].astype(float)
    return actual_kw, forecast_kw


def roll_by_slots(s: pd.Series, slots: int) -> pd.Series:
    if slots == 0:
        return s.copy()
    arr = s.to_numpy(dtype=float)
    return pd.Series(np.roll(arr, slots), index=s.index, name=s.name)


def compute_peak_reduction_pct(actual_kw: pd.Series, forecast_kw: pd.Series,
                               capacity_kwh: float, power_kw: float, eff: float,
                               initial_soc_pct: float, timestep_h: float,
                               action_penalty: float, final_soc_equality: bool) -> tuple[float, float, float]:
    # Build optimizer
    initial_kwh = capacity_kwh * (initial_soc_pct / 100.0)
    opt = LPPeakShavingOptimizer(
        battery_capacity_kwh=capacity_kwh,
        max_power_kw=power_kw,
        charge_efficiency=eff,
        discharge_efficiency=eff,
        initial_soc_kwh=initial_kwh,
        action_penalty_lambda=action_penalty,
        timestep_h=timestep_h,
        final_soc_equality=final_soc_equality,
    )
    demand_array = forecast_kw.to_numpy(dtype=float)
    time_index = pd.DatetimeIndex(forecast_kw.index)
    res = opt.optimize_peak_shaving(demand_array, time_index)
    c_t = pd.Series(res['c_t'], index=forecast_kw.index)
    d_t = pd.Series(res['d_t'], index=forecast_kw.index)
    residual_actual = (actual_kw + c_t - d_t).astype(float)

    peak_before = float(actual_kw.max(skipna=True))
    peak_after = float(residual_actual.max(skipna=True))
    reduction_pct = (peak_before - peak_after) / peak_before * 100.0
    return peak_before, peak_after, reduction_pct


def main():
    ap = argparse.ArgumentParser(description='Plot peak reduction vs shift slots (LP-based).')
    ap.add_argument('--csv', required=True, help='Input CSV path containing actual and forecast columns')
    ap.add_argument('--date', required=True, help='Target date (YYYY-MM-DD) JST')
    ap.add_argument('--ts-col', default='timestamp')
    ap.add_argument('--actual-col', default='demand_mw')
    ap.add_argument('--forecast-col', default='demand_forecast_mw')
    ap.add_argument('--slots-min', type=int, default=-3)
    ap.add_argument('--slots-max', type=int, default=3)
    ap.add_argument('--capacity-kwh', type=float, default=40000.0)
    ap.add_argument('--power-kw', type=float, default=10000.0)
    ap.add_argument('--eff', type=float, default=0.92)
    ap.add_argument('--initial-soc-pct', type=float, default=50.0)
    ap.add_argument('--timestep-h', type=float, default=0.5)
    ap.add_argument('--action-penalty', type=float, default=1e-4)
    ap.add_argument('--final-soc-equality', action='store_true', help='Enforce E(T)=E(0)')
    ap.add_argument('--out-dir', default='data/output/peak_shift')
    ap.add_argument('--format', choices=['png','svg','both'], default='png', help='Output image format')
    ap.add_argument('--dpi', type=int, default=600, help='Output image DPI for raster formats')
    ap.add_argument('--version', default='', help='Suffix tag for output filename')
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    actual_kw, forecast_kw0 = read_day(args.csv, args.date, args.ts_col, args.actual_col, args.forecast_col)

    slots_list = list(range(args.slots_min, args.slots_max + 1))
    reductions = []
    peaks_before = []
    peaks_after = []

    for s in slots_list:
        f_shift = roll_by_slots(forecast_kw0, s)
        pb, pa, r = compute_peak_reduction_pct(
            actual_kw, f_shift,
            capacity_kwh=args.capacity_kwh,
            power_kw=args.power_kw,
            eff=args.eff,
            initial_soc_pct=args.initial_soc_pct,
            timestep_h=args.timestep_h,
            action_penalty=args.action_penalty,
            final_soc_equality=args.final_soc_equality,
        )
        reductions.append(r)
        peaks_before.append(pb)
        peaks_after.append(pa)
        print(f"slot {s:+d}: Peak Before {pb:.1f} kW, After {pa:.1f} kW, Reduction {r:.2f}%")

    # Bar plot (paper style): white bg, no grid, larger fonts, with legend
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams.update({'font.size': 16})
    
    # Try to use Japanese font if available
    try:
        import matplotlib.font_manager as fm
        # Try common Japanese fonts
        japanese_fonts = ['IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP', 'Takao Gothic']
        available_fonts = {f.name for f in fm.fontManager.ttflist}
        japanese_font = next((font for font in japanese_fonts if font in available_fonts), None)
        if japanese_font:
            plt.rcParams['font.family'] = japanese_font
    except Exception:
        pass  # Fall back to default font
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Create bars with separate labels for legend
    colors = ['#2ca02c' if r>=0 else '#d62728' for r in reductions]
    bars = ax.bar([str(s) for s in slots_list], reductions, color=colors, alpha=0.85)
    
    # Create legend handles (use English if Japanese font not available)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', alpha=0.85, label='Peak reduction'),
        Patch(facecolor='#d62728', alpha=0.85, label='Peak increase')
    ]
    ax.legend(handles=legend_elements, fontsize=16, loc='best', frameon=True, edgecolor='black')

    ax.axhline(0, color='black', linewidth=2.2, alpha=0.9, zorder=3)
    ax.set_xlabel('Forecast time shift Δ [slots; 1 slot = 30 min]', fontsize=16)
    ax.set_ylabel('Peak-reduction rate [%]', fontsize=16)
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=16)
    # y-axis padding so annotations don't touch axis spines
    data_min, data_max = float(min(reductions)), float(max(reductions))
    base_range = max(1e-6, data_max - data_min)
    pad = max(0.5, 0.12 * base_range)  # more generous padding to avoid overlap with spines
    top_pad = max(pad, 4.0)
    ax.set_ylim(data_min - pad, data_max + top_pad)

    # Annotate only the paper callouts that matter for interpretation.
    ymin, ymax = ax.get_ylim()
    offset = 0.03 * (ymax - ymin)
    inner_margin = 0.04 * (ymax - ymin)
    annotation_specs = {
        -7: {'text': '8.27%', 'xytext': (0, 18)},
        -6: {'text': '7.95%', 'xytext': (0, 6)},
        0: {'text': '2.56%', 'xytext': (0, 18)},
        8: {'text': '-15.99%', 'xytext': (-10, -10)},
    }
    for rect, slot, r in zip(bars, slots_list, reductions):
        spec = annotation_specs.get(slot)
        if spec is None:
            continue
        x = rect.get_x() + rect.get_width() / 2
        y = min(r, ymax - inner_margin) if r >= 0 else max(r, ymin + inner_margin)
        ax.annotate(
            spec['text'],
            xy=(x, y),
            xytext=spec['xytext'],
            textcoords='offset points',
            ha='center',
            va='bottom' if r >= 0 else 'top',
            fontsize=16,
            clip_on=False,
        )

    zero_line_labels = {
        -7: {'label': 'Best case', 'ha': 'center', 'xytext': (0, -10), 'va': 'top'},
        0: {'label': 'No shift', 'ha': 'center', 'xytext': (0, -10), 'va': 'top'},
        8: {'label': 'Worst case', 'ha': 'center', 'xytext': (0, 8), 'va': 'bottom'},
    }
    for rect, slot in zip(bars, slots_list):
        item = zero_line_labels.get(slot)
        if item is None:
            continue
        ax.annotate(
            item['label'],
            xy=(rect.get_x() + rect.get_width() / 2, 0),
            xytext=item['xytext'],
            textcoords='offset points',
            ha=item['ha'],
            va=item['va'],
            fontsize=16,
            clip_on=False,
        )

    version_suffix = f"_{args.version}" if args.version else ''
    base_name = f"shift_sensitivity_{pd.to_datetime(args.date).strftime('%Y%m%d')}{version_suffix}"
    plt.tight_layout()
    if args.format in ('png','both'):
        out_png = Path(args.out_dir) / f"{base_name}.png"
        plt.savefig(out_png, dpi=args.dpi)
        print(f"Saved bar chart (PNG) to: {out_png}")
    if args.format in ('svg','both'):
        out_svg = Path(args.out_dir) / f"{base_name}.svg"
        plt.savefig(out_svg, format='svg')
        print(f"Saved bar chart (SVG) to: {out_svg}")
    plt.close()

    # Save CSV summary
    out_csv = Path(args.out_dir) / f"shift_sensitivity_{pd.to_datetime(args.date).strftime('%Y%m%d')}{version_suffix}.csv"
    pd.DataFrame({
        'shift_slots': slots_list,
        'peak_before_kw': peaks_before,
        'peak_after_kw': peaks_after,
        'reduction_pct': reductions,
    }).to_csv(out_csv, index=False)
    print(f"Saved summary CSV to: {out_csv}")


if __name__ == '__main__':
    main()
