#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
予測系列を -N〜+N コマシフトした各ケースで LP ピークシェービングを実行し、
削減率をまとめた CSV とグラフを出力する。

Example:
  python scripts/run_shift_analysis.py \
    --actual   data/fixtures/actual_demand_2024_08_05.csv \
    --forecast data/fixtures/lstm_forecast_2024_08_05.csv \
    --date 2024-08-05 \
    --out-dir data/output/peak_shift

シフト済みフィクスチャが fixtures/ に存在する場合は --fixture-dir で指定:
  python scripts/run_shift_analysis.py \
    --actual data/fixtures/actual_demand_2024_08_05.csv \
    --fixture-dir data/fixtures \
    --fixture-prefix lstm_august_2024_08_05__shift \
    --date 2024-08-05 --out-dir data/output/peak_shift
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from peakshift.lp_peak_shaving_optimizer import LPPeakShavingOptimizer

JST = 'Asia/Tokyo'


def load_series(csv_path: str, ts_col: str, val_col: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    df['_ts'] = pd.to_datetime(df[ts_col], errors='coerce')
    df['_ts'] = df['_ts'].dt.tz_localize(JST) if df['_ts'].dt.tz is None else df['_ts'].dt.tz_convert(JST)
    return df.set_index('_ts')[val_col].astype(float).sort_index()


def run_lp(actual_kw, forecast_kw, capacity_kwh, power_kw, eff, timestep_h, initial_soc_pct):
    initial_kwh = capacity_kwh * initial_soc_pct / 100.0
    opt = LPPeakShavingOptimizer(
        battery_capacity_kwh=capacity_kwh,
        max_power_kw=power_kw,
        charge_efficiency=eff,
        discharge_efficiency=eff,
        initial_soc_kwh=initial_kwh,
        action_penalty_lambda=1e-4,
        timestep_h=timestep_h,
        final_soc_equality=False,
    )
    res = opt.optimize_peak_shaving(
        forecast_kw.to_numpy(dtype=float),
        pd.DatetimeIndex(forecast_kw.index),
    )
    c_t = pd.Series(res['c_t'], index=forecast_kw.index)
    d_t = pd.Series(res['d_t'], index=forecast_kw.index)
    residual = (actual_kw + c_t - d_t).astype(float)
    return float(residual.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--actual', required=True, help='実績 CSV パス')
    ap.add_argument('--actual-col', default='demand_mw')
    ap.add_argument('--forecast', default=None, help='予測 CSV パス（shift=0 用）')
    ap.add_argument('--forecast-col', default='demand_forecast_mw')
    ap.add_argument('--fixture-dir', default=None, help='シフト済みフィクスチャのディレクトリ')
    ap.add_argument('--fixture-prefix', default='lstm_august_2024_08_05__shift',
                    help='フィクスチャファイル名プレフィックス（例: lstm_august_2024_08_05__shift）')
    ap.add_argument('--date', required=True, help='対象日 YYYY-MM-DD')
    ap.add_argument('--out-dir', default='data/output/peak_shift')
    ap.add_argument('--shift-min', type=int, default=-8)
    ap.add_argument('--shift-max', type=int, default=8)
    ap.add_argument('--capacity-kwh', type=float, default=40000.0)
    ap.add_argument('--power-kw', type=float, default=10000.0)
    ap.add_argument('--eff', type=float, default=0.92)
    ap.add_argument('--timestep-h', type=float, default=0.5)
    ap.add_argument('--initial-soc-pct', type=float, default=50.0)
    ap.add_argument('--version', default='')
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    target_day = pd.Timestamp(args.date).tz_localize(JST)
    actual_full = load_series(args.actual, 'timestamp', args.actual_col)
    mask = (actual_full.index >= target_day) & (actual_full.index < target_day + pd.Timedelta(days=1))
    actual_kw = actual_full.loc[mask]
    peak_before = float(actual_kw.max())

    # shift=0 の予測を読み込む
    # 優先順位: fixture_dir/_0slot.csv > --forecast
    # （両方指定時に --forecast が実績値ファイルでも _0slot.csv があれば上書きされる）
    forecast_0 = None
    if args.forecast:
        fst = load_series(args.forecast, 'timestamp', args.forecast_col)
        forecast_0 = fst.reindex(actual_kw.index).fillna(actual_kw)
    if args.fixture_dir:
        f0 = Path(args.fixture_dir) / f'{args.fixture_prefix}_0slot.csv'
        if f0.exists():
            fst = load_series(str(f0), 'timestamp', args.forecast_col)
            forecast_0 = fst.reindex(actual_kw.index).fillna(actual_kw)
            if args.forecast:
                print(f'[INFO] shift=0 は --forecast より {f0.name} を優先して使用します')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ver = f'_{args.version}' if args.version else ''
    date_tag = target_day.strftime('%Y%m%d')

    records = []
    for shift in range(args.shift_min, args.shift_max + 1):
        # 予測系列を決定
        if shift == 0 and forecast_0 is not None:
            forecast_kw = forecast_0
        elif args.fixture_dir:
            sign = f'+{shift}' if shift > 0 else str(shift)
            fname = f'{args.fixture_prefix}_{sign}slot.csv'
            fpath = Path(args.fixture_dir) / fname
            if not fpath.exists():
                print(f'  [SKIP] {fpath} not found')
                records.append({'shift_steps': shift, 'shift_hours': shift * args.timestep_h,
                                'peak_before_kw': peak_before, 'peak_after_kw': None, 'reduction_rate_pct': None})
                continue
            fst = load_series(str(fpath), 'timestamp', args.forecast_col)
            forecast_kw = fst.reindex(actual_kw.index).fillna(actual_kw)
        elif forecast_0 is not None:
            forecast_kw = forecast_0.shift(shift).fillna(actual_kw)
        else:
            print('[ERROR] --forecast または --fixture-dir のいずれかを指定してください')
            sys.exit(1)

        try:
            peak_after = run_lp(
                actual_kw, forecast_kw,
                args.capacity_kwh, args.power_kw, args.eff, args.timestep_h, args.initial_soc_pct,
            )
            reduction = (peak_before - peak_after) / peak_before * 100.0
            print(f'  shift={shift:+3d}: before={peak_before:.0f}, after={peak_after:.0f}, reduction={reduction:.2f}%')
        except Exception as e:
            print(f'  [WARN] shift={shift}: LP failed: {e}')
            peak_after = None
            reduction = None

        records.append({
            'shift_steps': shift,
            'shift_hours': shift * args.timestep_h,
            'peak_before_kw': peak_before,
            'peak_after_kw': peak_after,
            'reduction_rate_pct': reduction,
        })

    df = pd.DataFrame(records)

    csv_path = out_dir / f'shift_analysis_{date_tag}{ver}.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nCSV: {csv_path}')

    valid = df.dropna(subset=['reduction_rate_pct'])
    plt.style.use('seaborn-v0_8-white')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Forecast Shift Sensitivity  {args.date}', fontsize=16)

    colors = ['#e74c3c' if s == 0 else '#3498db' for s in valid['shift_steps']]
    bars = ax1.bar(valid['shift_steps'], valid['reduction_rate_pct'], color=colors, alpha=0.85, width=0.7)
    for bar, val in zip(bars, valid['reduction_rate_pct']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=1.2)
    ax1.set_xlabel(f'Forecast Shift (steps, {args.timestep_h*60:.0f}min/step)', fontsize=13)
    ax1.set_ylabel('Peak Reduction Rate (%)', fontsize=13)
    ax1.set_title('Peak Reduction Rate per Shift', fontsize=14)
    ax1.set_xticks(range(args.shift_min, args.shift_max + 1))
    ax1.set_xticklabels(
        [f'{s:+d}\n({s*args.timestep_h:+.1f}h)' for s in range(args.shift_min, args.shift_max + 1)],
        fontsize=8,
    )
    ax1.grid(axis='y', alpha=0.3)

    ax2.plot(valid['shift_steps'], valid['peak_after_kw'] / 10000, 'o-', color='#3498db', linewidth=2)
    ax2.axhline(peak_before / 10000, color='gray', linestyle=':', linewidth=1.5,
                label=f'Before: {peak_before/10000:.2f}')
    ax2.axvline(0, color='gray', linestyle='--', linewidth=1.2)
    ax2.set_xlabel('Forecast Shift (steps)', fontsize=13)
    ax2.set_ylabel('Peak Demand (10,000 kW)', fontsize=13)
    ax2.set_title('Post-Control Peak Demand per Shift', fontsize=14)
    ax2.set_xticks(range(args.shift_min, args.shift_max + 1))
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    png_path = out_dir / f'shift_analysis_{date_tag}{ver}.png'
    plt.savefig(png_path, dpi=args.dpi)
    plt.close()
    print(f'Graph: {png_path}')

    best = df.loc[df['reduction_rate_pct'].idxmax()]
    print(f'\nBest: shift={best["shift_steps"]:+.0f} ({best["shift_hours"]:+.1f}h) → {best["reduction_rate_pct"]:.2f}%')


if __name__ == '__main__':
    main()
