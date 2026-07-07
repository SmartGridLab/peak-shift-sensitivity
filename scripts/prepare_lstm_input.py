#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TEPCO エリア需給実績 CSV から LSTM 入力用 CSV を生成する。

論文の実験で LSTM 学習・推論の入力に使った CSV と同じ数値列スキーマ
（timestamp, demand_mw, probability, year, month, day, hour, slot_index,
minute, day_of_week, day_of_year）を決定論的に再構築する。
LSTMDemandForecaster は「timestamp / demand_kW 以外の全数値列」を特徴量に
使うため、このスキーマが変わると入力次元が変わり別モデルになる点に注意。

Example:
  python scripts/prepare_lstm_input.py \
    --tepco-dir data/tepco \
    --year 2024 \
    --output data/output/lstm_input_2024.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from peakshift.tepco_loader import TEPCODataLoader


def build_lstm_input(tepco_dir: Path, year: int) -> 'pd.DataFrame':
    import pandas as pd  # noqa: F401 (型コメント用)

    paths = sorted(tepco_dir.glob('*.csv'))
    if not paths:
        raise SystemExit(f'No CSV files found in {tepco_dir}. '
                         'Run scripts/download_tepco_data.sh first.')
    df = TEPCODataLoader.load_annual_demand_csv(paths, year)

    ts = df['timestamp']
    df['probability'] = 0.0
    df['year'] = ts.dt.year
    df['month'] = ts.dt.month
    df['day'] = ts.dt.day
    df['hour'] = ts.dt.hour
    df['minute'] = ts.dt.minute
    df['slot_index'] = ts.dt.hour * 2 + ts.dt.minute // 30
    df['day_of_week'] = ts.dt.dayofweek
    df['day_of_year'] = ts.dt.dayofyear

    cols = ['timestamp', 'demand_mw', 'probability', 'year', 'month', 'day',
            'hour', 'slot_index', 'minute', 'day_of_week', 'day_of_year']
    return df[cols]


def main():
    ap = argparse.ArgumentParser(description='Build the LSTM input CSV from TEPCO area demand CSVs.')
    ap.add_argument('--tepco-dir', default='data/tepco', help='Directory containing eria_jukyu_*.csv / area-*.csv')
    ap.add_argument('--year', type=int, default=2024)
    ap.add_argument('--output', default='data/output/lstm_input_2024.csv')
    args = ap.parse_args()

    out = build_lstm_input(Path(args.tepco_dir), args.year)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f'saved: {out_path} ({len(out)} rows, {out["timestamp"].min()} .. {out["timestamp"].max()})')


if __name__ == '__main__':
    main()
