#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSTM 需要予測をバッチ実行する。

指定期間の各日について、前日までの履歴で学習済みの LSTM から
48 コマ（24 時間）の需要予測を生成し、入力 CSV にマージして保存する。

注意: 乱数シードは固定していないため、実行ごとに予測値は変動する。
論文掲載値の厳密な再現には data/fixtures/ の同梱予測 CSV を使うこと。

Example:
  python scripts/run_lstm_demand_prediction.py \
    --input data/output/lstm_input_2024.csv \
    --output data/output/lstm_predictions_2024_august.csv \
    --start-day 2024-08-01 --end-day 2024-09-01 --epochs 10
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from peakshift.lstm_demand_forecaster import LSTMDemandForecaster


def run_batch_prediction(
    csv_path: Path,
    start_day_str: str,
    end_day_str: str,
    output_csv_path: Path,
    seq_len: int = 96,
    horizon: int = 48,
    epochs: int = 10,
):
    """指定期間の日々についてLSTM需要予測を実行し、結果をCSVに保存する"""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    forecaster = LSTMDemandForecaster(seq_len=seq_len, horizon=horizon)

    start_day = pd.Timestamp(start_day_str)
    end_day = pd.Timestamp(end_day_str)

    print(f"Fitting model with data until {start_day_str}...")
    forecaster.fit_until(df, start_day, epochs=epochs)

    day = start_day
    all_preds = []

    print("Starting daily prediction loop...")
    while day < end_day:
        print(f"  Predicting for {day.date()}...")
        try:
            pred_df = forecaster.predict_day(df, day)
            all_preds.append(pred_df)
        except ValueError as e:
            print(f"    Skipping {day.date()}: {e}")

        day += pd.Timedelta(days=1)

    if not all_preds:
        print("No predictions were generated.")
        return

    print("Merging predictions...")
    predictions_df = pd.concat(all_preds)

    final_df = df.merge(predictions_df[['timestamp', 'demand_predict_mw']], on="timestamp", how="left")
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv_path, index=False)
    print(f"Wrote merged output to: {output_csv_path}")


def main():
    ap = argparse.ArgumentParser(description='Run daily LSTM demand forecasts over a date range.')
    ap.add_argument('--input', required=True,
                    help='Input CSV (timestamp, demand_mw + numeric feature columns; see prepare_lstm_input.py)')
    ap.add_argument('--output', default='data/output/lstm_predictions.csv')
    ap.add_argument('--start-day', default='2024-08-01')
    ap.add_argument('--end-day', default='2024-09-01')
    ap.add_argument('--seq-len', type=int, default=96)
    ap.add_argument('--horizon', type=int, default=48)
    ap.add_argument('--epochs', type=int, default=10)
    args = ap.parse_args()

    run_batch_prediction(
        csv_path=Path(args.input),
        start_day_str=args.start_day,
        end_day_str=args.end_day,
        output_csv_path=Path(args.output),
        seq_len=args.seq_len,
        horizon=args.horizon,
        epochs=args.epochs,
    )


if __name__ == '__main__':
    main()
