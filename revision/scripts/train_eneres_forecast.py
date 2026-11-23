#!/usr/bin/env python3
"""Train and evaluate the fixed daily ENERES LSTM and seasonal-naive baseline."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast.eneres_day_ahead import TrainingConfig, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/output/eneres_revision/prepared_v2"))
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory; existing outputs are refused")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    def progress(row):
        print(f"Epoch {row['epoch']}: train MSE={row['train_mse_scaled']:.6f}, "
              f"validation MSE={row['validation_mse_scaled']:.6f}", flush=True)

    try:
        config = TrainingConfig(epochs=args.epochs, patience=args.patience, seed=args.seed, threads=args.threads)
        manifest, metrics = run_experiment(args.prepared_dir, args.out_dir, config,
                                           command=[sys.executable, *sys.argv], progress=progress)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Forecast experiment failed: {exc}\n")
    print(f"Selected epoch: {manifest['training']['best_epoch']}; "
          f"evaluation days: {manifest['training']['daily_windows']['evaluation']}")
    for name, values in metrics["evaluation"].items():
        print(f"{name}: RMSE={values['rmse_kw']:.2f} kW; WAPE={values['wape_percent']:.2f}%")
    print(f"Saved model, forecasts and run record: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
