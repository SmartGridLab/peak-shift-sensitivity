#!/usr/bin/env python3
"""Eight additional BESS scenarios, three forecasts, zero shift, 480 new cases."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.eneres_sensitivity import SolverSettings, run_sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, help="Separate frozen9-configuration snapshot; baseline design remains unchanged")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--time-limit-s", type=float, default=60)
    args = parser.parse_args()
    def progress(row, completed, total):
        if completed % 15 == 0 or completed == 1:
            print(f"[{completed}/{total}] {row['case_id']} ({row['execution']}): "
                  f"reduction={row['actual_peak_reduction_percent']:.2f}%", flush=True)
    try:
        _, summary = run_sensitivity(args.design_dir, args.baseline_dir, args.out_dir,
                                    SolverSettings(args.time_limit_s), command=[sys.executable, *sys.argv], progress=progress,
                                    config_dir=args.config_dir)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f"BESS comparison failed: {exc}\n")
    for row in summary:
        if row["season"] == "all":
            print(f"{row['config_id']}, {row['forecast_model']}: "
                  f"mean reduction={row['mean_actual_peak_reduction_percent']:.4f}%; "
                  f"change vs baseline={row['mean_change_vs_baseline_pp']:+.4f} pp")
    print(f"Saved 480 new + 60 reused baseline cases: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
