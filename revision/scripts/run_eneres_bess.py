#!/usr/bin/env python3
"""Run the frozen 20-day, three-forecast baseline BESS simulation."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.eneres_bess import SolverSettings, run_baseline


def main():
    """Keep all inputs read-only and require a fresh destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--time-limit-s", type=float, default=60)
    args = parser.parse_args()
    def progress(row, completed, total):
        print(f"[{completed}/{total}] {row['case_id']}: actual peak reduction "
              f"{row['actual_peak_reduction_percent']:.2f}% ({row['solver_status']})", flush=True)
    try:
        _, summaries = run_baseline(args.design_dir, args.out_dir, SolverSettings(args.time_limit_s),
                                    command=[sys.executable, *sys.argv], progress=progress)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f"BESS simulation failed: {exc}\n")
    for row in summaries:
        if row["season"] == "all":
            print(f"{row['forecast_model']}: mean reduction={row['mean_actual_peak_reduction_percent']:.2f}%; "
                  f"worsened days={row['worsened_days']}")
    print(f"Saved 60 verified baseline cases: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
