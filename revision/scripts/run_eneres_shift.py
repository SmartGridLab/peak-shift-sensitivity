#!/usr/bin/env python3
"""Run 20 days x 17 cyclic LSTM profile shifts with the frozen baseline BESS."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.eneres_shift import SolverSettings, run_shifts


def main():
    """Require frozen design, two-stage baseline and a fresh output destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--time-limit-s", type=float, default=60)
    parser.add_argument("--reuse-zero-baseline", action="store_true", help="Reuse20 exact baseline cases; optimize only320 nonzero shifts")
    args = parser.parse_args()
    def progress(row, completed, total):
        print(f"[{completed}/{total}] {row['case_id']}: reduction={row['actual_peak_reduction_percent']:.2f}%", flush=True)
    try:
        _, summary = run_shifts(args.design_dir, args.baseline_dir, args.out_dir, SolverSettings(args.time_limit_s),
                                command=[sys.executable, *sys.argv], progress=progress, reuse_zero_baseline=args.reuse_zero_baseline)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f"Shift simulation failed: {exc}\n")
    for row in summary:
        if row["season"] == "all":
            print(f"shift={row['shift_slots']:+d}: mean={row['mean_actual_peak_reduction_percent']:.2f}%; "
                  f"worsened vs uncontrolled={row['worsened_vs_uncontrolled_days']}/20")
    print(f"Saved 340 verified cases: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
