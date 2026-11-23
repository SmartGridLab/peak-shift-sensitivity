#!/usr/bin/env python3
"""Target-day-truth peak corrections: 60 new solves plus 40 recorded references."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.eneres_oracle import SolverSettings, run_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--shift-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--time-limit-s", type=float, default=60)
    args = parser.parse_args()
    def progress(row, completed, total):
        print(f"[{completed}/{total}] {row['case_id']} ({row['execution']}): "
              f"reduction={row['actual_peak_reduction_percent']:.2f}%", flush=True)
    try:
        _, summary = run_diagnostics(args.design_dir, args.baseline_dir, args.shift_dir, args.out_dir,
                                     SolverSettings(args.time_limit_s), command=[sys.executable, *sys.argv], progress=progress)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f"Oracle diagnostics failed: {exc}\n")
    for r in summary:
        if r["season"] == "all":
            print(f"{r['condition']}: mean reduction={r['mean_actual_peak_reduction_percent']:.2f}%; "
                  f"change={r['mean_change_vs_original_pp']:+.2f} pp")
    print(f"Saved 60 newly solved cases + 40 reference cases: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
