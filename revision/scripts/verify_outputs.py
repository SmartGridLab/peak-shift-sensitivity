"""Check the generated figure inventory and manuscript values at printed precision."""

import argparse
import csv
import json
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1] / "data/manuscript_reference.json"


def verify(run, main_only):
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    for name, values in expected["tables_without_row_labels"].items():
        if main_only and name in ("tableS1.csv", "tableS2.csv"):
            continue
        with (run / "tables" / name).open(encoding="utf-8", newline="") as stream:
            rows = csv.reader(stream)
            next(rows)
            generated = [row[1:] for row in rows]
        if generated != values:
            for i in range(max(len(generated), len(values))):
                actual = generated[i] if i < len(generated) else None
                target = values[i] if i < len(values) else None
                if actual != target:
                    raise ValueError(f"{name}, data row {i + 1}: expected {target}, got {actual}")
        print(f"{name}: {len(values)} rows match the manuscript")

    for i in range(5, 9):
        for suffix in (".pdf", ".svg"):
            path = run / "figures" / f"fig{i}{suffix}"
            if path.stat().st_size == 0:
                raise ValueError(f"Empty output: {path}")
    stats = json.loads((run / "tables/figure_summary.json").read_text(encoding="utf-8"))
    if stats["figure5"] != expected["figure5"]:
        raise ValueError("Figure 5 event counts differ from the manuscript")
    avg = {key: f"{value:.2f}" for key, value in stats["figure7_mean_change_pp"].items()}
    if avg != expected["figure7_mean_change_pp"]:
        raise ValueError(f"Figure 7 mean changes differ: {avg}")
    for key, value in expected["figure8"].items():
        actual = stats["figure8"][key]
        if isinstance(actual, float):
            actual = f"{actual:.2f}"
        if actual != value:
            raise ValueError(f"Figure 8 {key}: expected {value}, got {actual}")
    print("Figures 5–8 exist. Figure 5 event counts, Figure 7 mean changes and Figure 8 case values match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--main-only", action="store_true", help="Skip the checks for Tables S1 and S2")
    args = parser.parse_args()
    verify(args.run_dir, args.main_only)
