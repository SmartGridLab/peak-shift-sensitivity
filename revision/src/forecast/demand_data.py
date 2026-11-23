"""Validated ENERES half-hour energy data, in kW, without fitting a scaler.

The source date/start labels are assumed to be Japan local time. Each timestamp
labels the beginning of its interval. Raw source files are never modified.
"""

import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional, Sequence
from zoneinfo import ZoneInfo


JST = ZoneInfo("Asia/Tokyo")
HALF_HOUR = timedelta(minutes=30)
REQUIRED_COLUMNS = {"area_name", "date", "start", "end", "demand_value", "unit"}


@dataclass(frozen=True)
class DemandRecord:
    timestamp: datetime
    demand_kw: float


@dataclass(frozen=True)
class DemandDataset:
    records: tuple[DemandRecord, ...]
    area_name: str
    source_sha256: str


@dataclass(frozen=True)
class SplitPeriods:
    start: date = date(2024, 4, 1)
    validation_start: date = date(2025, 2, 1)
    evaluation_start: date = date(2025, 4, 1)
    end_exclusive: date = date(2026, 4, 1)

    def __post_init__(self):
        if not self.start < self.validation_start < self.evaluation_start < self.end_exclusive:
            raise ValueError("Split boundaries must be strictly increasing.")


def _minutes(value: str, allow_midnight_end: bool = False) -> int:
    parts = value.split(":")
    if len(parts) != 2 or any(len(p) != 2 or not p.isdigit() for p in parts):
        raise ValueError(f"Invalid HH:MM time: {value!r}")
    hour, minute = map(int, parts)
    if allow_midnight_end and hour == 24 and minute == 0:
        return 1440
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"Invalid time: {value!r}")
    return hour * 60 + minute


def load_eneres_demand(path: Path) -> DemandDataset:
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    fields = list(reader.fieldnames or ())
    # DictReader silently keeps the last of duplicated names, so require the exact header.
    duplicated = sorted({c for c in fields if fields.count(c) > 1})
    if duplicated:
        raise ValueError(f"Duplicate source columns: {duplicated}")
    missing = REQUIRED_COLUMNS - set(fields)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    unexpected = set(fields) - REQUIRED_COLUMNS
    if unexpected:
        raise ValueError(f"Unexpected source columns: {sorted(unexpected)}")
    records = []
    areas = set()
    for row in reader:
        try:
            if None in row or any(row.get(c) is None for c in REQUIRED_COLUMNS):
                raise ValueError("Malformed CSV row")
            area = row["area_name"].strip()
            if not area:
                raise ValueError("Empty area_name")
            areas.add(area)
            if row["unit"].strip() != "kWh":
                raise ValueError("Expected interval energy in kWh")
            day = date.fromisoformat(row["date"].strip())
            start = _minutes(row["start"].strip())
            end = _minutes(row["end"].strip(), allow_midnight_end=True)
            if start % 30 or end - start != 30:
                raise ValueError("Expected aligned 30-minute intervals, ending at 24:00")
            try:
                energy = Decimal(row["demand_value"].strip())
            except InvalidOperation as exc:
                raise ValueError("Invalid demand_value") from exc
            if not energy.is_finite() or energy < 0:
                raise ValueError("demand_value must be finite and nonnegative")
            power = float(energy / Decimal("0.5"))
            if power == float("inf"):
                raise ValueError("demand_value is too large")
            timestamp = datetime.combine(day, time(), tzinfo=JST) + timedelta(minutes=start)
            records.append(DemandRecord(timestamp, power))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Source line {reader.line_num}: {exc}") from exc
    if not records:
        raise ValueError("No source records")
    if len(areas) != 1:
        raise ValueError("Expected a single area; select an area before preprocessing")
    records.sort(key=lambda r: r.timestamp)
    if records[0].timestamp.time() != time() or records[-1].timestamp.time() != time(23, 30):
        raise ValueError("Source must contain complete days (00:00 through 23:30)")
    for previous, current in zip(records, records[1:]):
        if current.timestamp == previous.timestamp:
            raise ValueError(f"Duplicate timestamp: {current.timestamp.isoformat()}")
        if current.timestamp - previous.timestamp != HALF_HOUR:
            raise ValueError(f"Missing half-hour slot after {previous.timestamp.isoformat()}")
    return DemandDataset(tuple(records), areas.pop(), hashlib.sha256(raw).hexdigest())


def split_demand(dataset: DemandDataset, periods: SplitPeriods) -> dict[str, tuple[DemandRecord, ...]]:
    """Split complete data with exclusive end boundaries; no normalization."""
    start = datetime.combine(periods.start, time(), tzinfo=JST)
    end = datetime.combine(periods.end_exclusive, time(), tzinfo=JST)
    if dataset.records[0].timestamp != start or dataset.records[-1].timestamp + HALF_HOUR != end:
        raise ValueError("Source coverage must exactly match the requested split period")
    result = {"train": [], "validation": [], "evaluation": []}
    for record in dataset.records:
        day = record.timestamp.date()
        if day < periods.validation_start:
            result["train"].append(record)
        elif day < periods.evaluation_start:
            result["validation"].append(record)
        else:
            result["evaluation"].append(record)
    if any(not rows for rows in result.values()):
        raise ValueError("Every split must be nonempty")
    return {name: tuple(rows) for name, rows in result.items()}


def _git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _code_record(repo_root: Path, code_files: Sequence[Path]) -> dict:
    """Identify the exact code: file hashes, and whether HEAD already contains them."""
    files = {}
    for path in code_files:
        path = path.resolve()
        try:
            key = path.relative_to(repo_root).as_posix()
        except ValueError:
            key = str(path)
        files[key] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            # Empty status means tracked and identical to HEAD; untracked prints "??".
            "committed": _git(repo_root, "ls-files", "--error-unmatch", "--", key) is not None
            and _git(repo_root, "status", "--porcelain", "--", key) == "",
        }
    return {
        "git_head": _git(repo_root, "rev-parse", "HEAD"),
        "working_tree_status": _git(repo_root, "status", "--porcelain"),
        "files": files,
        "all_files_committed": all(f["committed"] for f in files.values()),
    }


def export_prepared_data(
    dataset: DemandDataset, periods: SplitPeriods, source_path: Path, output_dir: Path,
    command: Optional[Sequence[str]] = None, code_files: Sequence[Path] = (),
) -> dict:
    """Write private experiment inputs and a reproducibility manifest.

    Refuse existing output directories so earlier inputs are not overwritten.
    Files are written to a temporary sibling directory and renamed into place
    only after the manifest is complete, so a failure leaves no partial output.
    demand.csv retains the full history for lag features at split boundaries.
    No scaler is fitted here: later training must fit it on train only.
    `command` and `code_files` (e.g. the calling script) are recorded so the
    manifest alone identifies how the outputs were produced.
    """
    splits = split_demand(dataset, periods)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"path": str(source_path.resolve()), "sha256": dataset.source_sha256,
                   "size_bytes": source_path.stat().st_size, "rows": len(dataset.records),
                   "area_name": dataset.area_name, "unit": "kWh"},
        "timestamp": {"timezone_assumption": "Asia/Tokyo", "label": "interval_start",
                      "interval_minutes": 30},
        "conversion": {"output_unit": "kW", "formula": "interval_kWh / 0.5 h"},
        "normalization": "not_applied; fit on train only in the training step",
        "periods": {name: getattr(periods, name).isoformat() for name in
                    ("start", "validation_start", "evaluation_start", "end_exclusive")},
        "command": list(command) if command is not None else None,
        "runtime": {"python": sys.version, "implementation": platform.python_implementation(),
                    "platform": platform.platform()},
        "code": _code_record(repo_root, [Path(__file__), *code_files]),
        "outputs": {},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        for name, rows in {"demand": dataset.records, **splits}.items():
            path = staging / f"{name}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "demand_kw"])
                writer.writerows((r.timestamp.isoformat(), r.demand_kw) for r in rows)
            manifest["outputs"][name] = {
                "file": path.name, "rows": len(rows), "days": len(rows) // 48,
                "first_timestamp": rows[0].timestamp.isoformat(),
                "last_timestamp": rows[-1].timestamp.isoformat(),
                "min_demand_kw": min(r.demand_kw for r in rows),
                "max_demand_kw": max(r.demand_kw for r in rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # mkdtemp creates 0700; give the published directory normal permissions.
        os.chmod(staging, 0o755)
        # Fails if a nonempty directory appeared meanwhile, so nothing is overwritten.
        os.rename(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest
