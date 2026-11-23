"""Rebuild the frozen Tokyo-area inputs from the official TEPCO sources."""

import csv
from datetime import date, datetime, timedelta
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast.demand_data import SplitPeriods, export_prepared_data, load_eneres_demand
from forecast.eneres_day_ahead import sha256

PERIODS = SplitPeriods(date(2022, 4, 1), date(2024, 4, 1), date(2025, 4, 1), date(2026, 4, 1))
GAP_DAY = date(2025, 6, 14)
DONOR = "eria_jukyu_202506_03.csv"
MEMBER = re.compile(r"AREA_JISEKI_(\d{8})\.csv")


def source_file(raw_dir, name, source, download):
    group = "primary" if name.endswith(".zip") else "supplement"
    path = raw_dir / group / name
    if path.is_file():
        return path
    flat = raw_dir / name
    if flat.is_file():
        return flat
    if not download:
        raise FileNotFoundError(f"Place the TEPCO source at {path}, or use --download")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / name
        print(f"Downloading {name}", flush=True)
        with urlopen(source["url"], timeout=45) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        os.replace(temporary, path)
    return path


def read_primary(path):
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            match = MEMBER.fullmatch(Path(item.filename).name)
            if not match:
                continue
            day = datetime.strptime(match[1], "%Y%m%d").date()
            if path.stem != day.strftime("AREA_%Y%m"):
                raise ValueError(f"Archive/date mismatch: {item.filename}")
            raw = archive.read(item)
            try:
                content = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                content = raw.decode("cp932")
            rows = [r for r in csv.reader(io.StringIO(content)) if r and any(r)]
            if len(rows) != 51 or rows[1][2] != match[1] or rows[2][4] != "エリア総需要量":
                raise ValueError(f"Unexpected AREA_JISEKI layout: {item.filename}")
            for slot, row in enumerate(rows[3:]):
                start = slot * 30
                end = start + 30
                start_raw, end_raw = [tuple(map(int, label.split(":"))) for label in row[2:4]]
                if (len(row) != 7 or row[0] != match[1] or int(row[1]) != slot + 1
                        or start_raw != divmod(start, 60) or end_raw not in (divmod(end, 60), divmod(end % 1440, 60))):
                    raise ValueError(f"Unexpected interval on {day}, slot {slot + 1}")
                energy = Decimal(row[4])
                if not energy.is_finite() or energy < 0 or energy != energy.to_integral_value():
                    raise ValueError(f"Invalid interval energy on {day}, slot {slot + 1}")
                yield (day, slot), int(energy)


def read_replacements(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader)[0] != "単位[MW平均]" or next(reader)[:3] != ["DATE", "TIME", "エリア需要"]:
            raise ValueError("Unexpected TEPCO supplementary CSV header")
        replacements = {}
        for row in reader:
            if not row or not any(row):
                continue
            stamp = datetime.strptime(f"{row[0]} {row[1]}", "%Y/%m/%d %H:%M")
            slot = stamp.hour * 2 + stamp.minute // 30
            if stamp.minute not in (0, 30):
                raise ValueError(f"Unaligned supplementary timestamp: {stamp}")
            if stamp.date() != GAP_DAY or slot < 10:
                continue
            # The donor is 30-minute mean MW, so energy in kWh is MW × 1000 × 0.5.
            energy = Decimal(row[2]) * 500
            if not energy.is_finite() or energy <= 0 or energy != energy.to_integral_value() or slot in replacements:
                raise ValueError(f"Invalid supplementary interval: {stamp}")
            replacements[slot] = int(energy)
    if set(replacements) != set(range(10, 48)):
        raise ValueError("Expected the 38 supplementary intervals on June 14, 2025")
    return replacements


def prepare(raw_dir, run_dir, inputs, download):
    sources = inputs["sources"]
    paths = {name: source_file(raw_dir, name, record, download) for name, record in sources.items()}
    hashes = {name: sha256(path) for name, path in paths.items()}
    demand = {}
    for name, path in paths.items():
        if name == DONOR:
            continue
        for key, value in read_primary(path):
            if key in demand:
                raise ValueError(f"Duplicate primary interval: {key}")
            demand[key] = value
    expected = {(PERIODS.start + timedelta(days=offset), slot)
                for offset in range((PERIODS.end_exclusive - PERIODS.start).days)
                for slot in range(48)}
    if demand.keys() != expected:
        raise ValueError("Missing or extra primary intervals in the four-year period")
    missing = {key for key, value in demand.items() if value == 0}
    if missing != {(GAP_DAY, slot) for slot in range(10, 48)}:
        raise ValueError("Primary zeros do not match the frozen 38-interval gap")
    for slot, energy in read_replacements(paths[DONOR]).items():
        demand[GAP_DAY, slot] = energy
    canonical = run_dir / "canonical_energy.csv"
    with canonical.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("area_name", "date", "start", "end", "demand_value", "unit"))
        for (day, slot), energy in sorted(demand.items()):
            writer.writerow((
                "TEPCO Tokyo regional demand; adopted source policy with38 official replacements",
                day.isoformat(), "%02d:%02d" % divmod(slot * 30, 60),
                "%02d:%02d" % divmod((slot + 1) * 30, 60), energy, "kWh",
            ))
    changed = [name for name, value in hashes.items() if value != sources[name]["sha256"]]
    if sha256(canonical) != inputs["canonical_sha256"]:
        raise ValueError(f"Demand series differs from the manuscript snapshot. Changed source hashes: {changed}")
    (run_dir / "raw_source_hashes.json").write_text(
        json.dumps({"sha256": hashes, "different_archive_hashes": changed}, indent=2) + "\n", encoding="utf-8"
    )
    output = run_dir / "prepared"
    export_prepared_data(
        load_eneres_demand(canonical), PERIODS, canonical, output,
        command=[sys.executable, *sys.argv], code_files=[Path(__file__)],
    )
    return output
