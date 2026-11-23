"""Fixed-model, daily 48-slot forecasts from verified private ENERES inputs.

Every origin is 00:00 JST. Only observations strictly before that origin enter
features. Observations from earlier evaluation days are available as history;
weights and scaling remain fixed throughout validation and evaluation.
"""

import calendar
import csv
import hashlib
from importlib.metadata import distributions
import json
import math
import platform
import random
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, time, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from forecast.demand_data import (
    DemandDataset, DemandRecord, HALF_HOUR, JST, SplitPeriods, _code_record, split_demand,
)

SLOTS_PER_DAY = 48
WEEK_SLOTS = 336
FEATURE_NAMES = (
    "demand_scaled", "lag_48_scaled", "mean_48_scaled", "mean_336_scaled",
    "std_336_divided_by_train_range", "hour_sin", "hour_cos", "dow_sin",
    "dow_cos", "doy_sin", "doy_cos",
)


@dataclass(frozen=True)
class TrainingConfig:
    lookback: int = 96
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.1
    epochs: int = 100
    patience: int = 12
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42
    threads: int = 2

    def __post_init__(self):
        for name in ("lookback", "hidden", "layers", "epochs", "patience", "batch_size", "threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.lookback % SLOTS_PER_DAY:
            raise ValueError("lookback must be a whole number of days")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")


@dataclass(frozen=True)
class DemandScaler:
    minimum_kw: float
    maximum_kw: float
    span_kw: float

    @classmethod
    def fit(cls, train_values):
        values = np.asarray(train_values, dtype=np.float64)
        if not values.size or not np.isfinite(values).all():
            raise ValueError("Scaler requires finite nonempty training data")
        low, high = float(values.min()), float(values.max())
        return cls(low, high, high - low if high > low else 1.0)

    def transform(self, values):
        # Deliberately do not clip values outside the training range.
        return (values - self.minimum_kw) / self.span_kw

    def inverse(self, values):
        return values * self.span_kw + self.minimum_kw


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prepared(directory: Path):
    """Check all four CSV hashes, timestamps, values and split consistency."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("Use prepared data with manifest schema_version=2")
    periods = SplitPeriods(**{
        name: datetime.fromisoformat(value).date() for name, value in manifest["periods"].items()
    })
    csv_rows = {}
    for name in ("demand", "train", "validation", "evaluation"):
        info = manifest["outputs"][name]
        path = directory / f"{name}.csv"
        if info["file"] != path.name or sha256(path) != info["sha256"]:
            raise ValueError(f"Prepared CSV hash/name mismatch: {name}")
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["timestamp", "demand_kw"]:
                raise ValueError(f"Unexpected prepared columns: {name}")
            rows = tuple(reader)
        if len(rows) != info["rows"]:
            raise ValueError(f"Prepared row count mismatch: {name}")
        csv_rows[name] = rows
    records = []
    for row in csv_rows["demand"]:
        stamp = datetime.fromisoformat(row["timestamp"])
        value = float(row["demand_kw"])
        if stamp.utcoffset() != datetime(2024, 1, 1, tzinfo=JST).utcoffset():
            raise ValueError("Prepared timestamps must have the JST offset")
        if not math.isfinite(value) or value < 0:
            raise ValueError("Prepared demand must be finite and nonnegative")
        if stamp.second or stamp.microsecond or stamp.minute not in (0, 30):
            raise ValueError("Prepared timestamps must be half-hour aligned")
        records.append(DemandRecord(stamp, value))
    if not records:
        raise ValueError("Prepared demand is empty")
    for a, b in zip(records, records[1:]):
        if b.timestamp - a.timestamp != HALF_HOUR:
            raise ValueError("Prepared demand has a missing or duplicated interval")
    dataset = DemandDataset(tuple(records), manifest["source"]["area_name"], manifest["source"]["sha256"])
    splits = split_demand(dataset, periods)
    expected = tuple(csv_rows["train"] + csv_rows["validation"] + csv_rows["evaluation"])
    if expected != csv_rows["demand"]:
        raise ValueError("Prepared split CSVs differ from the full history")
    for name, rows in splits.items():
        if len(rows) != len(csv_rows[name]) or rows[0].timestamp.isoformat() != csv_rows[name][0]["timestamp"]:
            raise ValueError(f"Prepared split boundary mismatch: {name}")
    return dataset, periods, manifest


def build_features(dataset: DemandDataset, scaler: DemandScaler) -> np.ndarray:
    """Causal trailing means/std; no interpolation or backward filling."""
    values = np.asarray([r.demand_kw for r in dataset.records], dtype=np.float64)
    scaled = scaler.transform(values)
    lag = np.full(len(values), np.nan)
    lag[48:] = scaled[:-48]

    def rolling(window, squared=False):
        data = scaled * scaled if squared else scaled
        sums = np.concatenate(([0.0], np.cumsum(data)))
        result = np.full(len(data), np.nan)
        result[window - 1:] = (sums[window:] - sums[:-window]) / window
        return result

    mean_48, mean_336 = rolling(48), rolling(WEEK_SLOTS)
    std_336 = np.sqrt(np.maximum(rolling(WEEK_SLOTS, squared=True) - mean_336**2, 0))
    stamps = [r.timestamp for r in dataset.records]
    hour_phase = np.asarray([(s.hour + s.minute / 60) / 24 for s in stamps]) * 2 * np.pi
    dow_phase = np.asarray([s.weekday() / 7 for s in stamps]) * 2 * np.pi
    doy_phase = np.asarray([
        (s.timetuple().tm_yday - 1) / (366 if calendar.isleap(s.year) else 365) for s in stamps
    ]) * 2 * np.pi
    return np.column_stack((
        scaled, lag, mean_48, mean_336, std_336,
        np.sin(hour_phase), np.cos(hour_phase), np.sin(dow_phase), np.cos(dow_phase),
        np.sin(doy_phase), np.cos(doy_phase),
    )).astype(np.float32)


@dataclass(frozen=True)
class DailyWindows:
    inputs: np.ndarray
    targets: np.ndarray
    starts: np.ndarray


def make_daily_windows(dataset, features, scaler, lookback, start_day, end_day):
    values = np.asarray([r.demand_kw for r in dataset.records], dtype=np.float64)
    inputs, targets, starts = [], [], []
    for index, record in enumerate(dataset.records):
        stamp = record.timestamp
        if stamp.time() != time() or not start_day <= stamp.date() < end_day:
            continue
        if index < lookback or index + SLOTS_PER_DAY > len(values):
            continue
        window = features[index - lookback:index]
        if not np.isfinite(window).all():
            continue  # First training days lack a complete trailing week.
        inputs.append(window)
        targets.append(scaler.transform(values[index:index + SLOTS_PER_DAY]))
        starts.append(index)
    if not starts:
        raise ValueError("No complete daily windows for the requested period")
    return DailyWindows(np.stack(inputs), np.stack(targets).astype(np.float32), np.asarray(starts))


class EneresLSTM(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            len(FEATURE_NAMES), config.hidden, num_layers=config.layers,
            batch_first=True, dropout=config.dropout if config.layers > 1 else 0,
        )
        self.head = nn.Linear(config.hidden, SLOTS_PER_DAY)

    def forward(self, inputs):
        sequence, _ = self.lstm(inputs)
        return self.head(sequence[:, -1])


def train_model(train: DailyWindows, validation: DailyWindows, config: TrainingConfig, progress=None):
    """Select a checkpoint on validation MSE; evaluation data never enter."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    model = EneresLSTM(config).cpu()
    generator = torch.Generator().manual_seed(config.seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(train.inputs), torch.from_numpy(train.targets)),
        batch_size=config.batch_size, shuffle=True, generator=generator,
    )
    vx, vy = torch.from_numpy(validation.inputs), torch.from_numpy(validation.targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    history, best_state, best_epoch, best_loss = [], None, 0, math.inf
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(x)
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(vx), vy))
        if not math.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss")
        row = {"epoch": epoch, "train_mse_scaled": total / len(train.inputs),
               "validation_mse_scaled": validation_loss}
        history.append(row)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if progress is not None:
            progress(row)
        if epoch - best_epoch >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def predict_kw(model, windows: DailyWindows, scaler: DemandScaler, batch_size=32):
    result = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(windows.inputs), batch_size):
            prediction = model(torch.from_numpy(windows.inputs[offset:offset + batch_size])).numpy()
            result.append(scaler.inverse(prediction.astype(np.float64)))
    predicted = np.maximum(np.concatenate(result), 0.0)
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite forecast")
    return predicted


def error_metrics(actual, forecast):
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    error = forecast - actual
    nonzero = actual > 0
    return {
        "mae_kw": float(np.mean(np.abs(error))),
        "rmse_kw": float(np.sqrt(np.mean(error**2))),
        "wape_percent": float(np.sum(np.abs(error)) / np.sum(actual) * 100) if np.sum(actual) else None,
        "mape_nonzero_percent": float(np.mean(np.abs(error[nonzero]) / actual[nonzero]) * 100) if nonzero.any() else None,
        "mape_excluded_zero_slots": int(np.count_nonzero(~nonzero)),
    }


def write_predictions(output, split, dataset, windows, forecast):
    values = np.asarray([r.demand_kw for r in dataset.records])
    actual = np.stack([values[i:i + 48] for i in windows.starts])
    naive = np.stack([values[i - WEEK_SLOTS:i - WEEK_SLOTS + 48] for i in windows.starts])
    with (output / f"{split}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["origin_timestamp", "timestamp", "actual_demand_kw", "lstm_forecast_kw", "seasonal_naive_kw"])
        for day, start in enumerate(windows.starts):
            origin = dataset.records[start].timestamp.isoformat()
            for slot in range(48):
                writer.writerow([origin, dataset.records[start + slot].timestamp.isoformat(),
                                 actual[day, slot], forecast[day, slot], naive[day, slot]])
    daily = []
    summary = {}
    for name, pred in (("lstm", forecast), ("seasonal_naive", naive)):
        peak_time_errors = (np.argmax(pred, axis=1) - np.argmax(actual, axis=1)) * 30
        peak_errors = np.max(pred, axis=1) - np.max(actual, axis=1)
        summary[name] = {
            **error_metrics(actual, pred), "days": len(windows.starts), "slots": int(actual.size),
            "mean_absolute_peak_time_error_minutes": float(np.mean(np.abs(peak_time_errors))),
            "mean_absolute_peak_magnitude_error_kw": float(np.mean(np.abs(peak_errors))),
        }
        for day, start in enumerate(windows.starts):
            daily.append({
                "date": dataset.records[start].timestamp.date().isoformat(), "model": name,
                **error_metrics(actual[day], pred[day]),
                "peak_time_error_minutes": int(peak_time_errors[day]),
                "peak_magnitude_error_kw": float(peak_errors[day]),
            })
    with (output / f"{split}_daily_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(daily[0]))
        writer.writeheader()
        writer.writerows(daily)
    return summary


def run_experiment(prepared_dir: Path, output_dir: Path, config: TrainingConfig, command=None, progress=None):
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    dataset, periods, prepared_manifest = load_prepared(prepared_dir)
    train_values = np.asarray([r.demand_kw for r in dataset.records if r.timestamp.date() < periods.validation_start])
    scaler = DemandScaler.fit(train_values)
    features = build_features(dataset, scaler)
    windows = {
        "train": make_daily_windows(dataset, features, scaler, config.lookback, periods.start, periods.validation_start),
        "validation": make_daily_windows(dataset, features, scaler, config.lookback, periods.validation_start, periods.evaluation_start),
        "evaluation": make_daily_windows(dataset, features, scaler, config.lookback, periods.evaluation_start, periods.end_exclusive),
    }
    for name, start, end in (("validation", periods.validation_start, periods.evaluation_start),
                             ("evaluation", periods.evaluation_start, periods.end_exclusive)):
        if len(windows[name].starts) != (end - start).days:
            raise ValueError(f"Missing forecast days in {name}")
    repo = Path(__file__).resolve().parents[2]
    code_paths = [Path(__file__), repo / "src/forecast/demand_data.py", repo / "scripts/train_eneres_forecast.py"]
    code_before = _code_record(repo, code_paths)
    prepared_hash = sha256(prepared_dir / "manifest.json")
    model, history, best_epoch = train_model(windows["train"], windows["validation"], config, progress)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        def write_json(name, content):
            (staging / name).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_json("config.json", asdict(config))
        environment = sorted({f"{dist.metadata['Name']}=={dist.version}" for dist in distributions()})
        (staging / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")
        write_json("scaler.json", {**asdict(scaler), "fitted_period_end_exclusive": periods.validation_start.isoformat(),
                                   "fitted_rows": len(train_values), "clip_scaled_values": False})
        torch.save({"state_dict": model.state_dict(), "config": asdict(config), "feature_names": list(FEATURE_NAMES),
                    "scaler": asdict(scaler), "best_epoch": best_epoch}, staging / "model.pt")
        with (staging / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        metrics = {}
        for name in ("validation", "evaluation"):
            forecast = predict_kw(model, windows[name], scaler, config.batch_size)
            metrics[name] = write_predictions(staging, name, dataset, windows[name], forecast)
        write_json("metrics.json", metrics)
        # Reject concurrent edits rather than attributing a run to different code/input.
        code_after = _code_record(repo, code_paths)
        if code_after != code_before or sha256(prepared_dir / "manifest.json") != prepared_hash:
            raise ValueError("Code or prepared manifest changed during the run")
        load_prepared(prepared_dir)
        manifest = {
            "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": list(command) if command is not None else None, "code": code_before,
            "prepared_manifest": {"path": str((prepared_dir / "manifest.json").resolve()), "sha256": prepared_hash},
            "source": prepared_manifest["source"], "periods": prepared_manifest["periods"],
            "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                        "platform": platform.platform(), "device": "cpu", "threads": config.threads,
                        "deterministic_algorithms": True},
            "protocol": {
                "origin": "00:00 Asia/Tokyo; assumes observations through previous 23:30 interval available without reporting delay",
                "horizon_slots": 48, "training_stride_slots": 48, "history_features": list(FEATURE_NAMES),
                "trailing_std_ddof": 0, "future_demand_features": False,
                "scaler_fit": "train only; frozen for validation and evaluation",
                "model_selection": "minimum validation MSE on scaled, unclipped predictions; no evaluation tuning",
                "refit_on_validation": False, "history_during_evaluation": "observed preceding days; fixed weights",
                "forecast_postprocessing": "clip negative inverse-scaled forecasts to zero",
                "seasonal_naive": "same 48-slot profile seven days before each forecast origin",
                "peak_ties": "first maximum; timing difference is signed linear within-day minutes, without wrapping",
            },
            "training": {"best_epoch": best_epoch, "epochs_run": len(history),
                         "daily_windows": {name: len(w.starts) for name, w in windows.items()},
                         "first_train_target": dataset.records[windows["train"].starts[0]].timestamp.isoformat()},
            "outputs": {p.name: {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p in sorted(staging.iterdir())},
        }
        write_json("run_manifest.json", manifest)
        # Reserve destination exclusively, including when another process creates an empty directory.
        output_dir.mkdir()
        try:
            for path in staging.iterdir():
                path.rename(output_dir / path.name)
        except BaseException:
            shutil.rmtree(output_dir)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest, metrics
