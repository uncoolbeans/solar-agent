"""Offline 144-price forecast contract, separate from trading controllers."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from solaralpha.historical_pipeline import ROOT
from .features import FEATURE_FIELDS
from .full_horizon_dataset import HORIZONS
from .full_horizon_direct import OUTPUT_DIR, model_paths
from .full_horizon_tcn import load_saved as load_tcn
from .multi_horizon import LATEST_INDEX
from .sequence_dataset import WINDOW


@dataclass(frozen=True)
class ForecastRequest:
    decision_start_utc: datetime
    history_decision_starts_utc: tuple[datetime, ...]
    feature_history: np.ndarray
    feature_order: tuple[str, ...]
    latest_completed_interval_end_utc: datetime
    max_reading_age_minutes: float = 5.0


@dataclass(frozen=True)
class ForecastResponse:
    decision_start_utc: datetime
    target_interval_starts_utc: tuple[datetime, ...]
    predicted_wholesale_aud_per_kwh: tuple[float, ...]
    model_identifier: str
    input_reading_age_minutes: float
    input_valid: bool
    input_status: str
    unit: str = "AUD/kWh"


def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("Forecast timestamps must be timezone-aware UTC")
    return value.astimezone(timezone.utc)


def validate_request(request):
    t = _utc(request.decision_start_utc)
    if t.second or t.microsecond or t.minute % 5:
        raise ValueError("Decision is not on a five-minute UTC grid")
    times = tuple(_utc(value) for value in request.history_decision_starts_utc)
    if len(times) != WINDOW or times[-1] != t or any(times[i] - times[i-1] != timedelta(minutes=5)
                                                   for i in range(1, len(times))):
        raise ValueError("Feature history must be 72 chronological five-minute decision rows ending at t")
    matrix = np.asarray(request.feature_history)
    if (tuple(request.feature_order) != FEATURE_FIELDS or matrix.shape != (WINDOW, len(FEATURE_FIELDS)) or
            not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all()):
        raise ValueError("Feature order, history shape or finite values differ")
    latest = _utc(request.latest_completed_interval_end_utc)
    age = (t - latest).total_seconds() / 60
    if age < 0 or not math.isfinite(request.max_reading_age_minutes) or request.max_reading_age_minutes < 0:
        raise ValueError("Future reading or invalid freshness limit")
    return matrix.astype(np.float32), age


def assemble_response(request, model_identifier, predicted_prices):
    _, age = validate_request(request)
    price = np.asarray(predicted_prices, dtype=np.float64)
    if price.shape != (len(HORIZONS),) or not np.isfinite(price).all():
        raise ValueError("Forecast must contain 144 finite AUD/kWh prices")
    t = _utc(request.decision_start_utc)
    valid = age <= request.max_reading_age_minutes
    return ForecastResponse(t, tuple(t + timedelta(minutes=h) for h in HORIZONS),
                            tuple(float(value) for value in price), model_identifier,
                            age, valid, "fresh" if valid else "stale")


class FullHorizonForecaster:
    """Load saved models only; ordinary inference never opens realised future data."""

    def __init__(self, model_identifier, output_dir=OUTPUT_DIR):
        if model_identifier not in ("direct", "tcn", "hybrid_tuned", "hybrid_fixed"):
            raise ValueError("Unknown full-horizon model identifier")
        self.identifier = model_identifier
        self.output_dir = output_dir
        self.models = None
        self.tcn = None
        self.meta = None
        self.choices = None
        if model_identifier == "tcn":
            self.tcn, self.meta = load_tcn(output_dir / "tcn")
        else:
            import xgboost as xgb
            self.models = []
            for h in HORIZONS:
                path, metadata_path = model_paths(h, output_dir)
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (meta["horizon_minutes"] != h or tuple(meta["feature_order"]) != FEATURE_FIELDS or
                        sha256(path.read_bytes()).hexdigest() != meta["model_sha256"]):
                    raise ValueError("Saved direct model identity differs")
                model = xgb.XGBRegressor()
                model.load_model(path)
                self.models.append(model)
            if model_identifier.startswith("hybrid"):
                selection = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
                key = "choices" if model_identifier == "hybrid_tuned" else "fixed_choices"
                self.choices = tuple(selection[key])
                if (tuple(selection["horizons_minutes"]) != HORIZONS or
                        len(self.choices) != len(HORIZONS) or
                        any(choice not in ("direct", "persistence") for choice in self.choices)):
                    raise ValueError("Saved hybrid choice vector differs")

    def forecast(self, request):
        matrix, _ = validate_request(request)
        if self.tcn is not None:
            import torch
            scaled = ((matrix - np.asarray(self.meta["scaler_mean"])) /
                      np.asarray(self.meta["scaler_scale"])).astype(np.float32)
            with torch.inference_mode():
                prices = self.tcn(torch.from_numpy(scaled[None])).numpy()[0]
        else:
            row = matrix[-1:]
            latest = float(row[0, LATEST_INDEX])
            prices = np.asarray([
                latest if self.choices is not None and self.choices[i] == "persistence"
                else model.predict(row)[0] for i, model in enumerate(self.models)])
        return assemble_response(request, self.identifier, prices)
