import json
import os
import uuid
from datetime import datetime, timezone
from threading import Lock

PREDICTIONS_FILE = "predictions.json"

# FastAPI can serve concurrent requests, and two requests writing
# predictions.json at the same time could corrupt it or drop a record.
# This lock makes each read-modify-write atomic. It only protects a single
# process, so run uvicorn with one worker (or move to a real database).
_lock = Lock()


def load_predictions():
    try:
        with open(PREDICTIONS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_predictions(predictions):
    # Write to a temp file, then swap it in, so a crash mid-write
    # can't leave a half-written predictions.json behind.
    tmp = PREDICTIONS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(predictions, f, indent=2)
    os.replace(tmp, PREDICTIONS_FILE)


def _resolved(predictions):
    """Resolved records, oldest -> newest by when the outcome was reported."""
    rows = [r for r in predictions.values() if r["status"] == "resolved"]
    rows.sort(key=lambda r: r.get("reported_at", ""))
    return rows


def _brier(rows):
    """Mean squared error between predicted risk and the 0/1 outcome."""
    return sum((r["predicted"] - r["actual"]) ** 2 for r in rows) / len(rows)


# ---------------------------------------------------------------------------
# JOB 1: PREDICT (log every prediction as "pending")
# ---------------------------------------------------------------------------
def log_prediction(input_data: dict, predicted_risk: float, model_used: str) -> str:
    """
    Call right after /predict computes a risk score.
    Returns the record_id to hand back to the caller.
    """
    record_id = uuid.uuid4().hex[:12]
    with _lock:
        predictions = load_predictions()
        predictions[record_id] = {
            "input": input_data,
            "predicted": predicted_risk,
            "model_used": model_used,
            "status": "pending",
            "actual": None,
            "predicted_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_predictions(predictions)
    return record_id


# ---------------------------------------------------------------------------
# JOB 2: REPORT (what actually happened: fraud = 1, no fraud = 0)
# ---------------------------------------------------------------------------
def report_outcome(record_id: str, occurred: bool) -> str:
    """
    Returns "ok", "not_found", or "already_reported" so the caller can map
    these to 200 / 404 / 409.
    """
    with _lock:
        predictions = load_predictions()
        record = predictions.get(record_id)
        if record is None:
            return "not_found"
        if record["status"] == "resolved":
            return "already_reported"

        record["actual"] = 1.0 if occurred else 0.0
        record["status"] = "resolved"
        record["reported_at"] = datetime.now(timezone.utc).isoformat()
        _save_predictions(predictions)
    return "ok"


# ---------------------------------------------------------------------------
# JOB 3: CHECK (all-time accuracy)
# ---------------------------------------------------------------------------
def check_accuracy() -> dict:
    """
    Predictions are risk scores in [0, 1] and outcomes are 0/1, so the main
    metric is the Brier score (lower is better). For reference, always
    predicting the observed base rate scores base_rate * (1 - base_rate).
    Don't compare these numbers to the training PR-AUC: the metrics are different.
    """
    predictions = load_predictions()
    resolved = _resolved(predictions)
    pending = sum(1 for r in predictions.values() if r["status"] == "pending")

    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "total_predictions": len(predictions),
        "still_pending": pending,
        "resolved": len(resolved),
    }

    if not resolved:
        result.update(
            brier_score=None,
            observed_fraud_rate=None,
            mean_predicted_risk=None,
            message="No reported outcomes yet - can't compute accuracy.",
        )
        return result

    base_rate = sum(r["actual"] for r in resolved) / len(resolved)
    result.update(
        brier_score=round(_brier(resolved), 4),
        baseline_brier=round(base_rate * (1 - base_rate), 4),
        observed_fraud_rate=round(base_rate, 4),
        # If these two are far apart, the model is miscalibrated.
        mean_predicted_risk=round(
            sum(r["predicted"] for r in resolved) / len(resolved), 4
        ),
    )
    return result


# ---------------------------------------------------------------------------
# JOB 4: DRIFT DETECTION (windowed comparison)
# ---------------------------------------------------------------------------
def check_for_drift(window_size: int = 200, threshold_pct: float = 20.0) -> dict:
    """
    Compares the Brier score of the last N resolved predictions against the
    N before that. Outcomes are binary, so small windows are very noisy;
    keep window_size in the hundreds.
    """
    resolved = _resolved(load_predictions())

    if len(resolved) < 2 * window_size:
        return {
            "drift_detected": None,
            "message": f"Need {2 * window_size} resolved predictions, have {len(resolved)}",
        }

    previous = resolved[-2 * window_size : -window_size]
    recent = resolved[-window_size:]

    previous_brier = _brier(previous)
    recent_brier = _brier(recent)

    if previous_brier == 0:
        return {
            "drift_detected": None,
            "message": "Previous window Brier score is zero (can't compute % change)",
        }

    percent_change = (recent_brier - previous_brier) / previous_brier * 100

    return {
        "drift_detected": percent_change > threshold_pct,
        "percent_change": round(percent_change, 2),
        "recent_brier": round(recent_brier, 4),
        "previous_brier": round(previous_brier, 4),
        "window_size": window_size,
        "threshold_pct": threshold_pct,
    }
