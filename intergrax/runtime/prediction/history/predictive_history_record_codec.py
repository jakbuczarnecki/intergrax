# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""JSON codec for prediction history document rows (PREDICTIVE R2)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from intergrax.contracts.predictive_history import (
    PREDICTIVE_HISTORY_SCHEMA_VERSION,
    PredictiveHistoryOutcomeStatus,
    PredictiveRiskHistoryRecord,
)
from intergrax.contracts.predictive_risk import PredictiveWindow


def encode_predictive_history_record(record: PredictiveRiskHistoryRecord) -> dict[str, Any]:
    return {
        "schema": PREDICTIVE_HISTORY_SCHEMA_VERSION,
        "prediction_run_id": record.prediction_run_id,
        "risk_signal_id": record.risk_signal_id,
        "tenant_id": record.tenant_id,
        "analyzer_id": record.analyzer_id,
        "analyzer_version": record.analyzer_version,
        "generated_at": record.generated_at.isoformat(),
        "prediction_window": {
            "duration_seconds": record.prediction_window.duration_seconds,
            "label": record.prediction_window.label,
        },
        "confidence": record.confidence,
        "evidence_refs": list(record.evidence_refs),
        "outcome_status": record.outcome_status.value,
        "subject_identity": record.subject_identity,
        "risk_type": record.risk_type,
        "summary": record.summary,
    }


def decode_predictive_history_record(payload: dict[str, Any]) -> PredictiveRiskHistoryRecord:
    window_raw = payload["prediction_window"]
    return PredictiveRiskHistoryRecord(
        prediction_run_id=str(payload["prediction_run_id"]),
        risk_signal_id=str(payload["risk_signal_id"]),
        tenant_id=str(payload["tenant_id"]),
        analyzer_id=str(payload["analyzer_id"]),
        analyzer_version=str(payload["analyzer_version"]),
        generated_at=datetime.fromisoformat(str(payload["generated_at"])),
        prediction_window=PredictiveWindow(
            duration_seconds=int(window_raw["duration_seconds"]),
            label=str(window_raw["label"]),
        ),
        confidence=float(payload["confidence"]),
        evidence_refs=tuple(str(x) for x in payload["evidence_refs"]),
        outcome_status=PredictiveHistoryOutcomeStatus(str(payload["outcome_status"])),
        subject_identity=str(payload["subject_identity"]),
        risk_type=str(payload["risk_type"]),
        summary=str(payload.get("summary") or ""),
    )


def encode_predictive_history_json(record: PredictiveRiskHistoryRecord) -> str:
    return json.dumps(encode_predictive_history_record(record), sort_keys=True)


def decode_predictive_history_json(raw: str) -> PredictiveRiskHistoryRecord:
    return decode_predictive_history_record(json.loads(raw))


__all__ = [
    "decode_predictive_history_json",
    "decode_predictive_history_record",
    "encode_predictive_history_json",
    "encode_predictive_history_record",
]
