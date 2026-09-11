# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Predictive risk signal contracts — never diagnostic authority (PREDICTIVE R1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final
from uuid import uuid4

PREDICTIVE_SIGNAL_SCHEMA_VERSION: Final = "predictive_risk_signal.v1"

# Analyzers emit with this run id; PredictionEngine replaces before exposure.
PREDICTION_RUN_ID_ENGINE_STAMP: Final = "__prediction_run_pending__"


class PredictiveRiskScope(StrEnum):
    """Where a risk signal applies — not a Problem scope."""

    SYSTEM = "system"
    APPLICATION = "application"
    SCENARIO = "scenario"
    EXECUTION = "execution"
    COMPONENT = "component"


class PredictiveRiskSeverity(StrEnum):
    """Operator-facing risk level — not proven failure."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class PredictiveWindow:
    """Bounded forward-looking interval for a risk projection."""

    duration_seconds: int
    label: str


@dataclass(frozen=True, slots=True)
class PredictiveAnalyzerMetadata:
    """Audit identity for the analyzer that emitted a signal."""

    analyzer_id: str
    analyzer_version: str

    def __post_init__(self) -> None:
        if not self.analyzer_id.strip():
            raise ValueError("analyzer_id must be non-empty")
        if not self.analyzer_version.strip():
            raise ValueError("analyzer_version must be non-empty")


@dataclass(frozen=True, slots=True)
class PredictiveRiskSignal:
    """
    Forward-looking risk projection from evidence — never a Problem or root cause.

    PredictionEngine emits these; Diagnostic Engine remains canonical for failures.

    Forbidden payload semantics (must never appear on this contract):
    raw_exception, full_log_dump, secret_payload, customer_data_copy.
    """

    signal_id: str
    prediction_run_id: str
    tenant_id: str
    scope: PredictiveRiskScope
    subject_identity: str
    risk_type: str
    severity: PredictiveRiskSeverity
    confidence: float
    evidence_refs: tuple[str, ...]
    prediction_window: PredictiveWindow
    generated_at: datetime
    analyzer_metadata: PredictiveAnalyzerMetadata
    model_version: str
    summary: str
    recommended_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.signal_id.strip():
            raise ValueError("signal_id must be non-empty")
        if not self.prediction_run_id.strip():
            raise ValueError("prediction_run_id must be non-empty")
        if self.prediction_run_id != PREDICTION_RUN_ID_ENGINE_STAMP and not self.prediction_run_id.startswith(
            "prun_",
        ):
            raise ValueError("prediction_run_id must be prun_* or engine stamp")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if not self.subject_identity.strip():
            raise ValueError("subject_identity must be non-empty")
        if not self.risk_type.strip():
            raise ValueError("risk_type must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        if not self.evidence_refs:
            raise ValueError("evidence_refs must be non-empty")
        if self.prediction_window.duration_seconds <= 0:
            raise ValueError("prediction_window.duration_seconds must be positive")
        if not self.model_version.strip():
            raise ValueError("model_version must be non-empty")


def mint_predictive_signal_id() -> str:
    return f"prsig_{uuid4().hex}"


def mint_prediction_run_id() -> str:
    return f"prun_{uuid4().hex}"


__all__ = [
    "PREDICTIVE_SIGNAL_SCHEMA_VERSION",
    "PREDICTION_RUN_ID_ENGINE_STAMP",
    "PredictiveAnalyzerMetadata",
    "PredictiveRiskScope",
    "PredictiveRiskSeverity",
    "PredictiveRiskSignal",
    "PredictiveWindow",
    "mint_prediction_run_id",
    "mint_predictive_signal_id",
]
