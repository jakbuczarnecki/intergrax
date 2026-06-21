# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Evidence posture summary contracts (HEP Band 2ae · EVID-POSTURE-01)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

EVIDENCE_POSTURE_SCHEMA_VERSION = "1.0.0"


class EvidencePostureLevel(str, Enum):
    ONBOARDING_READY = "ONBOARDING_READY"
    PARTIAL = "PARTIAL"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class EvidenceSignalStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    PASSED = "PASSED"
    FAILED = "FAILED"
    MISSING = "MISSING"
    UNKNOWN = "UNKNOWN"
    DEFERRED = "DEFERRED"
    SEPARATE = "SEPARATE"


class EvidenceSignalKind(str, Enum):
    REPO_HEALTH = "REPO_HEALTH"
    PYTEST_GATE = "PYTEST_GATE"
    CORE_CERTIFICATION = "CORE_CERTIFICATION"
    TRACE_TIMELINE = "TRACE_TIMELINE"
    W_ADAPT_L4 = "W_ADAPT_L4"
    LIVE_TIER0_PROBES = "LIVE_TIER0_PROBES"
    EVAL_REGRESSION = "EVAL_REGRESSION"
    COST_EVIDENCE = "COST_EVIDENCE"


class EvidenceBasis(str, Enum):
    DETERMINISTIC_MOCK = "DETERMINISTIC_MOCK"
    REPORT_DERIVED = "REPORT_DERIVED"
    LIVE_RUNTIME = "LIVE_RUNTIME"
    SEPARATE = "SEPARATE"
    UNKNOWN = "UNKNOWN"


class EvidencePostureArtifactKind(str, Enum):
    CORE_REPORT_JSON = "CORE_REPORT_JSON"
    CORE_REPORT_MARKDOWN = "CORE_REPORT_MARKDOWN"
    TRACE_TIMELINE_JSON = "TRACE_TIMELINE_JSON"
    TRACE_TIMELINE_MARKDOWN = "TRACE_TIMELINE_MARKDOWN"
    POSTURE_JSON = "POSTURE_JSON"
    POSTURE_MARKDOWN = "POSTURE_MARKDOWN"
    OTHER = "OTHER"


class EvidencePostureArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: EvidencePostureArtifactKind
    path: str
    description: str = ""


class EvidenceSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: EvidenceSignalKind
    status: EvidenceSignalStatus
    title: str
    message: str = ""
    basis: EvidenceBasis = EvidenceBasis.UNKNOWN
    artifact_refs: list[EvidencePostureArtifactRef] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class EvidencePostureSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = EVIDENCE_POSTURE_SCHEMA_VERSION
    posture_id: str
    level: EvidencePostureLevel
    title: str
    summary: str = ""
    signals: list[EvidenceSignal]
    artifact_refs: list[EvidencePostureArtifactRef] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


_ONBOARDING_READY_CORE_STATUSES: frozenset[EvidenceSignalStatus] = frozenset(
    {
        EvidenceSignalStatus.PASSED,
        EvidenceSignalStatus.AVAILABLE,
    }
)

_ONBOARDING_READY_TRACE_STATUSES: frozenset[EvidenceSignalStatus] = frozenset(
    {
        EvidenceSignalStatus.AVAILABLE,
        EvidenceSignalStatus.PASSED,
    }
)

_LIVE_TIER0_ALLOWED_STATUSES: frozenset[EvidenceSignalStatus] = frozenset(
    {
        EvidenceSignalStatus.DEFERRED,
        EvidenceSignalStatus.AVAILABLE,
        EvidenceSignalStatus.PASSED,
        EvidenceSignalStatus.FAILED,
        EvidenceSignalStatus.UNKNOWN,
    }
)

_POSITIVE_EVIDENCE_STATUSES: frozenset[EvidenceSignalStatus] = frozenset(
    {
        EvidenceSignalStatus.AVAILABLE,
        EvidenceSignalStatus.PASSED,
    }
)


def generate_evidence_posture_id(*, root_label: str = "local") -> str:
    """Return a deterministic posture identifier for ``root_label``."""
    return f"evidence-posture:{root_label}"


def create_evidence_signal(
    *,
    kind: EvidenceSignalKind,
    status: EvidenceSignalStatus,
    title: str,
    message: str = "",
    basis: EvidenceBasis = EvidenceBasis.UNKNOWN,
    artifact_refs: list[EvidencePostureArtifactRef] | None = None,
    metadata: dict[str, str] | None = None,
) -> EvidenceSignal:
    """Build an evidence signal with empty collections when omitted."""
    return EvidenceSignal(
        kind=kind,
        status=status,
        title=title,
        message=message,
        basis=basis,
        artifact_refs=list(artifact_refs or []),
        metadata=dict(metadata or {}),
    )


def _signal_for_kind(
    signals: list[EvidenceSignal],
    kind: EvidenceSignalKind,
) -> EvidenceSignal | None:
    for signal in signals:
        if signal.kind is kind:
            return signal
    return None


def derive_posture_level(signals: list[EvidenceSignal]) -> EvidencePostureLevel:
    """Derive overall posture level from signal statuses (no artifact I/O)."""
    core = _signal_for_kind(signals, EvidenceSignalKind.CORE_CERTIFICATION)
    trace = _signal_for_kind(signals, EvidenceSignalKind.TRACE_TIMELINE)

    if core is not None and core.status is EvidenceSignalStatus.FAILED:
        return EvidencePostureLevel.FAILED
    if trace is not None and trace.status is EvidenceSignalStatus.FAILED:
        return EvidencePostureLevel.FAILED

    if (
        core is not None
        and core.status in _ONBOARDING_READY_CORE_STATUSES
        and trace is not None
        and trace.status in _ONBOARDING_READY_TRACE_STATUSES
    ):
        return EvidencePostureLevel.ONBOARDING_READY

    if (
        core is not None
        and core.status is EvidenceSignalStatus.MISSING
    ) or (
        trace is not None
        and trace.status is EvidenceSignalStatus.MISSING
    ):
        return EvidencePostureLevel.MISSING_EVIDENCE

    has_positive = any(signal.status in _POSITIVE_EVIDENCE_STATUSES for signal in signals)
    has_onboarding_pair = (
        core is not None
        and core.status in _ONBOARDING_READY_CORE_STATUSES
        and trace is not None
        and trace.status in _ONBOARDING_READY_TRACE_STATUSES
    )
    if has_positive and not has_onboarding_pair:
        return EvidencePostureLevel.PARTIAL

    return EvidencePostureLevel.UNKNOWN


def validate_evidence_posture_summary(summary: EvidencePostureSummary) -> None:
    """Raise ``ValueError`` when ``summary`` violates EVID-POSTURE-01 contracts."""
    violations: list[str] = []

    if not summary.signals:
        violations.append("signals must not be empty")

    if not summary.posture_id.strip():
        violations.append("posture_id must not be empty")

    if not summary.title.strip():
        violations.append("title must not be empty")

    kind_counts: dict[EvidenceSignalKind, int] = {}
    for signal in summary.signals:
        kind_counts[signal.kind] = kind_counts.get(signal.kind, 0) + 1
    for kind, count in kind_counts.items():
        if count > 1:
            violations.append(f"duplicate signal kind: {kind.value}")

    for artifact_ref in summary.artifact_refs:
        if not artifact_ref.path.strip():
            violations.append("summary artifact path must not be empty")

    signals_by_kind = {
        signal.kind: signal for signal in summary.signals
    }

    for signal in summary.signals:
        for artifact_ref in signal.artifact_refs:
            if not artifact_ref.path.strip():
                violations.append(
                    f"signal artifact path must not be empty ({signal.kind.value})"
                )

        if signal.kind is EvidenceSignalKind.LIVE_TIER0_PROBES:
            if signal.status not in _LIVE_TIER0_ALLOWED_STATUSES:
                violations.append(
                    "LIVE_TIER0_PROBES status must be DEFERRED, AVAILABLE, "
                    f"PASSED, FAILED, or UNKNOWN (got {signal.status.value})"
                )

    if summary.level is EvidencePostureLevel.ONBOARDING_READY:
        core = signals_by_kind.get(EvidenceSignalKind.CORE_CERTIFICATION)
        trace = signals_by_kind.get(EvidenceSignalKind.TRACE_TIMELINE)

        if core is None:
            violations.append(
                "ONBOARDING_READY requires CORE_CERTIFICATION signal"
            )
        elif core.status not in _ONBOARDING_READY_CORE_STATUSES:
            violations.append(
                "ONBOARDING_READY requires CORE_CERTIFICATION status PASSED or "
                f"AVAILABLE (got {core.status.value})"
            )

        if trace is None:
            violations.append(
                "ONBOARDING_READY requires TRACE_TIMELINE signal"
            )
        elif trace.status not in _ONBOARDING_READY_TRACE_STATUSES:
            violations.append(
                "ONBOARDING_READY requires TRACE_TIMELINE status AVAILABLE or "
                f"PASSED (got {trace.status.value})"
            )

    if violations:
        raise ValueError("; ".join(violations))
