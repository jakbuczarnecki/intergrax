# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Live core probe contracts (EVID-CORE-FU-01 · EVID-CORE-FU-01A)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

LIVE_CORE_PROBE_SCHEMA_VERSION = "1.0.0"

LIVE_CORE_PROBE_CATALOG_ORDER = (
    "basic_run_completed_live",
    "trace_persisted_live",
    "tool_denied_by_policy_live",
)


class LiveCoreProbeKind(str, Enum):
    BASIC_RUN_COMPLETED_LIVE = "basic_run_completed_live"
    TRACE_PERSISTED_LIVE = "trace_persisted_live"
    TOOL_DENIED_BY_POLICY_LIVE = "tool_denied_by_policy_live"


class LiveCoreProbeStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    UNAVAILABLE = "UNAVAILABLE"


class LiveCoreProbeEvidenceBasis(str, Enum):
    LIVE_RUNTIME = "LIVE_RUNTIME"
    MOCK_LLM = "MOCK_LLM"
    LOCAL_NO_NETWORK = "LOCAL_NO_NETWORK"
    UNKNOWN = "UNKNOWN"


class LiveCoreProbeArtifactKind(str, Enum):
    LIVE_CORE_REPORT_JSON = "LIVE_CORE_REPORT_JSON"
    LIVE_CORE_REPORT_MARKDOWN = "LIVE_CORE_REPORT_MARKDOWN"
    TRACE_RECORD = "TRACE_RECORD"
    RUNTIME_EVENT_LOG = "RUNTIME_EVENT_LOG"
    OTHER = "OTHER"


class LiveCoreProbeArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: LiveCoreProbeArtifactKind
    path: str
    description: str = ""


class LiveCoreProbeExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe_kind: LiveCoreProbeKind
    description: str
    requires_runtime_path: bool = True
    requires_mock_llm: bool = True
    requires_no_network: bool = True
    requires_no_provider_calls: bool = True


class LiveCoreProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe_kind: LiveCoreProbeKind
    status: LiveCoreProbeStatus
    title: str
    message: str = ""
    evidence_basis: list[LiveCoreProbeEvidenceBasis]
    artifact_refs: list[LiveCoreProbeArtifactRef] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class LiveCoreProbeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = LIVE_CORE_PROBE_SCHEMA_VERSION
    report_id: str
    title: str = "Selected live Tier-0 probes"
    summary: str = ""
    status: LiveCoreProbeStatus
    results: list[LiveCoreProbeResult]
    artifact_refs: list[LiveCoreProbeArtifactRef] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


_REQUIRED_EVIDENCE_BASIS: frozenset[LiveCoreProbeEvidenceBasis] = frozenset(
    {
        LiveCoreProbeEvidenceBasis.LIVE_RUNTIME,
        LiveCoreProbeEvidenceBasis.MOCK_LLM,
        LiveCoreProbeEvidenceBasis.LOCAL_NO_NETWORK,
    }
)

_ACTIVE_PROBE_KINDS: frozenset[LiveCoreProbeKind] = frozenset(
    LiveCoreProbeKind(probe_id) for probe_id in LIVE_CORE_PROBE_CATALOG_ORDER
)


def generate_live_core_probe_report_id(*, root_label: str = "local") -> str:
    """Return a deterministic live core probe report identifier."""
    return f"live-core-probes:{root_label}"


def live_core_probe_kinds() -> tuple[LiveCoreProbeKind, ...]:
    """Return active probe kinds in canonical catalog order."""
    return tuple(LiveCoreProbeKind(probe_id) for probe_id in LIVE_CORE_PROBE_CATALOG_ORDER)


def create_live_core_probe_result(
    *,
    probe_kind: LiveCoreProbeKind,
    status: LiveCoreProbeStatus,
    title: str,
    message: str = "",
    evidence_basis: list[LiveCoreProbeEvidenceBasis] | None = None,
    artifact_refs: list[LiveCoreProbeArtifactRef] | None = None,
    metadata: dict[str, str] | None = None,
) -> LiveCoreProbeResult:
    """Build a live core probe result with default evidence basis when omitted."""
    return LiveCoreProbeResult(
        probe_kind=probe_kind,
        status=status,
        title=title,
        message=message,
        evidence_basis=list(
            evidence_basis
            or [
                LiveCoreProbeEvidenceBasis.LIVE_RUNTIME,
                LiveCoreProbeEvidenceBasis.MOCK_LLM,
                LiveCoreProbeEvidenceBasis.LOCAL_NO_NETWORK,
            ]
        ),
        artifact_refs=list(artifact_refs or []),
        metadata=dict(metadata or {}),
    )


def derive_live_core_probe_report_status(
    results: list[LiveCoreProbeResult],
) -> LiveCoreProbeStatus:
    """Derive overall report status from probe results."""
    if not results:
        return LiveCoreProbeStatus.UNAVAILABLE

    statuses = {result.status for result in results}

    if LiveCoreProbeStatus.FAILED in statuses:
        return LiveCoreProbeStatus.FAILED

    if all(result.status is LiveCoreProbeStatus.PASSED for result in results):
        return LiveCoreProbeStatus.PASSED

    if LiveCoreProbeStatus.PASSED in statuses and (
        LiveCoreProbeStatus.SKIPPED in statuses
        or LiveCoreProbeStatus.UNAVAILABLE in statuses
    ):
        return LiveCoreProbeStatus.SKIPPED

    return LiveCoreProbeStatus.UNAVAILABLE


def validate_live_core_probe_report(report: LiveCoreProbeReport) -> None:
    """Raise ``ValueError`` when ``report`` violates live core probe contracts."""
    violations: list[str] = []

    if not report.report_id.strip():
        violations.append("report_id must not be empty")

    if not report.title.strip():
        violations.append("title must not be empty")

    if not report.results:
        violations.append("results must not be empty")

    kind_counts: dict[LiveCoreProbeKind, int] = {}
    for result in report.results:
        kind_counts[result.probe_kind] = kind_counts.get(result.probe_kind, 0) + 1
        if result.probe_kind not in _ACTIVE_PROBE_KINDS:
            violations.append(
                f"unknown probe kind outside active catalog: {result.probe_kind.value}"
            )

    for kind, count in kind_counts.items():
        if count > 1:
            violations.append(f"duplicate probe kind: {kind.value}")

    for artifact_ref in report.artifact_refs:
        if not artifact_ref.path.strip():
            violations.append("report artifact path must not be empty")

    for result in report.results:
        if not result.title.strip():
            violations.append(f"result title must not be empty ({result.probe_kind.value})")

        if not result.evidence_basis:
            violations.append(
                f"result evidence_basis must not be empty ({result.probe_kind.value})"
            )
        else:
            basis_set = frozenset(result.evidence_basis)
            for required in _REQUIRED_EVIDENCE_BASIS:
                if required not in basis_set:
                    violations.append(
                        f"result must include {required.value} "
                        f"({result.probe_kind.value})"
                    )

        for artifact_ref in result.artifact_refs:
            if not artifact_ref.path.strip():
                violations.append(
                    f"result artifact path must not be empty ({result.probe_kind.value})"
                )

    expected_status = derive_live_core_probe_report_status(report.results)
    if report.status is not expected_status:
        violations.append(
            f"report status must be {expected_status.value} "
            f"(got {report.status.value})"
        )

    if violations:
        raise ValueError("; ".join(violations))
