# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.evidence.evidence_posture_contracts import (
    EvidenceBasis,
    EvidencePostureArtifactKind,
    EvidencePostureArtifactRef,
    EvidencePostureLevel,
    EvidencePostureSummary,
    EvidenceSignal,
    EvidenceSignalKind,
    EvidenceSignalStatus,
    create_evidence_signal,
    derive_posture_level,
    generate_evidence_posture_id,
    validate_evidence_posture_summary,
)

pytestmark = pytest.mark.unit

_EVIDENCE_ROOT = Path(__file__).resolve().parents[3] / "intergrax" / "runtime" / "evidence"


def _core_signal(
    *,
    status: EvidenceSignalStatus = EvidenceSignalStatus.PASSED,
) -> EvidenceSignal:
    return create_evidence_signal(
        kind=EvidenceSignalKind.CORE_CERTIFICATION,
        status=status,
        title="Core certification",
        basis=EvidenceBasis.DETERMINISTIC_MOCK,
    )


def _trace_signal(
    *,
    status: EvidenceSignalStatus = EvidenceSignalStatus.AVAILABLE,
) -> EvidenceSignal:
    return create_evidence_signal(
        kind=EvidenceSignalKind.TRACE_TIMELINE,
        status=status,
        title="Trace timeline",
        basis=EvidenceBasis.REPORT_DERIVED,
    )


def _onboarding_ready_summary(
    *,
    core_status: EvidenceSignalStatus = EvidenceSignalStatus.PASSED,
    trace_status: EvidenceSignalStatus = EvidenceSignalStatus.AVAILABLE,
) -> EvidencePostureSummary:
    return EvidencePostureSummary(
        posture_id=generate_evidence_posture_id(root_label="local"),
        level=EvidencePostureLevel.ONBOARDING_READY,
        title="Harness evidence posture",
        signals=[
            _core_signal(status=core_status),
            _trace_signal(status=trace_status),
        ],
    )


def test_generate_evidence_posture_id_is_deterministic() -> None:
    first = generate_evidence_posture_id(root_label="local")
    second = generate_evidence_posture_id(root_label="local")
    assert first == second == "evidence-posture:local"


def test_create_evidence_signal_uses_empty_defaults() -> None:
    signal = create_evidence_signal(
        kind=EvidenceSignalKind.PYTEST_GATE,
        status=EvidenceSignalStatus.UNKNOWN,
        title="Pytest gate",
    )
    assert signal.artifact_refs == []
    assert signal.metadata == {}


def test_validate_evidence_posture_summary_minimal_onboarding_ready_passes() -> None:
    validate_evidence_posture_summary(_onboarding_ready_summary())


def test_validate_evidence_posture_summary_rejects_empty_signals() -> None:
    summary = EvidencePostureSummary(
        posture_id="evidence-posture:local",
        level=EvidencePostureLevel.UNKNOWN,
        title="Harness evidence posture",
        signals=[],
    )
    with pytest.raises(ValueError, match="signals must not be empty"):
        validate_evidence_posture_summary(summary)


def test_validate_evidence_posture_summary_rejects_duplicate_signal_kind() -> None:
    summary = EvidencePostureSummary(
        posture_id="evidence-posture:local",
        level=EvidencePostureLevel.PARTIAL,
        title="Harness evidence posture",
        signals=[
            _core_signal(),
            _core_signal(status=EvidenceSignalStatus.AVAILABLE),
        ],
    )
    with pytest.raises(ValueError, match="duplicate signal kind: CORE_CERTIFICATION"):
        validate_evidence_posture_summary(summary)


def test_validate_evidence_posture_summary_rejects_empty_posture_id() -> None:
    summary = _onboarding_ready_summary()
    summary = summary.model_copy(update={"posture_id": "   "})
    with pytest.raises(ValueError, match="posture_id must not be empty"):
        validate_evidence_posture_summary(summary)


def test_validate_evidence_posture_summary_rejects_empty_title() -> None:
    summary = _onboarding_ready_summary()
    summary = summary.model_copy(update={"title": ""})
    with pytest.raises(ValueError, match="title must not be empty"):
        validate_evidence_posture_summary(summary)


def test_validate_evidence_posture_summary_rejects_empty_summary_artifact_path() -> None:
    summary = _onboarding_ready_summary()
    summary = summary.model_copy(
        update={
            "artifact_refs": [
                EvidencePostureArtifactRef(
                    kind=EvidencePostureArtifactKind.POSTURE_JSON,
                    path="   ",
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="summary artifact path must not be empty"):
        validate_evidence_posture_summary(summary)


def test_validate_evidence_posture_summary_rejects_empty_signal_artifact_path() -> None:
    core = _core_signal()
    core = core.model_copy(
        update={
            "artifact_refs": [
                EvidencePostureArtifactRef(
                    kind=EvidencePostureArtifactKind.CORE_REPORT_JSON,
                    path="",
                )
            ]
        }
    )
    summary = EvidencePostureSummary(
        posture_id="evidence-posture:local",
        level=EvidencePostureLevel.PARTIAL,
        title="Harness evidence posture",
        signals=[core, _trace_signal()],
    )
    with pytest.raises(
        ValueError,
        match="signal artifact path must not be empty \\(CORE_CERTIFICATION\\)",
    ):
        validate_evidence_posture_summary(summary)


def test_validate_evidence_posture_summary_onboarding_ready_requires_core_signal() -> None:
    summary = EvidencePostureSummary(
        posture_id="evidence-posture:local",
        level=EvidencePostureLevel.ONBOARDING_READY,
        title="Harness evidence posture",
        signals=[_trace_signal()],
    )
    with pytest.raises(
        ValueError,
        match="ONBOARDING_READY requires CORE_CERTIFICATION signal",
    ):
        validate_evidence_posture_summary(summary)


def test_validate_evidence_posture_summary_onboarding_ready_requires_trace_signal() -> None:
    summary = EvidencePostureSummary(
        posture_id="evidence-posture:local",
        level=EvidencePostureLevel.ONBOARDING_READY,
        title="Harness evidence posture",
        signals=[_core_signal()],
    )
    with pytest.raises(
        ValueError,
        match="ONBOARDING_READY requires TRACE_TIMELINE signal",
    ):
        validate_evidence_posture_summary(summary)


def test_validate_evidence_posture_summary_onboarding_ready_rejects_failed_core() -> None:
    with pytest.raises(
        ValueError,
        match="ONBOARDING_READY requires CORE_CERTIFICATION status PASSED or AVAILABLE",
    ):
        validate_evidence_posture_summary(
            _onboarding_ready_summary(core_status=EvidenceSignalStatus.FAILED)
        )


def test_derive_posture_level_onboarding_ready_for_core_passed_and_trace_available() -> None:
    level = derive_posture_level(
        [
            _core_signal(status=EvidenceSignalStatus.PASSED),
            _trace_signal(status=EvidenceSignalStatus.AVAILABLE),
        ]
    )
    assert level is EvidencePostureLevel.ONBOARDING_READY


def test_derive_posture_level_failed_for_core_failed() -> None:
    level = derive_posture_level([_core_signal(status=EvidenceSignalStatus.FAILED)])
    assert level is EvidencePostureLevel.FAILED


def test_derive_posture_level_missing_evidence_for_trace_missing() -> None:
    level = derive_posture_level([_trace_signal(status=EvidenceSignalStatus.MISSING)])
    assert level is EvidencePostureLevel.MISSING_EVIDENCE


def test_derive_posture_level_partial_for_core_passed_only() -> None:
    level = derive_posture_level([_core_signal(status=EvidenceSignalStatus.PASSED)])
    assert level is EvidencePostureLevel.PARTIAL


def test_derive_posture_level_unknown_for_deferred_and_separate_only() -> None:
    level = derive_posture_level(
        [
            create_evidence_signal(
                kind=EvidenceSignalKind.LIVE_TIER0_PROBES,
                status=EvidenceSignalStatus.DEFERRED,
                title="Live Tier-0 probes",
            ),
            create_evidence_signal(
                kind=EvidenceSignalKind.W_ADAPT_L4,
                status=EvidenceSignalStatus.SEPARATE,
                title="W-ADAPT L4",
                basis=EvidenceBasis.SEPARATE,
            ),
        ]
    )
    assert level is EvidencePostureLevel.UNKNOWN


def test_evidence_modules_have_no_applications_or_agents_imports() -> None:
    forbidden = ("applications.", "agents.", "from applications", "from agents")
    for path in _EVIDENCE_ROOT.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{path.name} contains forbidden import token: {token}"
