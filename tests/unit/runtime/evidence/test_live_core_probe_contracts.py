# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.evidence.live_core_probe_contracts import (
    LIVE_CORE_PROBE_CATALOG_ORDER,
    LiveCoreProbeArtifactKind,
    LiveCoreProbeArtifactRef,
    LiveCoreProbeEvidenceBasis,
    LiveCoreProbeKind,
    LiveCoreProbeReport,
    LiveCoreProbeStatus,
    create_live_core_probe_result,
    derive_live_core_probe_report_status,
    generate_live_core_probe_report_id,
    live_core_probe_kinds,
    validate_live_core_probe_report,
)

pytestmark = pytest.mark.unit

_CONTRACTS_PATH = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "evidence"
    / "live_core_probe_contracts.py"
)


def _passed_result(probe_kind: LiveCoreProbeKind) -> LiveCoreProbeResult:
    return create_live_core_probe_result(
        probe_kind=probe_kind,
        status=LiveCoreProbeStatus.PASSED,
        title=f"{probe_kind.value} passed",
    )


def _valid_report(
    *,
    results: list | None = None,
    status: LiveCoreProbeStatus | None = None,
) -> LiveCoreProbeReport:
    probe_results = results or [
        _passed_result(kind) for kind in live_core_probe_kinds()
    ]
    report_status = status or derive_live_core_probe_report_status(probe_results)
    return LiveCoreProbeReport(
        report_id=generate_live_core_probe_report_id(root_label="local"),
        status=report_status,
        results=probe_results,
    )


def test_generate_live_core_probe_report_id_is_deterministic() -> None:
    first = generate_live_core_probe_report_id(root_label="local")
    second = generate_live_core_probe_report_id(root_label="local")
    assert first == second == "live-core-probes:local"


def test_active_catalog_has_three_probes_in_order() -> None:
    kinds = live_core_probe_kinds()
    assert len(kinds) == 3
    assert tuple(kind.value for kind in kinds) == LIVE_CORE_PROBE_CATALOG_ORDER


def test_create_live_core_probe_result_uses_default_evidence_basis() -> None:
    result = create_live_core_probe_result(
        probe_kind=LiveCoreProbeKind.BASIC_RUN_COMPLETED_LIVE,
        status=LiveCoreProbeStatus.PASSED,
        title="Basic run completed",
    )
    assert result.evidence_basis == [
        LiveCoreProbeEvidenceBasis.LIVE_RUNTIME,
        LiveCoreProbeEvidenceBasis.MOCK_LLM,
        LiveCoreProbeEvidenceBasis.LOCAL_NO_NETWORK,
    ]
    assert result.artifact_refs == []
    assert result.metadata == {}


def test_derive_live_core_probe_report_status_empty() -> None:
    assert (
        derive_live_core_probe_report_status([])
        is LiveCoreProbeStatus.UNAVAILABLE
    )


def test_derive_live_core_probe_report_status_all_passed() -> None:
    results = [_passed_result(kind) for kind in live_core_probe_kinds()]
    assert (
        derive_live_core_probe_report_status(results)
        is LiveCoreProbeStatus.PASSED
    )


def test_derive_live_core_probe_report_status_any_failed() -> None:
    results = [
        create_live_core_probe_result(
            probe_kind=LiveCoreProbeKind.BASIC_RUN_COMPLETED_LIVE,
            status=LiveCoreProbeStatus.FAILED,
            title="Failed probe",
        )
    ]
    assert (
        derive_live_core_probe_report_status(results)
        is LiveCoreProbeStatus.FAILED
    )


def test_derive_live_core_probe_report_status_passed_and_skipped() -> None:
    results = [
        _passed_result(LiveCoreProbeKind.BASIC_RUN_COMPLETED_LIVE),
        create_live_core_probe_result(
            probe_kind=LiveCoreProbeKind.TRACE_PERSISTED_LIVE,
            status=LiveCoreProbeStatus.SKIPPED,
            title="Skipped probe",
        ),
    ]
    assert (
        derive_live_core_probe_report_status(results)
        is LiveCoreProbeStatus.SKIPPED
    )


def test_validate_live_core_probe_report_valid_three_passed() -> None:
    validate_live_core_probe_report(_valid_report())


def test_validate_live_core_probe_report_rejects_empty_report_id() -> None:
    report = _valid_report()
    report = report.model_copy(update={"report_id": "   "})
    with pytest.raises(ValueError, match="report_id must not be empty"):
        validate_live_core_probe_report(report)


def test_validate_live_core_probe_report_rejects_empty_results() -> None:
    report = LiveCoreProbeReport(
        report_id=generate_live_core_probe_report_id(root_label="local"),
        status=LiveCoreProbeStatus.UNAVAILABLE,
        results=[],
    )
    with pytest.raises(ValueError, match="results must not be empty"):
        validate_live_core_probe_report(report)


def test_validate_live_core_probe_report_rejects_duplicate_probe_kind() -> None:
    duplicate = _passed_result(LiveCoreProbeKind.BASIC_RUN_COMPLETED_LIVE)
    report = _valid_report(results=[duplicate, duplicate])
    with pytest.raises(ValueError, match="duplicate probe kind: basic_run_completed_live"):
        validate_live_core_probe_report(report)


def test_validate_live_core_probe_report_rejects_missing_required_evidence_basis() -> None:
    result = create_live_core_probe_result(
        probe_kind=LiveCoreProbeKind.BASIC_RUN_COMPLETED_LIVE,
        status=LiveCoreProbeStatus.PASSED,
        title="Missing basis",
        evidence_basis=[LiveCoreProbeEvidenceBasis.LIVE_RUNTIME],
    )
    report = _valid_report(results=[result])
    with pytest.raises(ValueError, match="result must include MOCK_LLM"):
        validate_live_core_probe_report(report)


def test_validate_live_core_probe_report_rejects_empty_artifact_path() -> None:
    result = _passed_result(LiveCoreProbeKind.BASIC_RUN_COMPLETED_LIVE)
    result = result.model_copy(
        update={
            "artifact_refs": [
                LiveCoreProbeArtifactRef(
                    kind=LiveCoreProbeArtifactKind.TRACE_RECORD,
                    path="   ",
                )
            ]
        }
    )
    report = _valid_report(results=[result])
    with pytest.raises(
        ValueError,
        match="result artifact path must not be empty \\(basic_run_completed_live\\)",
    ):
        validate_live_core_probe_report(report)


def test_validate_live_core_probe_report_rejects_status_mismatch() -> None:
    report = _valid_report(status=LiveCoreProbeStatus.FAILED)
    with pytest.raises(ValueError, match="report status must be PASSED"):
        validate_live_core_probe_report(report)


def test_live_core_probe_contracts_have_no_applications_or_agents_imports() -> None:
    forbidden = ("applications.", "agents.", "from applications", "from agents")
    source = _CONTRACTS_PATH.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, (
            f"live_core_probe_contracts.py contains forbidden import token: {token}"
        )
