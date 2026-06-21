# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.evidence.live_core_probe_contracts import (
    LIVE_CORE_PROBE_CATALOG_ORDER,
    LiveCoreProbeEvidenceBasis,
    LiveCoreProbeKind,
    LiveCoreProbeStatus,
    create_live_core_probe_result,
    validate_live_core_probe_report,
)
from intergrax.runtime.evidence.live_core_probe_runner import (
    build_live_core_probe_report,
    generate_live_core_probe_run_id,
    run_basic_run_completed_live_probe,
    run_live_core_probes,
    run_tool_denied_by_policy_live_probe,
    run_trace_persisted_live_probe,
)

pytestmark = pytest.mark.unit

_RUNNER_PATH = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "evidence"
    / "live_core_probe_runner.py"
)

_REQUIRED_EVIDENCE_BASIS = [
    LiveCoreProbeEvidenceBasis.LIVE_RUNTIME,
    LiveCoreProbeEvidenceBasis.MOCK_LLM,
    LiveCoreProbeEvidenceBasis.LOCAL_NO_NETWORK,
]


def test_generate_live_core_probe_run_id_is_deterministic() -> None:
    first = generate_live_core_probe_run_id(root_label="local")
    second = generate_live_core_probe_run_id(root_label="local")
    assert first == second == "live-core-probe-run:local"


@pytest.mark.parametrize(
    "runner",
    [
        run_basic_run_completed_live_probe,
        run_trace_persisted_live_probe,
        run_tool_denied_by_policy_live_probe,
    ],
)
def test_individual_probe_returns_passed(runner) -> None:
    result = runner(root_label="local")
    assert result.status is LiveCoreProbeStatus.PASSED


@pytest.mark.parametrize(
    "runner",
    [
        run_basic_run_completed_live_probe,
        run_trace_persisted_live_probe,
        run_tool_denied_by_policy_live_probe,
    ],
)
def test_individual_probe_includes_required_evidence_basis(runner) -> None:
    result = runner(root_label="local")
    assert result.evidence_basis == _REQUIRED_EVIDENCE_BASIS


def test_run_live_core_probes_returns_passed_report() -> None:
    report = run_live_core_probes(root_label="local")
    assert report.status is LiveCoreProbeStatus.PASSED


def test_run_live_core_probes_returns_three_results_in_canonical_order() -> None:
    report = run_live_core_probes(root_label="local")
    assert len(report.results) == 3
    assert tuple(result.probe_kind.value for result in report.results) == LIVE_CORE_PROBE_CATALOG_ORDER


def test_run_live_core_probes_report_validates() -> None:
    report = run_live_core_probes(root_label="local")
    validate_live_core_probe_report(report)


def test_build_live_core_probe_report_derives_failed_when_one_result_failed() -> None:
    results = [
        create_live_core_probe_result(
            probe_kind=LiveCoreProbeKind.BASIC_RUN_COMPLETED_LIVE,
            status=LiveCoreProbeStatus.PASSED,
            title="Passed probe",
        ),
        create_live_core_probe_result(
            probe_kind=LiveCoreProbeKind.TRACE_PERSISTED_LIVE,
            status=LiveCoreProbeStatus.FAILED,
            title="Failed probe",
        ),
        create_live_core_probe_result(
            probe_kind=LiveCoreProbeKind.TOOL_DENIED_BY_POLICY_LIVE,
            status=LiveCoreProbeStatus.PASSED,
            title="Passed probe",
        ),
    ]
    report = build_live_core_probe_report(results=results, root_label="local")
    assert report.status is LiveCoreProbeStatus.FAILED
    assert "failed" in report.summary.lower()


def test_live_core_probe_runner_has_no_applications_or_agents_imports() -> None:
    forbidden = (
        "applications.",
        "agents.",
        "from applications",
        "from agents",
        "intergrax.cli",
    )
    source = _RUNNER_PATH.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, (
            f"live_core_probe_runner.py contains forbidden import token: {token}"
        )


def test_live_core_probe_runner_has_no_obvious_network_imports() -> None:
    forbidden = ("requests", "httpx", "urllib", "socket")
    source = _RUNNER_PATH.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, (
            f"live_core_probe_runner.py contains forbidden network token: {token}"
        )
