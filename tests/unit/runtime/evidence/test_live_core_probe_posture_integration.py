# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.evidence.evidence_posture_collector import (
    collect_evidence_posture,
    load_live_core_probe_report_if_available,
)
from intergrax.runtime.evidence.evidence_posture_contracts import (
    EvidenceBasis,
    EvidencePostureLevel,
    EvidenceSignalKind,
    EvidenceSignalStatus,
)
from intergrax.runtime.evidence.live_core_probe_contracts import (
    LiveCoreProbeKind,
    LiveCoreProbeStatus,
    create_live_core_probe_result,
)
from intergrax.runtime.evidence.live_core_probe_export import write_live_core_probe_report
from intergrax.runtime.evidence.live_core_probe_runner import (
    build_live_core_probe_report,
    run_live_core_probes,
)

pytestmark = pytest.mark.unit


def _live_signal(summary):
    return next(
        signal
        for signal in summary.signals
        if signal.kind is EvidenceSignalKind.LIVE_TIER0_PROBES
    )


def test_collect_evidence_posture_missing_live_report_keeps_deferred(tmp_path: Path) -> None:
    summary = collect_evidence_posture(root=tmp_path)
    signal = _live_signal(summary)

    assert signal.status is EvidenceSignalStatus.DEFERRED
    assert signal.basis is EvidenceBasis.UNKNOWN


def test_collect_evidence_posture_passed_live_report_maps_passed(tmp_path: Path) -> None:
    report = run_live_core_probes(root_label="local")
    out_dir = tmp_path / "live_core_probes"
    json_path, _ = write_live_core_probe_report(report, out_dir)

    summary = collect_evidence_posture(
        root=tmp_path,
        live_core_probe_report_path=json_path,
    )
    signal = _live_signal(summary)

    assert signal.status is EvidenceSignalStatus.PASSED
    assert signal.basis is EvidenceBasis.LIVE_RUNTIME
    assert signal.metadata["scope"] == "selected_tier0_probes"
    assert signal.metadata["llm"] == "mock"
    assert signal.metadata["network"] == "disabled"
    assert signal.metadata["provider_calls"] == "disabled"


def test_collect_evidence_posture_failed_live_report_maps_failed(tmp_path: Path) -> None:
    failed_result = create_live_core_probe_result(
        probe_kind=LiveCoreProbeKind.BASIC_RUN_COMPLETED_LIVE,
        status=LiveCoreProbeStatus.FAILED,
        title="Basic run completed (live probe)",
        message="forced failure",
    )
    passed_results = [
        create_live_core_probe_result(
            probe_kind=LiveCoreProbeKind.TRACE_PERSISTED_LIVE,
            status=LiveCoreProbeStatus.PASSED,
            title="Trace persisted (live probe)",
        ),
        create_live_core_probe_result(
            probe_kind=LiveCoreProbeKind.TOOL_DENIED_BY_POLICY_LIVE,
            status=LiveCoreProbeStatus.PASSED,
            title="Tool denied by policy (live probe)",
        ),
    ]
    report = build_live_core_probe_report(
        results=[failed_result, *passed_results],
        root_label="local",
    )
    out_dir = tmp_path / "live_core_probes"
    json_path, _ = write_live_core_probe_report(report, out_dir)

    summary = collect_evidence_posture(
        root=tmp_path,
        live_core_probe_report_path=json_path,
    )
    signal = _live_signal(summary)

    assert signal.status is EvidenceSignalStatus.FAILED
    assert signal.basis is EvidenceBasis.LIVE_RUNTIME
    assert "1/3" in signal.message


def test_load_live_core_probe_report_if_available_raises_on_corrupt(tmp_path: Path) -> None:
    report_path = tmp_path / "live_core_report.json"
    report_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="failed to parse live core probe report at"):
        load_live_core_probe_report_if_available(report_path)


def test_live_probes_alone_do_not_make_posture_onboarding_ready(tmp_path: Path) -> None:
    report = run_live_core_probes(root_label="local")
    out_dir = tmp_path / "live_core_probes"
    json_path, _ = write_live_core_probe_report(report, out_dir)

    summary = collect_evidence_posture(
        root=tmp_path,
        live_core_probe_report_path=json_path,
    )

    assert summary.level is not EvidencePostureLevel.ONBOARDING_READY
    assert _live_signal(summary).status is EvidenceSignalStatus.PASSED
