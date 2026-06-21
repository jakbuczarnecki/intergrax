# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from intergrax.cli.evidence import run_evidence_posture, run_evidence_posture_export
from intergrax.runtime.evidence.cost_evidence_contracts import (
    CostEvidenceCheckKind,
    CostEvidenceStatus,
    create_cost_evidence_check_result,
)
from intergrax.runtime.evidence.cost_evidence_export import write_cost_evidence_report
from intergrax.runtime.evidence.cost_evidence_runner import build_cost_evidence_report
from intergrax.runtime.evidence.evidence_posture_collector import (
    collect_evidence_posture,
    load_cost_evidence_report_if_available,
)
from intergrax.runtime.evidence.evidence_posture_contracts import (
    EvidenceBasis,
    EvidencePostureLevel,
    EvidenceSignalKind,
    EvidenceSignalStatus,
)
from intergrax.runtime.evidence.scenario_runner import run_core_certification
from intergrax.runtime.evidence.trace_timeline_adapter import (
    build_timeline_from_certification_report,
)

pytestmark = pytest.mark.unit


def _cost_signal(summary):
    return next(
        (
            signal
            for signal in summary.signals
            if signal.kind is EvidenceSignalKind.COST_EVIDENCE
        ),
        None,
    )


def _write_cost_report(
    tmp_path: Path,
    *,
    status: CostEvidenceStatus,
) -> Path:
    if status is CostEvidenceStatus.PASSED:
        results = [
            create_cost_evidence_check_result(
                check_id="trace-budget-facets",
                check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
                status=CostEvidenceStatus.PASSED,
                title="Trace budget facets check available",
            )
        ]
    elif status is CostEvidenceStatus.FAILED:
        results = [
            create_cost_evidence_check_result(
                check_id="trace-budget-facets",
                check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
                status=CostEvidenceStatus.FAILED,
                title="Trace budget facets check failed",
            )
        ]
    elif status is CostEvidenceStatus.SKIPPED:
        results = [
            create_cost_evidence_check_result(
                check_id="trace-budget-facets",
                check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
                status=CostEvidenceStatus.PASSED,
                title="Trace budget facets check available",
            ),
            create_cost_evidence_check_result(
                check_id="core-budget-signals",
                check_kind=CostEvidenceCheckKind.CORE_BUDGET_SIGNALS,
                status=CostEvidenceStatus.SKIPPED,
                title="Core budget signals skipped",
            ),
        ]
    else:
        results = [
            create_cost_evidence_check_result(
                check_id="trace-budget-facets",
                check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
                status=CostEvidenceStatus.UNAVAILABLE,
                title="Trace budget facets check unavailable",
            )
        ]

    report = build_cost_evidence_report(results=results, root_label="local")
    out_dir = tmp_path / "cost"
    json_path, _ = write_cost_evidence_report(report, out_dir)
    return json_path


def _write_core_and_timeline(tmp_path: Path) -> tuple[Path, Path]:
    core_dir = tmp_path / "core"
    report = run_core_certification("L2", output_dir=core_dir)
    report_path = core_dir / "report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    timeline = build_timeline_from_certification_report(
        report,
        source_report_path=str(report_path),
    )
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    timeline_path = trace_dir / "timeline.json"
    timeline_path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
    return report_path, timeline_path


def _posture_args(
    tmp_path: Path,
    *,
    core_report: Path | None = None,
    trace_timeline: Path | None = None,
    cost_report: Path | None = None,
    output_dir: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        root=tmp_path,
        core_report=core_report,
        trace_timeline=trace_timeline,
        live_core_report=None,
        eval_report=None,
        cost_report=cost_report,
        root_label="local",
        no_operational_unknowns=False,
        output_dir=output_dir,
    )


def test_collect_evidence_posture_missing_cost_report_no_cost_signal(
    tmp_path: Path,
) -> None:
    summary = collect_evidence_posture(root=tmp_path)

    assert _cost_signal(summary) is None


def test_collect_evidence_posture_passed_cost_report_maps_passed(tmp_path: Path) -> None:
    json_path = _write_cost_report(tmp_path, status=CostEvidenceStatus.PASSED)

    summary = collect_evidence_posture(
        root=tmp_path,
        cost_evidence_report_path=json_path,
    )
    signal = _cost_signal(summary)

    assert signal is not None
    assert signal.status is EvidenceSignalStatus.PASSED
    assert signal.basis is EvidenceBasis.REPORT_DERIVED
    assert "1/1" in signal.message
    assert signal.metadata["scope"] == "cost_evidence"
    assert signal.metadata["provider_pricing"] == "disabled"
    assert signal.metadata["billing"] == "disabled"
    assert signal.metadata["cloud_cost_estimation"] == "disabled"
    assert signal.metadata["real_llm_metering"] == "disabled"
    assert signal.metadata["network"] == "disabled"


def test_collect_evidence_posture_failed_cost_report_maps_failed(tmp_path: Path) -> None:
    json_path = _write_cost_report(tmp_path, status=CostEvidenceStatus.FAILED)

    summary = collect_evidence_posture(
        root=tmp_path,
        cost_evidence_report_path=json_path,
    )
    signal = _cost_signal(summary)

    assert signal is not None
    assert signal.status is EvidenceSignalStatus.FAILED
    assert signal.basis is EvidenceBasis.REPORT_DERIVED
    assert "1/1" in signal.message


def test_collect_evidence_posture_skipped_cost_report_maps_unknown(tmp_path: Path) -> None:
    json_path = _write_cost_report(tmp_path, status=CostEvidenceStatus.SKIPPED)

    summary = collect_evidence_posture(
        root=tmp_path,
        cost_evidence_report_path=json_path,
    )
    signal = _cost_signal(summary)

    assert signal is not None
    assert signal.status is EvidenceSignalStatus.UNKNOWN
    assert signal.basis is EvidenceBasis.REPORT_DERIVED
    assert signal.message


def test_collect_evidence_posture_unavailable_cost_report_maps_unknown(
    tmp_path: Path,
) -> None:
    json_path = _write_cost_report(tmp_path, status=CostEvidenceStatus.UNAVAILABLE)

    summary = collect_evidence_posture(
        root=tmp_path,
        cost_evidence_report_path=json_path,
    )
    signal = _cost_signal(summary)

    assert signal is not None
    assert signal.status is EvidenceSignalStatus.UNKNOWN
    assert signal.basis is EvidenceBasis.REPORT_DERIVED


def test_load_cost_evidence_report_if_available_raises_on_corrupt(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="failed to parse cost evidence report at"):
        load_cost_evidence_report_if_available(report_path)


def test_cost_report_alone_does_not_make_posture_onboarding_ready(tmp_path: Path) -> None:
    json_path = _write_cost_report(tmp_path, status=CostEvidenceStatus.PASSED)

    summary = collect_evidence_posture(
        root=tmp_path,
        cost_evidence_report_path=json_path,
    )

    assert summary.level is not EvidencePostureLevel.ONBOARDING_READY
    assert _cost_signal(summary).status is EvidenceSignalStatus.PASSED


def test_core_trace_and_cost_passed_keeps_onboarding_ready(tmp_path: Path) -> None:
    report_path, timeline_path = _write_core_and_timeline(tmp_path)
    cost_path = _write_cost_report(tmp_path, status=CostEvidenceStatus.PASSED)

    summary = collect_evidence_posture(
        root=tmp_path,
        core_report_path=report_path,
        trace_timeline_path=timeline_path,
        cost_evidence_report_path=cost_path,
    )

    assert summary.level is EvidencePostureLevel.ONBOARDING_READY
    assert _cost_signal(summary).status is EvidenceSignalStatus.PASSED


def test_run_evidence_posture_accepts_cost_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cost_path = _write_cost_report(tmp_path, status=CostEvidenceStatus.PASSED)
    code = run_evidence_posture(_posture_args(tmp_path, cost_report=cost_path))
    captured = capsys.readouterr()

    assert code == 0
    assert "COST_EVIDENCE" in captured.out


def test_run_evidence_posture_export_accepts_cost_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cost_path = _write_cost_report(tmp_path, status=CostEvidenceStatus.PASSED)
    out_dir = tmp_path / "posture"
    code = run_evidence_posture_export(
        _posture_args(tmp_path, cost_report=cost_path, output_dir=out_dir)
    )
    captured = capsys.readouterr()

    assert code == 0
    assert (out_dir / "posture.json").is_file()
    assert "COST_EVIDENCE" in (out_dir / "posture.json").read_text(encoding="utf-8")
    assert "posture.json" in captured.out


def test_cost_signal_does_not_imply_provider_pricing_or_billing(tmp_path: Path) -> None:
    json_path = _write_cost_report(tmp_path, status=CostEvidenceStatus.PASSED)

    summary = collect_evidence_posture(
        root=tmp_path,
        cost_evidence_report_path=json_path,
    )
    signal = _cost_signal(summary)

    assert signal is not None
    assert signal.metadata["provider_pricing"] == "disabled"
    assert signal.metadata["billing"] == "disabled"
    assert signal.metadata["cloud_cost_estimation"] == "disabled"
    assert signal.metadata["real_llm_metering"] == "disabled"
    assert signal.metadata["network"] == "disabled"
