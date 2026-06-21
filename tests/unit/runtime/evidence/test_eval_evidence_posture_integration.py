# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from intergrax.cli.evidence import run_evidence_posture, run_evidence_posture_export
from intergrax.runtime.evidence.eval_evidence_contracts import (
    EvalEvidenceCheckKind,
    EvalEvidenceStatus,
    create_eval_evidence_check_result,
)
from intergrax.runtime.evidence.eval_evidence_export import write_eval_evidence_report
from intergrax.runtime.evidence.eval_evidence_runner import build_eval_evidence_report
from intergrax.runtime.evidence.evidence_posture_collector import (
    collect_evidence_posture,
    load_eval_evidence_report_if_available,
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


def _eval_signal(summary):
    return next(
        (
            signal
            for signal in summary.signals
            if signal.kind is EvidenceSignalKind.EVAL_REGRESSION
        ),
        None,
    )


def _write_eval_report(
    tmp_path: Path,
    *,
    status: EvalEvidenceStatus,
) -> Path:
    if status is EvalEvidenceStatus.PASSED:
        results = [
            create_eval_evidence_check_result(
                check_id="eval-scenario-library",
                check_kind=EvalEvidenceCheckKind.SCENARIO_LIBRARY,
                status=EvalEvidenceStatus.PASSED,
                title="Eval scenario library check available",
            )
        ]
    elif status is EvalEvidenceStatus.FAILED:
        results = [
            create_eval_evidence_check_result(
                check_id="eval-scenario-library",
                check_kind=EvalEvidenceCheckKind.SCENARIO_LIBRARY,
                status=EvalEvidenceStatus.FAILED,
                title="Eval scenario library check failed",
            )
        ]
    elif status is EvalEvidenceStatus.SKIPPED:
        results = [
            create_eval_evidence_check_result(
                check_id="eval-scenario-library",
                check_kind=EvalEvidenceCheckKind.SCENARIO_LIBRARY,
                status=EvalEvidenceStatus.PASSED,
                title="Eval scenario library check available",
            ),
            create_eval_evidence_check_result(
                check_id="eval-regression-surface",
                check_kind=EvalEvidenceCheckKind.REGRESSION_SURFACE,
                status=EvalEvidenceStatus.SKIPPED,
                title="Eval regression surface skipped",
            ),
        ]
    else:
        results = [
            create_eval_evidence_check_result(
                check_id="eval-scenario-library",
                check_kind=EvalEvidenceCheckKind.SCENARIO_LIBRARY,
                status=EvalEvidenceStatus.UNAVAILABLE,
                title="Eval scenario library check unavailable",
            )
        ]

    report = build_eval_evidence_report(results=results, root_label="local")
    out_dir = tmp_path / "eval"
    json_path, _ = write_eval_evidence_report(report, out_dir)
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
    eval_report: Path | None = None,
    output_dir: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        root=tmp_path,
        core_report=core_report,
        trace_timeline=trace_timeline,
        live_core_report=None,
        eval_report=eval_report,
        root_label="local",
        no_operational_unknowns=False,
        output_dir=output_dir,
    )


def test_collect_evidence_posture_missing_eval_report_no_eval_signal(
    tmp_path: Path,
) -> None:
    summary = collect_evidence_posture(root=tmp_path)

    assert _eval_signal(summary) is None


def test_collect_evidence_posture_passed_eval_report_maps_passed(tmp_path: Path) -> None:
    json_path = _write_eval_report(tmp_path, status=EvalEvidenceStatus.PASSED)

    summary = collect_evidence_posture(
        root=tmp_path,
        eval_evidence_report_path=json_path,
    )
    signal = _eval_signal(summary)

    assert signal is not None
    assert signal.status is EvidenceSignalStatus.PASSED
    assert signal.basis is EvidenceBasis.REPORT_DERIVED
    assert "1/1" in signal.message
    assert signal.metadata["scope"] == "eval_regression_evidence"
    assert signal.metadata["llm"] == "none"
    assert signal.metadata["network"] == "disabled"
    assert signal.metadata["provider_calls"] == "disabled"
    assert signal.metadata["real_llm_evaluation"] == "disabled"


def test_collect_evidence_posture_failed_eval_report_maps_failed(tmp_path: Path) -> None:
    json_path = _write_eval_report(tmp_path, status=EvalEvidenceStatus.FAILED)

    summary = collect_evidence_posture(
        root=tmp_path,
        eval_evidence_report_path=json_path,
    )
    signal = _eval_signal(summary)

    assert signal is not None
    assert signal.status is EvidenceSignalStatus.FAILED
    assert signal.basis is EvidenceBasis.REPORT_DERIVED
    assert "1/1" in signal.message


def test_collect_evidence_posture_skipped_eval_report_maps_unknown(tmp_path: Path) -> None:
    json_path = _write_eval_report(tmp_path, status=EvalEvidenceStatus.SKIPPED)

    summary = collect_evidence_posture(
        root=tmp_path,
        eval_evidence_report_path=json_path,
    )
    signal = _eval_signal(summary)

    assert signal is not None
    assert signal.status is EvidenceSignalStatus.UNKNOWN
    assert signal.basis is EvidenceBasis.REPORT_DERIVED
    assert signal.message


def test_collect_evidence_posture_unavailable_eval_report_maps_unknown(
    tmp_path: Path,
) -> None:
    json_path = _write_eval_report(tmp_path, status=EvalEvidenceStatus.UNAVAILABLE)

    summary = collect_evidence_posture(
        root=tmp_path,
        eval_evidence_report_path=json_path,
    )
    signal = _eval_signal(summary)

    assert signal is not None
    assert signal.status is EvidenceSignalStatus.UNKNOWN
    assert signal.basis is EvidenceBasis.REPORT_DERIVED


def test_load_eval_evidence_report_if_available_raises_on_corrupt(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="failed to parse eval evidence report at"):
        load_eval_evidence_report_if_available(report_path)


def test_eval_report_alone_does_not_make_posture_onboarding_ready(tmp_path: Path) -> None:
    json_path = _write_eval_report(tmp_path, status=EvalEvidenceStatus.PASSED)

    summary = collect_evidence_posture(
        root=tmp_path,
        eval_evidence_report_path=json_path,
    )

    assert summary.level is not EvidencePostureLevel.ONBOARDING_READY
    assert _eval_signal(summary).status is EvidenceSignalStatus.PASSED


def test_core_trace_and_eval_passed_keeps_onboarding_ready(tmp_path: Path) -> None:
    report_path, timeline_path = _write_core_and_timeline(tmp_path)
    eval_path = _write_eval_report(tmp_path, status=EvalEvidenceStatus.PASSED)

    summary = collect_evidence_posture(
        root=tmp_path,
        core_report_path=report_path,
        trace_timeline_path=timeline_path,
        eval_evidence_report_path=eval_path,
    )

    assert summary.level is EvidencePostureLevel.ONBOARDING_READY
    assert _eval_signal(summary).status is EvidenceSignalStatus.PASSED


def test_run_evidence_posture_accepts_eval_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    eval_path = _write_eval_report(tmp_path, status=EvalEvidenceStatus.PASSED)
    code = run_evidence_posture(_posture_args(tmp_path, eval_report=eval_path))
    captured = capsys.readouterr()

    assert code == 0
    assert "EVAL_REGRESSION" in captured.out


def test_run_evidence_posture_export_accepts_eval_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    eval_path = _write_eval_report(tmp_path, status=EvalEvidenceStatus.PASSED)
    out_dir = tmp_path / "posture"
    code = run_evidence_posture_export(
        _posture_args(tmp_path, eval_report=eval_path, output_dir=out_dir)
    )
    captured = capsys.readouterr()

    assert code == 0
    assert (out_dir / "posture.json").is_file()
    assert "EVAL_REGRESSION" in (out_dir / "posture.json").read_text(encoding="utf-8")
    assert "posture.json" in captured.out
