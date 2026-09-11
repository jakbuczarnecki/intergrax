# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-L1.R4.R1.OBS regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.diagnostics.completion_alignment_diag import (
    AlignmentDirection,
    AlignmentStatus,
    CompletionMode,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment_correction import (
    CompletionAlignmentDirection,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_coverage_analysis import (
    _aggregate_coverage_metrics,
    _fifteen_kb_effect_from_coverage,
    run_behavioral_coverage_analysis,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_coverage_evidence import (
    AlignmentEvidenceStatus,
    BehavioralCoverageRunVerdict,
    classify_run_coverage_from_item,
    extract_coverage_evidence,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_analysis import (
    FifteenKBEffectVerdict,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeStatus,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
    COMPLETION_ALIGNMENT_TRACE_SCHEMA,
)


_RUN_A = "run_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_RUN_B = "run_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
_RUN_C = "run_cccccccccccccccccccccccccccccccc"


def _alignment_event(
    run_id: str = _RUN_A,
    **fields: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": run_id,
        "node_id": "node_incident_investigator",
        "completion_mode": CompletionMode.UNRESOLVED.value,
        "alignment_status": AlignmentStatus.MATCH.value,
        "alignment_direction": AlignmentDirection.NONE.value,
        "mismatch_reason": None,
        "correctable": False,
        "supported_state_present": False,
        "supported_hypothesis_id": None,
        "supported_resolution": None,
    }
    payload.update(fields)
    return {
        "payload_schema_id": COMPLETION_ALIGNMENT_TRACE_SCHEMA,
        "payload": payload,
    }


def _attempt_event(
    run_id: str, *, attempt_index: int = 0, max_iterations: int = 2
) -> dict[str, object]:
    return {
        "payload_schema_id": CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
        "payload": {
            "run_id": run_id,
            "node_id": "investigator",
            "attempt_index": attempt_index,
            "max_iterations": max_iterations,
        },
    }


def test_match_path_valid_match() -> None:
    run = {
        "run_id": _RUN_A,
        "trace_available": True,
        "trace_events": [_alignment_event(run_id=_RUN_A)],
    }
    evidence = extract_coverage_evidence(run)
    assert evidence.alignment_event_status is AlignmentEvidenceStatus.PRESENT
    assert evidence.mismatch_detected is False
    assert evidence.revision_path_entered is False
    assert classify_run_coverage_from_item(run) is BehavioralCoverageRunVerdict.VALID_MATCH


def test_missing_telemetry_incomplete_evidence() -> None:
    run = {
        "run_id": _RUN_B,
        "trace_available": True,
        "trace_events": [],
    }
    evidence = extract_coverage_evidence(run)
    assert evidence.alignment_event_status is AlignmentEvidenceStatus.NOT_EMITTED
    assert evidence.mismatch_detected is None
    assert classify_run_coverage_from_item(run) is BehavioralCoverageRunVerdict.INCOMPLETE_EVIDENCE


def test_reverse_correction_revision_expected() -> None:
    run = {
        "run_id": _RUN_C,
        "trace_available": True,
        "trace_events": [
            _alignment_event(
                run_id=_RUN_C,
                completion_mode=CompletionMode.SUPPORTED_DIAGNOSIS.value,
                alignment_status=AlignmentStatus.MISMATCH.value,
                alignment_direction=AlignmentDirection.REVERSE.value,
                mismatch_reason="supported_diagnosis_without_supported_state",
                correctable=True,
                supported_state_present=False,
            ),
            _attempt_event(_RUN_C, attempt_index=0, max_iterations=2),
        ],
    }
    assert classify_run_coverage_from_item(run) is BehavioralCoverageRunVerdict.REVISION_EXPECTED


def test_no_false_positive_cohort_not_proven() -> None:
    runs = [
        {
            "run_id": f"run-{index}",
            "trace_available": True,
            "trace_events": [_alignment_event()],
        }
        for index in range(3)
    ]
    evidence = tuple(extract_coverage_evidence(item) for item in runs)
    alignment, correction, typed_context, repair = _aggregate_coverage_metrics(evidence)
    effect = _fifteen_kb_effect_from_coverage(
        SourceFreezeStatus.PASS,
        alignment,
        correction,
        typed_context,
        repair,
        evidence,
    )
    assert effect is FifteenKBEffectVerdict.NOT_PROVEN


def test_cohort_zero_alignment_events_inconclusive() -> None:
    runs = [
        {
            "run_id": "run-missing",
            "trace_available": True,
            "trace_events": [],
        }
    ]
    evidence = tuple(extract_coverage_evidence(item) for item in runs)
    alignment, correction, typed_context, repair = _aggregate_coverage_metrics(evidence)
    effect = _fifteen_kb_effect_from_coverage(
        SourceFreezeStatus.PASS,
        alignment,
        correction,
        typed_context,
        repair,
        evidence,
    )
    assert alignment.alignment_event_present == 0
    assert effect is FifteenKBEffectVerdict.INCONCLUSIVE


def test_r4r1_obs_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    session_dir = (
        repo_root / ".tmp" / "session" / "DS-E2E-15J-L1-R4-R1" / "cohort-retry-dirty"
    )
    if not (session_dir / "runs.json").is_file():
        pytest.skip("frozen cohort session not available locally")
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    first = run_behavioral_coverage_analysis(
        repo_root=repo_root,
        session_dir=session_dir,
        output_dir=out_a,
    )
    second = run_behavioral_coverage_analysis(
        repo_root=repo_root,
        session_dir=session_dir,
        output_dir=out_b,
    )
    assert first.source_freeze.status is SourceFreezeStatus.PASS
    assert first.alignment.alignment_event_present == second.alignment.alignment_event_present
    assert (out_a / "artifact-manifest.txt").is_file()
    assert (out_a / "coverage_summary.md").read_text(encoding="utf-8") == (
        out_b / "coverage_summary.md"
    ).read_text(encoding="utf-8")
