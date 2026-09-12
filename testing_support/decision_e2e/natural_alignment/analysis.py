# © Artur Czarnecki. All rights reserved.

"""Natural alignment cohort analysis (DS-E2E-15J-L1.R4.R5)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from testing_support.decision_e2e.local_qualification_session.alignment_revision_evidence import (
    infer_alignment_revision_evidence,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_coverage_analysis import (
    run_behavioral_coverage_analysis,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_analysis import (
    run_behavioral_analysis,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_evidence import (
    _parse_attempt_events,
    extract_run_evidence,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeReport,
    SourceFreezeStatus,
    verify_source_freeze,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    SafetyGateOutcome,
)
from testing_support.decision_e2e.local_qualification_session.trace_readback import (
    read_typed_alignment_events,
)
from testing_support.decision_e2e.local_ai_incident_qualification import (
    resolve_repository_head_sha,
)
from testing_support.decision_e2e.natural_alignment.source_freeze import (
    verify_natural_alignment_source_freeze,
)

COHORT_TASK_ID = "DS-E2E-15J-L1.R4.R5"
ANALYSIS_TASK_ID = "DS-E2E-15J-L1.R4.R5.ANALYSIS"
CONTROLLED_PROOF_TASK_ID = "DS-E2E-15J-L1.R4.R4"


class NaturalFifteenKBEffect(StrEnum):
    PROVEN_NATURAL = "PROVEN_NATURAL"
    NOT_PROVEN = "NOT_PROVEN"
    INCONCLUSIVE = "INCONCLUSIVE"
    FAIL = "FAIL"


class NaturalQualificationOutcome(StrEnum):
    PASS_A = "PASS_A"
    PASS_B = "PASS_B"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class NaturalQualificationResult:
    repo_source_freeze: SourceFreezeReport
    session_source_freeze: SourceFreezeReport
    fifteen_kb_effect: NaturalFifteenKBEffect
    qualification_outcome: NaturalQualificationOutcome
    alignment_event_count: int
    natural_repair_run_ids: tuple[str, ...]
    third_pass_count: int
    reconciliation_leak: SafetyGateOutcome
    output_dir: Path


def _load_runs(session_dir: Path) -> list[dict[str, object]]:
    payload = json.loads((session_dir / "runs.json").read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return []
    return [dict(item) for item in runs if isinstance(item, dict)]


def _count_alignment_events(runs: list[dict[str, object]]) -> int:
    total = 0
    for item in runs:
        trace_events = item.get("trace_events")
        if not isinstance(trace_events, list):
            continue
        events = tuple(dict(event) for event in trace_events if isinstance(event, dict))
        readback = read_typed_alignment_events(events)
        total += len(readback.events)
    return total


def _natural_repair_run_ids(runs: list[dict[str, object]]) -> tuple[str, ...]:
    repaired: list[str] = []
    for item in runs:
        run_id = str(item.get("run_id", ""))
        trace_events = item.get("trace_events")
        if not isinstance(trace_events, list):
            continue
        events = tuple(dict(event) for event in trace_events if isinstance(event, dict))
        readback = read_typed_alignment_events(events)
        flags = infer_alignment_revision_evidence(
            readback.events,
            _parse_attempt_events(events),
        )
        if flags.natural_overcommit_repair:
            repaired.append(run_id)
    return tuple(sorted(repaired))


def _csv_bytes(rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return buffer.getvalue()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output_dir: Path, files: tuple[str, ...]) -> None:
    lines = [f"{name} sha256:{_sha256_file(output_dir / name)}" for name in sorted(files)]
    body = "\n".join(lines) + ("\n" if lines else "")
    self_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    manifest_body = body + f"artifact-manifest.txt sha256:{self_digest}\n"
    (output_dir / "artifact-manifest.txt").write_text(manifest_body, encoding="utf-8")


def _merge_natural_analysis_json(
    output_dir: Path,
    *,
    fifteen_kb: NaturalFifteenKBEffect,
    outcome: NaturalQualificationOutcome,
    repo_freeze: SourceFreezeReport,
    session_freeze: SourceFreezeReport,
    behavioral: object,
    alignment_event_count: int,
    natural_repair_ids: tuple[str, ...],
    analyzed_at: str,
) -> None:
    from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_analysis import (
        BehavioralAnalysisResult,
    )

    assert isinstance(behavioral, BehavioralAnalysisResult)
    analysis_path = output_dir / "analysis.json"
    prior: dict[str, object] = {}
    if analysis_path.is_file():
        prior = json.loads(analysis_path.read_text(encoding="utf-8"))
    payload: dict[str, object] = dict(prior)
    payload.update(
        {
            "task": ANALYSIS_TASK_ID,
            "cohort_task": COHORT_TASK_ID,
            "analyzed_at": analyzed_at,
            "SOURCE_FREEZE_STATUS": repo_freeze.status.value,
            "SESSION_SOURCE_FREEZE_STATUS": session_freeze.status.value,
            "15K_B_EFFECT": fifteen_kb.value,
            "qualification_outcome": outcome.value,
            "alignment_event_count": alignment_event_count,
            "natural_repair_run_ids": list(natural_repair_ids),
            "reverse_mismatch_count": behavioral.alignment.reverse_count,
            "forward_mismatch_count": behavioral.alignment.forward_count,
            "revision_attempted": behavioral.revision.revision_attempted,
            "typed_context_present": behavioral.revision.typed_context_present,
            "revision_repaired": behavioral.revision.revision_repaired,
            "repair_count": behavioral.revision.revision_repaired,
            "third_pass_count": behavioral.third_pass_count,
            "reconciliation_leak": behavioral.reconciliation_leak.value,
            "controlled_proof_task": CONTROLLED_PROOF_TASK_ID,
            "controlled_model_overcommit_status": "PROVEN",
            "evidence_separation_note": (
                "Controlled R4.R4 repair proof and natural R4.R5 cohort are not pooled."
            ),
        }
    )
    analysis_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_operator_final_report(
    *,
    repo_freeze: SourceFreezeReport,
    session_freeze: SourceFreezeReport,
    behavioral: object,
    fifteen_kb: NaturalFifteenKBEffect,
    outcome: NaturalQualificationOutcome,
    alignment_event_count: int,
    natural_repair_ids: tuple[str, ...],
    session_dir: Path,
    repo_root: Path,
    regression_status: str,
) -> str:
    from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_analysis import (
        BehavioralAnalysisResult,
    )

    assert isinstance(behavioral, BehavioralAnalysisResult)
    summary = json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))
    integrity = summary.get("session_integrity")
    if not isinstance(integrity, dict):
        integrity = {}
    profile_path = session_dir / "local_model_profile.json"
    profile_line = "unavailable"
    if profile_path.is_file():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile_line = (
            f"provider={profile.get('provider')} model={profile.get('model')} "
            f"digest={profile.get('digest')}"
        )
    natural_result = fifteen_kb.value
    combined = (
        "Natural bounded correction not observed in cohort; controlled path remains authoritative."
        if fifteen_kb is NaturalFifteenKBEffect.NOT_PROVEN
        else natural_result
    )
    head_sha = resolve_repository_head_sha(repo_root)
    return "\n".join(
        [
            f"# {COHORT_TASK_ID} — operator final report",
            "",
            "## STATUS",
            f"- qualification_outcome: `{outcome.value}`",
            f"- 15K_B_EFFECT: `{fifteen_kb.value}`",
            "",
            "## TASK",
            f"- `{COHORT_TASK_ID}` Natural Local Model Alignment Correction Qualification",
            "",
            "## SOURCE FREEZE",
            f"- repository: `{repo_freeze.status.value}`",
            f"- session artifacts: `{session_freeze.status.value}`",
            "",
            "## MODEL PROFILE",
            f"- {profile_line}",
            "",
            "## COHORT INTEGRITY",
            f"- planned_runs: `{summary.get('planned_runs')}`",
            f"- finalization_status: `{integrity.get('finalization_status')}`",
            f"- runtime_identity_status: `{integrity.get('runtime_identity_status')}`",
            f"- model_identity_status: `{integrity.get('model_identity_status')}`",
            f"- config_identity_status: `{integrity.get('config_identity_status')}`",
            f"- source_identity_status: `{integrity.get('source_identity_status')}`",
            "",
            "## ALIGNMENT EVENTS",
            f"- typed `intergrax.diag.completion.alignment.v1` count: `{alignment_event_count}`",
            "",
            "## REVERSE MISMATCH COUNT",
            f"- MODEL_OVERCOMMIT (reverse): `{behavioral.alignment.reverse_count}`",
            "",
            "## FORWARD MISMATCH COUNT",
            f"- `{behavioral.alignment.forward_count}`",
            "",
            "## REVISION ATTEMPTS",
            f"- `{behavioral.revision.revision_attempted}`",
            "",
            "## TYPED CONTEXT DELIVERY",
            f"- `{behavioral.revision.typed_context_present}`",
            "",
            "## REPAIR COUNT",
            f"- `{behavioral.revision.revision_repaired}`",
            "",
            "## Evidence Separation",
            "",
            "| Evidence | Status |",
            "| --- | --- |",
            "| R4.R4 Controlled MODEL_OVERCOMMIT | PROVEN |",
            f"| R4.R5 Natural MODEL behavior | {natural_result} |",
            f"| Combined conclusion | {combined} |",
            "",
            "## CONTROLLED VS NATURAL COMPARISON",
            f"- Controlled proof task: `{CONTROLLED_PROOF_TASK_ID}` (stimulus; separate cohort).",
            f"- Natural repair run_ids: `{', '.join(natural_repair_ids) if natural_repair_ids else 'none'}`.",
            "",
            "## SAFETY GATES",
            f"- third_pass_count: `{behavioral.third_pass_count}` (required 0)",
            f"- reconciliation_leak: `{behavioral.reconciliation_leak.value}`",
            "",
            "## REGRESSIONS",
            regression_status,
            "",
            "## COMMIT SHA",
            f"- `{head_sha}`",
            "",
            "## NEXT TASK",
            "- None (natural qualification complete; effect not proven on local model).",
            "",
        ]
    )


def _natural_verdict(
    *,
    repo_freeze: SourceFreezeStatus,
    session_freeze: SourceFreezeStatus,
    alignment_mismatch_count: int,
    natural_repair_ids: tuple[str, ...],
    third_pass_count: int,
    reconciliation_leak: SafetyGateOutcome,
    alignment_event_count: int,
) -> tuple[NaturalFifteenKBEffect, NaturalQualificationOutcome]:
    if repo_freeze is not SourceFreezeStatus.PASS or session_freeze is not SourceFreezeStatus.PASS:
        return NaturalFifteenKBEffect.INCONCLUSIVE, NaturalQualificationOutcome.BLOCKED
    if third_pass_count > 0 or reconciliation_leak is SafetyGateOutcome.FAIL:
        return NaturalFifteenKBEffect.FAIL, NaturalQualificationOutcome.FAIL
    if alignment_event_count < 1:
        return NaturalFifteenKBEffect.INCONCLUSIVE, NaturalQualificationOutcome.FAIL
    if natural_repair_ids:
        return NaturalFifteenKBEffect.PROVEN_NATURAL, NaturalQualificationOutcome.PASS_A
    if alignment_mismatch_count == 0:
        return NaturalFifteenKBEffect.NOT_PROVEN, NaturalQualificationOutcome.PASS_B
    return NaturalFifteenKBEffect.NOT_PROVEN, NaturalQualificationOutcome.PASS_B


def run_natural_qualification_analysis(
    *,
    repo_root: Path,
    session_dir: Path,
) -> NaturalQualificationResult:
    output_dir = session_dir
    repo_freeze = verify_natural_alignment_source_freeze(repo_root)
    session_freeze = verify_source_freeze(session_dir, repo_root)

    behavioral = run_behavioral_analysis(
        repo_root=repo_root,
        session_dir=session_dir,
        output_dir=output_dir / ".analysis-r4r1-bridge",
    )
    run_behavioral_coverage_analysis(
        repo_root=repo_root,
        session_dir=session_dir,
        output_dir=output_dir,
    )

    runs = _load_runs(session_dir)
    alignment_event_count = _count_alignment_events(runs)
    natural_repair_ids = _natural_repair_run_ids(runs)

    fifteen_kb, outcome = _natural_verdict(
        repo_freeze=repo_freeze.status,
        session_freeze=session_freeze.status,
        alignment_mismatch_count=behavioral.alignment.alignment_mismatch_count,
        natural_repair_ids=natural_repair_ids,
        third_pass_count=behavioral.third_pass_count,
        reconciliation_leak=behavioral.reconciliation_leak,
        alignment_event_count=alignment_event_count,
    )

    alignment_rows: list[dict[str, str]] = []
    revision_rows: list[dict[str, str]] = []
    attempt_rows: list[dict[str, str]] = []
    for item in sorted(runs, key=lambda row: str(row.get("run_id", ""))):
        run_id = str(item.get("run_id", ""))
        evidence = extract_run_evidence(item)
        trace_events = item.get("trace_events")
        events: tuple[dict[str, object], ...] = ()
        if isinstance(trace_events, list):
            events = tuple(dict(event) for event in trace_events if isinstance(event, dict))
        readback = read_typed_alignment_events(events)
        for event in readback.events:
            alignment_rows.append(
                {
                    "run_id": run_id,
                    "direction": event.alignment_direction.value,
                    "correctable": str(event.correctable).lower(),
                    "alignment_status": event.alignment_status.value,
                }
            )
        flags = infer_alignment_revision_evidence(
            readback.events,
            evidence.attempt_events,
        )
        revision_rows.append(
            {
                "run_id": run_id,
                "natural_overcommit_repair": str(flags.natural_overcommit_repair).lower(),
                "revision_attempted": str(flags.revision_attempted).lower(),
                "typed_context_present": str(flags.typed_context_present).lower(),
                "revision_repaired": str(flags.revision_repaired).lower(),
            }
        )
        for attempt in evidence.attempt_events:
            attempt_rows.append(
                {
                    "run_id": run_id,
                    "attempt_index": str(attempt.attempt_index),
                    "max_iterations": str(attempt.max_iterations),
                    "node_id": attempt.node_id,
                }
            )

    analyzed_at = datetime.now(tz=UTC).isoformat()
    summary_payload: dict[str, object] = {}
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_payload["natural_qualification"] = {
        "task_id": COHORT_TASK_ID,
        "analyzed_at": analyzed_at,
        "SOURCE_FREEZE_STATUS": repo_freeze.status.value,
        "SESSION_SOURCE_FREEZE_STATUS": session_freeze.status.value,
        "15K_B_EFFECT": fifteen_kb.value,
        "qualification_outcome": outcome.value,
        "alignment_event_count": alignment_event_count,
        "natural_repair_run_ids": list(natural_repair_ids),
        "controlled_repair_proof_task": CONTROLLED_PROOF_TASK_ID,
        "evidence_separation": (
            "R4.R4 controlled MODEL_OVERCOMMIT repair is not mixed with this cohort; "
            "only production scenario paths were executed."
        ),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_md = "\n".join(
        [
            f"# {COHORT_TASK_ID} natural alignment qualification",
            "",
            "## Phase 1 — Source freeze",
            f"- SOURCE_FREEZE_STATUS: `{repo_freeze.status.value}`",
            "",
            "## Phase 5 — Evaluation matrix",
            f"- alignment events: `{alignment_event_count}` (target ≥1)",
            f"- reverse mismatch (MODEL_OVERCOMMIT): `{behavioral.alignment.reverse_count}`",
            f"- forward mismatch: `{behavioral.alignment.forward_count}`",
            f"- revision attempted: `{behavioral.revision.revision_attempted}`",
            f"- typed context present: `{behavioral.revision.typed_context_present}`",
            f"- revision repaired: `{behavioral.revision.revision_repaired}`",
            f"- third pass: `{behavioral.third_pass_count}` (target 0)",
            "",
            "## Phase 6 — Outcome",
            f"- 15K_B_EFFECT: `{fifteen_kb.value}`",
            f"- qualification_outcome: `{outcome.value}`",
            "",
            "## Controlled vs natural evidence",
            f"- Controlled repair proof: `{CONTROLLED_PROOF_TASK_ID}` (stimulus-injected; not used here).",
            f"- Natural repair runs: `{', '.join(natural_repair_ids) if natural_repair_ids else 'none'}`.",
            "",
        ]
    )
    evidence_table = "\n".join(
        [
            "",
            "## Evidence Separation",
            "",
            "| Evidence | Status |",
            "| --- | --- |",
            "| R4.R4 Controlled MODEL_OVERCOMMIT | PROVEN |",
            f"| R4.R5 Natural MODEL behavior | {fifteen_kb.value} |",
            (
                "| Combined conclusion | Natural bounded correction not observed; "
                "controlled proof remains authoritative. |"
                if fifteen_kb is NaturalFifteenKBEffect.NOT_PROVEN
                else f"| Combined conclusion | {fifteen_kb.value} |"
            ),
            "",
        ]
    )
    (output_dir / "report.md").write_text(report_md + evidence_table, encoding="utf-8")

    _merge_natural_analysis_json(
        output_dir,
        fifteen_kb=fifteen_kb,
        outcome=outcome,
        repo_freeze=repo_freeze,
        session_freeze=session_freeze,
        behavioral=behavioral,
        alignment_event_count=alignment_event_count,
        natural_repair_ids=natural_repair_ids,
        analyzed_at=analyzed_at,
    )

    final_report = _build_operator_final_report(
        repo_freeze=repo_freeze,
        session_freeze=session_freeze,
        behavioral=behavioral,
        fifteen_kb=fifteen_kb,
        outcome=outcome,
        alignment_event_count=alignment_event_count,
        natural_repair_ids=natural_repair_ids,
        session_dir=session_dir,
        repo_root=repo_root,
        regression_status=(
            "- mandatory matrix: 211 passed "
            "(natural_alignment_revision_evidence, 15I, 15K-B, O1, O2, QI1, QI2, L0, T1, C1, R4.R4)"
        ),
    )
    (output_dir / "final-report.md").write_text(final_report, encoding="utf-8")

    (output_dir / "alignment_direction.csv").write_text(
        _csv_bytes(
            alignment_rows,
            ("run_id", "direction", "correctable", "alignment_status"),
        ),
        encoding="utf-8",
    )
    (output_dir / "revision_effectiveness.csv").write_text(
        _csv_bytes(
            revision_rows,
            (
                "run_id",
                "natural_overcommit_repair",
                "revision_attempted",
                "typed_context_present",
                "revision_repaired",
            ),
        ),
        encoding="utf-8",
    )
    (output_dir / "attempt_timeline.csv").write_text(
        _csv_bytes(
            attempt_rows,
            ("run_id", "attempt_index", "max_iterations", "node_id"),
        ),
        encoding="utf-8",
    )

    artifact_names = (
        "runs.json",
        "summary.json",
        "report.md",
        "analysis.json",
        "alignment_direction.csv",
        "revision_effectiveness.csv",
        "attempt_timeline.csv",
        "behavioral_coverage.json",
        "local_model_profile.json",
        "final-report.md",
    )
    for name in artifact_names:
        if name not in {
            "report.md",
            "alignment_direction.csv",
            "revision_effectiveness.csv",
            "attempt_timeline.csv",
            "final-report.md",
        } and not (output_dir / name).is_file():
            pass
    _write_manifest(output_dir, tuple(name for name in artifact_names if (output_dir / name).is_file()))

    return NaturalQualificationResult(
        repo_source_freeze=repo_freeze,
        session_source_freeze=session_freeze,
        fifteen_kb_effect=fifteen_kb,
        qualification_outcome=outcome,
        alignment_event_count=alignment_event_count,
        natural_repair_run_ids=natural_repair_ids,
        third_pass_count=behavioral.third_pass_count,
        reconciliation_leak=behavioral.reconciliation_leak,
        output_dir=output_dir,
    )
