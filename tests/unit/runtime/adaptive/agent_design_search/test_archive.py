# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from intergrax.contracts.orchestration_enums import MergeStrategy
from intergrax.runtime.adaptive.agent_design_search.archive import (
    AgentDesignArchiveError,
    AgentDesignArchiveScopeError,
    InMemoryAgentDesignArchive,
)
from intergrax.runtime.adaptive.agent_design_search.contracts import (
    AdasW1BaselineRetentionProvenance,
    AdasW1LabStopEvidence,
    AgentCandidateEvaluationResult,
    AgentCandidateEvidenceBundle,
    AgentCandidateRecord,
    AgentCandidateStatus,
    ConfigurationDelta,
    SearchSelectionOutcome,
    W1TerminalEvidenceOutcome,
    adas_json_to_plain,
    compute_evidence_bundle_digest,
)
from intergrax.runtime.adaptive.agent_design_search.sqlite_archive import SQLiteAgentDesignArchive
from pydantic import ValidationError
from tests.unit.runtime.adaptive.agent_design_search.test_contracts import (
    _DIGEST,
    _NOW,
    sample_baseline,
    sample_evaluation_mapping,
    sample_record,
    sample_w1_baseline_retention_provenance,
    sample_w1_cost_report,
    sample_w1_execution_stop_gate,
    sample_w1_lab_stop_pre_execution,
)

_TENANT = "tenant-a"
_WORKSPACE = "ws-a"


@pytest.fixture(params=["memory", "sqlite"])
def archive_backend(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[InMemoryAgentDesignArchive | SQLiteAgentDesignArchive]:
    if request.param == "memory":
        yield InMemoryAgentDesignArchive()
        return
    db_path = tmp_path / f"adas-{request.node.name}.sqlite3"
    archive = SQLiteAgentDesignArchive(db_path)
    archive._connect()
    try:
        yield archive
    finally:
        archive.close()


def test_snapshot_append_history_and_latest(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    first = sample_record(sequence=0, status=AgentCandidateStatus.DRAFT)
    second = sample_record(
        sequence=1,
        status=AgentCandidateStatus.STATIC_GATE_PASSED,
        static_gate_result={"passed": True},
    )
    archive.append_candidate_snapshot(first, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(second, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    history = archive.get_candidate_history(first.candidate_id, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert [item.status for item in history] == [
        AgentCandidateStatus.DRAFT,
        AgentCandidateStatus.STATIC_GATE_PASSED,
    ]
    latest = archive.get_latest_candidate_snapshot(first.candidate_id, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert latest is not None and latest.sequence == 1


def test_idempotent_and_conflicting_duplicates(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    record = sample_record()
    archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    conflict = record.model_copy(update={"lineage_id": "line-2"})
    with pytest.raises(AgentDesignArchiveError):
        archive.append_candidate_snapshot(conflict, tenant_id=_TENANT, workspace_id=_WORKSPACE)


def test_invalid_transition_and_lineage_checks(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    parent = sample_record(candidate_id="parent-1", sequence=0)
    archive.append_candidate_snapshot(parent, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    child = sample_record(
        candidate_id="child-1",
        sequence=0,
    ).model_copy(
        update={
            "parent_candidate_id": "parent-1",
            "lineage_id": "line-child",
        }
    )
    archive.append_candidate_snapshot(child, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    bad = child.model_copy(update={"sequence": 1, "snapshot_id": "snap-bad", "status": AgentCandidateStatus.PROMOTED})
    with pytest.raises(AgentDesignArchiveError):
        archive.append_candidate_snapshot(bad, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    cycle = sample_record(candidate_id="cycle-1", sequence=0).model_copy(update={"parent_candidate_id": "cycle-1"})
    with pytest.raises(AgentDesignArchiveError):
        archive.append_candidate_snapshot(cycle, tenant_id=_TENANT, workspace_id=_WORKSPACE)


def test_cross_scope_rejection(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    record = sample_record()
    archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert (
        archive.get_latest_candidate_snapshot(record.candidate_id, tenant_id="other", workspace_id=_WORKSPACE) is None
    )
    foreign = sample_record().model_copy(update={"tenant_id": "tenant-b"})
    with pytest.raises(AgentDesignArchiveScopeError):
        archive.append_candidate_snapshot(foreign, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    with pytest.raises(AgentDesignArchiveScopeError):
        archive.get_latest_candidate_snapshot("cand-1", tenant_id=" ", workspace_id=_WORKSPACE)


def test_scoped_listing_parent_child_and_sealed_bundle(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    record = sample_record()
    archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    listed = archive.list_candidates_for_objective("obj-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(listed) == 1
    gate_fail = sample_record(
        sequence=1,
        status=AgentCandidateStatus.STATIC_GATE_FAILED,
        static_gate_result={"passed": False},
        rejection_reason="static gate rejected",
    )
    archive.append_candidate_snapshot(gate_fail, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    bundle = AgentCandidateEvidenceBundle(
        bundle_id="bundle-1",
        candidate_id=record.candidate_id,
        objective_id=record.objective_id,
        tenant_id=record.tenant_id,
        workspace_id=record.workspace_id,
        created_at=datetime.now(UTC),
        created_by="operator",
        baseline_configuration_digest="a" * 64,
        candidate_configuration_digest="b" * 64,
        candidate_record=gate_fail,
        static_gate_result={"passed": False},
        terminal_evidence_outcome=W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
        assembled_at=datetime.now(UTC),
        sealed=True,
    )
    digest = compute_evidence_bundle_digest(bundle)
    sealed = bundle.model_copy(update={"bundle_digest": digest})
    archive.append_sealed_evidence_bundle(sealed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert archive.get_sealed_evidence_bundle("bundle-1", tenant_id=_TENANT, workspace_id=_WORKSPACE) is not None
    with pytest.raises((AgentDesignArchiveError, ValidationError, ValueError)):
        archive.append_sealed_evidence_bundle(
            sealed.model_copy(update={"bundle_digest": "c" * 64}),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )


def test_mutation_escape_blocked(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    record = sample_record().model_copy(update={"agent_contract_snapshot": {"layer": {"k": "v"}}})
    stored = archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    with pytest.raises(TypeError):
        stored.agent_contract_snapshot["layer"]["k"] = "mutated"  # type: ignore[index]


def test_ia01_rejected_candidate_cannot_have_selected_sealed_evidence(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    draft = sample_record(sequence=0, status=AgentCandidateStatus.DRAFT)
    gate_failed = sample_record(
        sequence=1,
        status=AgentCandidateStatus.STATIC_GATE_FAILED,
        static_gate_result={"passed": False},
        rejection_reason="static gate rejected",
    )
    archive.append_candidate_snapshot(draft, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(gate_failed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    history_before = len(
        archive.get_candidate_history(draft.candidate_id, tenant_id=_TENANT, workspace_id=_WORKSPACE),
    )
    with pytest.raises((AgentDesignArchiveError, ValidationError, ValueError)):
        bundle = AgentCandidateEvidenceBundle(
            bundle_id="bundle-ia01",
            candidate_id=draft.candidate_id,
            objective_id=draft.objective_id,
            tenant_id=draft.tenant_id,
            workspace_id=draft.workspace_id,
            created_at=datetime.now(UTC),
            created_by="operator",
            baseline_configuration_digest=_DIGEST,
            candidate_configuration_digest=gate_failed.candidate_configuration_digest,
            candidate_record=gate_failed,
            static_gate_result={"passed": True},
            selection_outcome=SearchSelectionOutcome.SELECT_CANDIDATE,
            terminal_evidence_outcome=W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
            assembled_at=datetime.now(UTC),
            sealed=True,
        )
        digest = compute_evidence_bundle_digest(bundle)
        sealed = bundle.model_copy(update={"bundle_digest": digest})
        archive.append_sealed_evidence_bundle(sealed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    history_after = archive.get_candidate_history(draft.candidate_id, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(history_after) == history_before
    assert archive.get_sealed_evidence_bundle("bundle-ia01", tenant_id=_TENANT, workspace_id=_WORKSPACE) is None


def test_ia02_evaluated_snapshot_requires_valid_typed_evaluation(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    archive.append_candidate_snapshot(
        sample_record(sequence=0, status=AgentCandidateStatus.DRAFT),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    archive.append_candidate_snapshot(
        sample_record(
            sequence=1,
            status=AgentCandidateStatus.STATIC_GATE_PASSED,
            static_gate_result={"passed": True},
        ),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    history_before = archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    bad_evaluated = sample_record(
        sequence=2,
        status=AgentCandidateStatus.EVALUATED,
        static_gate_result={"passed": True},
        evaluation_result={
            "passed": "nonsense",
            "execution_ref": "exec-only",
        },
    )
    with pytest.raises((AgentDesignArchiveError, ValidationError, ValueError)):
        archive.append_candidate_snapshot(bad_evaluated, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    history_after = archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(history_after) == len(history_before)
    latest = archive.get_latest_candidate_snapshot("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert latest is not None and latest.sequence == 1


def test_q11_evidence_binds_exact_historical_snapshot(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    draft = sample_record(sequence=0, status=AgentCandidateStatus.DRAFT, snapshot_id="snap-eval-0")
    gate_passed = sample_record(
        sequence=1,
        status=AgentCandidateStatus.STATIC_GATE_PASSED,
        snapshot_id="snap-eval-1",
        static_gate_result={"passed": True},
    )
    eval_mapping = sample_evaluation_mapping(passed=True)
    evaluated = sample_record(
        sequence=2,
        status=AgentCandidateStatus.EVALUATED,
        snapshot_id="snap-eval-2",
        static_gate_result={"passed": True},
        evaluation_result=eval_mapping,
    )
    archive.append_candidate_snapshot(draft, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(gate_passed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(evaluated, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    bundle_eval = AgentCandidateEvaluationResult.model_validate(
        {"candidate_id": evaluated.candidate_id, **eval_mapping},
    )
    bundle = AgentCandidateEvidenceBundle(
        bundle_id="bundle-historical",
        candidate_id=draft.candidate_id,
        objective_id=draft.objective_id,
        tenant_id=draft.tenant_id,
        workspace_id=draft.workspace_id,
        search_run_id=evaluated.search_run_id,
        created_at=datetime.now(UTC),
        created_by="operator",
        baseline_configuration_digest=_DIGEST,
        candidate_configuration_digest=evaluated.candidate_configuration_digest,
        candidate_record=evaluated,
        static_gate_result={"passed": True},
        evaluation_result=bundle_eval,
        terminal_evidence_outcome=W1TerminalEvidenceOutcome.EVALUATION_COMPLETE,
        assembled_at=datetime.now(UTC),
        sealed=True,
    )
    digest = compute_evidence_bundle_digest(bundle)
    sealed = bundle.model_copy(update={"bundle_digest": digest})
    stored = archive.append_sealed_evidence_bundle(sealed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(
        sample_record(
            sequence=3,
            status=AgentCandidateStatus.REJECTED,
            snapshot_id="snap-eval-3",
            static_gate_result={"passed": True},
            evaluation_result=eval_mapping,
            rejection_reason="policy",
        ),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    again = archive.append_sealed_evidence_bundle(sealed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert again.model_dump(mode="json") == stored.model_dump(mode="json")
    assert again.bundle_digest == digest


def test_q07_rejects_successor_omitting_prior_evaluation(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    archive.append_candidate_snapshot(
        sample_record(sequence=0, status=AgentCandidateStatus.DRAFT),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    gate_passed = sample_record(
        sequence=1,
        status=AgentCandidateStatus.STATIC_GATE_PASSED,
        static_gate_result={"passed": True},
    )
    evaluated = sample_record(
        sequence=2,
        status=AgentCandidateStatus.EVALUATED,
        static_gate_result={"passed": True},
        evaluation_result=sample_evaluation_mapping(passed=True),
    )
    archive.append_candidate_snapshot(gate_passed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(evaluated, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    omitted = sample_record(
        sequence=3,
        status=AgentCandidateStatus.REJECTED,
        static_gate_result={"passed": True},
        rejection_reason="later",
        evaluation_result=None,
    )
    with pytest.raises(AgentDesignArchiveError):
        archive.append_candidate_snapshot(omitted, tenant_id=_TENANT, workspace_id=_WORKSPACE)


def test_q05_unknown_parent_same_as_foreign_parent(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    child = sample_record(candidate_id="child-2", sequence=0).model_copy(
        update={"parent_candidate_id": "missing-parent"},
    )
    with pytest.raises(AgentDesignArchiveError) as exc_missing:
        archive.append_candidate_snapshot(child, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    foreign_parent = sample_record(candidate_id="foreign-parent", sequence=0).model_copy(
        update={
            "tenant_id": "tenant-b",
            "envelope": sample_record().envelope.model_copy(update={"tenant_id": "tenant-b"}),
        }
    )
    archive.append_candidate_snapshot(foreign_parent, tenant_id="tenant-b", workspace_id=_WORKSPACE)
    cross_child = sample_record(candidate_id="cross-child", sequence=0).model_copy(
        update={"parent_candidate_id": "foreign-parent"},
    )
    with pytest.raises(AgentDesignArchiveError) as exc_foreign:
        archive.append_candidate_snapshot(cross_child, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert type(exc_missing.value) is type(exc_foreign.value)
    assert str(exc_missing.value) == str(exc_foreign.value)


def test_c02_contradictory_evaluation_replacement_rejected(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    archive.append_candidate_snapshot(sample_record(sequence=0), tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(
        sample_record(
            sequence=1,
            status=AgentCandidateStatus.STATIC_GATE_PASSED,
            static_gate_result={"passed": True},
        ),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    archive.append_candidate_snapshot(
        sample_record(
            sequence=2,
            status=AgentCandidateStatus.EVALUATED,
            static_gate_result={"passed": True},
            evaluation_result=sample_evaluation_mapping(passed=True),
        ),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    with pytest.raises(AgentDesignArchiveError, match="contradictory evaluation evidence"):
        archive.append_candidate_snapshot(
            sample_record(
                sequence=3,
                status=AgentCandidateStatus.REJECTED,
                static_gate_result={"passed": True},
                evaluation_result={
                    **sample_evaluation_mapping(passed=True),
                    "execution_ref": "exec-2",
                    "evaluation_ref": "eval-2",
                },
                rejection_reason="changed",
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )


def test_c02_static_gate_evidence_loss_rejected(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    archive.append_candidate_snapshot(sample_record(sequence=0), tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(
        sample_record(
            sequence=1,
            status=AgentCandidateStatus.STATIC_GATE_FAILED,
            static_gate_result={"passed": False, "reason": "fail"},
            rejection_reason="static",
        ),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    with pytest.raises(AgentDesignArchiveError, match="preserve static gate evidence"):
        archive.append_candidate_snapshot(
            sample_record(
                sequence=2,
                status=AgentCandidateStatus.ARCHIVED,
                static_gate_result=None,
                rejection_reason="static",
                archived_at=datetime.now(UTC),
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )


def _append_draft_gate_passed(archive: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive) -> AgentCandidateRecord:
    archive.append_candidate_snapshot(sample_record(sequence=0), tenant_id=_TENANT, workspace_id=_WORKSPACE)
    gate_passed = sample_record(
        sequence=1,
        status=AgentCandidateStatus.STATIC_GATE_PASSED,
        static_gate_result={"passed": True},
    )
    archive.append_candidate_snapshot(gate_passed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    return gate_passed


@pytest.mark.parametrize(
    "evaluation_result",
    [
        {"passed": False},
        {"passed": False, "execution_ref": "exec-1"},
        {"passed": False, "evaluation_ref": "eval-1"},
    ],
)
def test_ia03_incomplete_negative_evaluation_rejected(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
    evaluation_result: dict[str, object],
) -> None:
    archive = archive_backend
    _append_draft_gate_passed(archive)
    before = archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    bad = sample_record(
        sequence=2,
        status=AgentCandidateStatus.EVALUATED,
        static_gate_result={"passed": True},
        evaluation_result=evaluation_result,  # type: ignore[arg-type]
    )
    with pytest.raises((AgentDesignArchiveError, ValidationError, ValueError)):
        archive.append_candidate_snapshot(bad, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    after = archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(after) == len(before)
    latest = archive.get_latest_candidate_snapshot("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert latest is not None and latest.sequence == 1


def test_ia04_nonselecting_outcomes_on_static_gate_failed(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    archive.append_candidate_snapshot(sample_record(sequence=0), tenant_id=_TENANT, workspace_id=_WORKSPACE)
    gate_failed = sample_record(
        sequence=1,
        status=AgentCandidateStatus.STATIC_GATE_FAILED,
        static_gate_result={"passed": False, "reason": "policy"},
        rejection_reason="static gate rejected",
    )
    archive.append_candidate_snapshot(gate_failed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    base_bundle = dict(
        bundle_id="bundle-ia04",
        candidate_id=gate_failed.candidate_id,
        objective_id=gate_failed.objective_id,
        tenant_id=gate_failed.tenant_id,
        workspace_id=gate_failed.workspace_id,
        created_at=datetime.now(UTC),
        created_by="operator",
        baseline_configuration_digest=_DIGEST,
        candidate_configuration_digest=gate_failed.candidate_configuration_digest,
        candidate_record=gate_failed,
        static_gate_result={"passed": False, "reason": "policy"},
        terminal_evidence_outcome=W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
        assembled_at=datetime.now(UTC),
        sealed=True,
    )
    for index, outcome in enumerate((SearchSelectionOutcome.ABSTAIN, SearchSelectionOutcome.RETAIN_BASELINE)):
        kwargs = dict(base_bundle)
        kwargs["bundle_id"] = f"bundle-ia04-{outcome.value}"
        kwargs["selection_outcome"] = outcome
        if outcome is SearchSelectionOutcome.RETAIN_BASELINE:
            kwargs["baseline_evaluation_result"] = AgentCandidateEvaluationResult.model_validate(
                {
                    "candidate_id": gate_failed.candidate_id,
                    "baseline_agent_id": "baseline-1",
                    "passed": True,
                    "quality_score": 1.0,
                    "cost_normalized": 0.0,
                    "latency_ms": 1,
                    "total_tokens": 1,
                    "utility_score": 1.0,
                    "golden_pass_rate": 1.0,
                    "negative_pass_rate": 1.0,
                    "adversarial_pass_rate": 1.0,
                    "execution_ref": "base-exec",
                    "evaluation_ref": "base-eval",
                    "baseline_execution_ref": {"execution_id": "base-exec"},
                    "eval_run_ref": {"eval_run_id": "base-run"},
                    "eval_comparison_ref": {"comparison_id": "base-cmp"},
                    "evidence_complete": True,
                }
            )
            kwargs["w1_baseline_retention_provenance"] = AdasW1BaselineRetentionProvenance.model_validate(
                sample_w1_baseline_retention_provenance(
                    candidate_id=gate_failed.candidate_id,
                    evaluation_ref="base-eval",
                    baseline_execution_ref={"execution_id": "base-exec"},
                )
            )
        bundle = AgentCandidateEvidenceBundle(**kwargs)
        digest = compute_evidence_bundle_digest(bundle)
        archive.append_sealed_evidence_bundle(
            bundle.model_copy(update={"bundle_digest": digest}),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    with pytest.raises((AgentDesignArchiveError, ValidationError, ValueError)):
        select_bundle = AgentCandidateEvidenceBundle(
            **base_bundle,
            selection_outcome=SearchSelectionOutcome.SELECT_CANDIDATE,
        )
        digest = compute_evidence_bundle_digest(select_bundle)
        archive.append_sealed_evidence_bundle(
            select_bundle.model_copy(update={"bundle_digest": digest}),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )


def test_ia05_pre_gate_rejection_sealed_and_negative(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    draft = sample_record(sequence=0)
    stop = AdasW1LabStopEvidence.model_validate(
        sample_w1_lab_stop_pre_execution(sample_record(sequence=1, status=AgentCandidateStatus.REJECTED))
    )
    rejected = sample_record(
        sequence=1,
        status=AgentCandidateStatus.REJECTED,
        rejection_reason="budget admission refused",
        w1_lab_stop_evidence=stop,
    )
    archive.append_candidate_snapshot(draft, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(rejected, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    bundle = AgentCandidateEvidenceBundle(
        bundle_id="bundle-ia05",
        candidate_id=rejected.candidate_id,
        objective_id=rejected.objective_id,
        tenant_id=rejected.tenant_id,
        workspace_id=rejected.workspace_id,
        created_at=datetime.now(UTC),
        created_by="operator",
        baseline_configuration_digest=_DIGEST,
        candidate_configuration_digest=rejected.candidate_configuration_digest,
        candidate_record=rejected,
        w1_lab_stop_evidence=stop,
        terminal_evidence_outcome=W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
        selection_outcome=SearchSelectionOutcome.ABSTAIN,
        assembled_at=datetime.now(UTC),
        sealed=True,
    )
    digest = compute_evidence_bundle_digest(bundle)
    archive.append_sealed_evidence_bundle(
        bundle.model_copy(update={"bundle_digest": digest}), tenant_id=_TENANT, workspace_id=_WORKSPACE
    )
    bad = bundle.model_copy(
        update={
            "bundle_id": "bundle-ia05-bad",
            "static_gate_result": {"passed": True},
        }
    )
    with pytest.raises((AgentDesignArchiveError, ValidationError, ValueError)):
        bad_digest = compute_evidence_bundle_digest(bad)
        archive.append_sealed_evidence_bundle(
            bad.model_copy(update={"bundle_digest": bad_digest}), tenant_id=_TENANT, workspace_id=_WORKSPACE
        )


@pytest.mark.parametrize(
    "cost_overrides",
    [
        {"completeness": "partial"},
        {"tokens_consumed": None},
        {"monetary_consumed": None},
        {"root_executions_consumed": None},
        {"candidate_id": "foreign"},
    ],
)
def test_ia06_incomplete_accounting_rejects_selection(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
    cost_overrides: dict[str, object],
) -> None:
    archive = archive_backend
    _append_draft_gate_passed(archive)
    evaluated = sample_record(
        sequence=2,
        status=AgentCandidateStatus.EVALUATED,
        static_gate_result={"passed": True},
        evaluation_result=sample_evaluation_mapping(passed=True),
    )
    archive.append_candidate_snapshot(evaluated, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    cost = sample_w1_cost_report(**cost_overrides)  # type: ignore[arg-type]
    with pytest.raises((AgentDesignArchiveError, ValidationError, ValueError)):
        bundle = AgentCandidateEvidenceBundle(
            bundle_id="bundle-ia06-bad",
            candidate_id=evaluated.candidate_id,
            objective_id=evaluated.objective_id,
            tenant_id=evaluated.tenant_id,
            workspace_id=evaluated.workspace_id,
            created_at=datetime.now(UTC),
            created_by="operator",
            baseline_configuration_digest=_DIGEST,
            candidate_configuration_digest=evaluated.candidate_configuration_digest,
            candidate_record=evaluated,
            static_gate_result={"passed": True},
            evaluation_result=AgentCandidateEvaluationResult.model_validate(
                {"candidate_id": evaluated.candidate_id, **sample_evaluation_mapping(passed=True)},
            ),
            cost_report=cost,
            terminal_evidence_outcome=W1TerminalEvidenceOutcome.EVALUATION_COMPLETE,
            selection_outcome=SearchSelectionOutcome.SELECT_CANDIDATE,
            assembled_at=datetime.now(UTC),
            sealed=True,
        )
        digest = compute_evidence_bundle_digest(bundle)
        archive.append_sealed_evidence_bundle(
            bundle.model_copy(update={"bundle_digest": digest}),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    assert archive.get_sealed_evidence_bundle("bundle-ia06-bad", tenant_id=_TENANT, workspace_id=_WORKSPACE) is None


def test_ia06_complete_accounting_allows_zero_monetary(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    _append_draft_gate_passed(archive)
    evaluated = sample_record(
        sequence=2,
        status=AgentCandidateStatus.EVALUATED,
        static_gate_result={"passed": True},
        evaluation_result=sample_evaluation_mapping(passed=True),
    )
    archive.append_candidate_snapshot(evaluated, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    bundle = AgentCandidateEvidenceBundle(
        bundle_id="bundle-ia06-ok",
        candidate_id=evaluated.candidate_id,
        objective_id=evaluated.objective_id,
        tenant_id=evaluated.tenant_id,
        workspace_id=evaluated.workspace_id,
        created_at=datetime.now(UTC),
        created_by="operator",
        baseline_configuration_digest=_DIGEST,
        candidate_configuration_digest=evaluated.candidate_configuration_digest,
        candidate_record=evaluated,
        static_gate_result={"passed": True},
        evaluation_result=AgentCandidateEvaluationResult.model_validate(
            {"candidate_id": evaluated.candidate_id, **sample_evaluation_mapping(passed=True)},
        ),
        cost_report=sample_w1_cost_report(monetary_consumed=0.0),
        terminal_evidence_outcome=W1TerminalEvidenceOutcome.EVALUATION_COMPLETE,
        selection_outcome=SearchSelectionOutcome.SELECT_CANDIDATE,
        assembled_at=datetime.now(UTC),
        sealed=True,
    )
    digest = compute_evidence_bundle_digest(bundle)
    archive.append_sealed_evidence_bundle(
        bundle.model_copy(update={"bundle_digest": digest}), tenant_id=_TENANT, workspace_id=_WORKSPACE
    )


@pytest.mark.parametrize("kind", ["execution_failure", "execution_cancellation"])
def test_ia07_failed_or_cancelled_execution(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
    kind: str,
) -> None:
    archive = archive_backend
    _append_draft_gate_passed(archive)
    gate_passed = archive.get_latest_candidate_snapshot("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert gate_passed is not None
    rejected = sample_record(
        sequence=2,
        status=AgentCandidateStatus.REJECTED,
        static_gate_result=sample_w1_execution_stop_gate(gate_passed, kind=kind),
        rejection_reason="execution did not complete",
    )
    archive.append_candidate_snapshot(rejected, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    stop = AdasW1LabStopEvidence.model_validate(
        sample_w1_execution_stop_gate(gate_passed, kind=kind)["w1_lab_stop_evidence"]  # type: ignore[index]
    )
    base = dict(
        bundle_id=f"bundle-ia07-{kind}",
        candidate_id=rejected.candidate_id,
        objective_id=rejected.objective_id,
        tenant_id=rejected.tenant_id,
        workspace_id=rejected.workspace_id,
        created_at=datetime.now(UTC),
        created_by="operator",
        baseline_configuration_digest=_DIGEST,
        candidate_configuration_digest=rejected.candidate_configuration_digest,
        candidate_record=rejected,
        static_gate_result=sample_w1_execution_stop_gate(gate_passed, kind=kind),
        w1_lab_stop_evidence=stop,
        terminal_evidence_outcome=W1TerminalEvidenceOutcome.EXECUTION_FAILED_OR_CANCELLED,
        assembled_at=datetime.now(UTC),
        sealed=True,
    )
    for outcome in (SearchSelectionOutcome.ABSTAIN, SearchSelectionOutcome.RETAIN_BASELINE):
        kwargs = dict(base)
        kwargs["bundle_id"] = f"bundle-ia07-{kind}-{outcome.value}"
        kwargs["selection_outcome"] = outcome
        if outcome is SearchSelectionOutcome.RETAIN_BASELINE:
            kwargs["baseline_evaluation_result"] = AgentCandidateEvaluationResult.model_validate(
                {
                    "candidate_id": rejected.candidate_id,
                    "baseline_agent_id": "baseline-1",
                    "passed": True,
                    "quality_score": 1.0,
                    "cost_normalized": 0.0,
                    "latency_ms": 1,
                    "total_tokens": 1,
                    "utility_score": 1.0,
                    "golden_pass_rate": 1.0,
                    "negative_pass_rate": 1.0,
                    "adversarial_pass_rate": 1.0,
                    "execution_ref": "base-exec",
                    "evaluation_ref": "base-eval",
                    "baseline_execution_ref": {"execution_id": "base-exec"},
                    "eval_run_ref": {"eval_run_id": "base-run"},
                    "eval_comparison_ref": {"comparison_id": "base-cmp"},
                    "evidence_complete": True,
                }
            )
            kwargs["w1_baseline_retention_provenance"] = AdasW1BaselineRetentionProvenance.model_validate(
                sample_w1_baseline_retention_provenance(
                    candidate_id=rejected.candidate_id,
                    evaluation_ref="base-eval",
                    baseline_execution_ref={"execution_id": "base-exec"},
                )
            )
        bundle = AgentCandidateEvidenceBundle(**kwargs)
        digest = compute_evidence_bundle_digest(bundle)
        archive.append_sealed_evidence_bundle(
            bundle.model_copy(update={"bundle_digest": digest}), tenant_id=_TENANT, workspace_id=_WORKSPACE
        )
    with pytest.raises((AgentDesignArchiveError, ValidationError, ValueError)):
        bad = AgentCandidateEvidenceBundle(**base, selection_outcome=SearchSelectionOutcome.SELECT_CANDIDATE)
        digest = compute_evidence_bundle_digest(bad)
        archive.append_sealed_evidence_bundle(
            bad.model_copy(update={"bundle_digest": digest}), tenant_id=_TENANT, workspace_id=_WORKSPACE
        )


def test_ia07_rejection_without_typed_execution_evidence_rejected(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    _append_draft_gate_passed(archive)
    with pytest.raises(AgentDesignArchiveError):
        archive.append_candidate_snapshot(
            sample_record(
                sequence=2,
                status=AgentCandidateStatus.REJECTED,
                static_gate_result={"passed": True},
                rejection_reason="free text only",
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )


def test_q08_compatible_additive_nested_evidence(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    archive.append_candidate_snapshot(sample_record(sequence=0), tenant_id=_TENANT, workspace_id=_WORKSPACE)
    mapping = sample_evaluation_mapping(passed=True)
    archive.append_candidate_snapshot(
        sample_record(
            sequence=1,
            status=AgentCandidateStatus.STATIC_GATE_PASSED,
            static_gate_result={"passed": True, "detail": {"code": "ok"}},
        ),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    archive.append_candidate_snapshot(
        sample_record(
            sequence=2,
            status=AgentCandidateStatus.EVALUATED,
            static_gate_result={"passed": True, "detail": {"code": "ok"}},
            evaluation_result=mapping,
        ),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    archive.append_candidate_snapshot(
        sample_record(
            sequence=3,
            status=AgentCandidateStatus.REJECTED,
            static_gate_result={
                "passed": True,
                "detail": {"code": "ok", "meta": {"layer": "extra"}},
            },
            evaluation_result=mapping,
            rejection_reason="policy",
        ),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )


# --- Independent audit cases (IA-08..IA-13) ported from architect suite ---
_IA_S = {"tenant_id": _TENANT, "workspace_id": _WORKSPACE}
_IA_ERR = (ValidationError, ValueError, AgentDesignArchiveError)


@pytest.fixture(params=["memory", "sqlite"])
def audit_archive(request, tmp_path):  # noqa: ANN001
    archive = (
        InMemoryAgentDesignArchive() if request.param == "memory" else SQLiteAgentDesignArchive(tmp_path / "audit.db")
    )
    yield archive
    if isinstance(archive, SQLiteAgentDesignArchive):
        archive.close()


def _ia_append(archive, record):  # noqa: ANN001
    return archive.append_candidate_snapshot(record, **_IA_S)


def _ia_start(archive):  # noqa: ANN001
    return _ia_append(archive, sample_record())


def _ia_passgate(archive):  # noqa: ANN001
    _ia_start(archive)
    return _ia_append(
        archive,
        sample_record(
            sequence=1,
            status=AgentCandidateStatus.STATIC_GATE_PASSED,
            static_gate_result={"passed": True},
        ),
    )


def _ia_eval_record(archive):  # noqa: ANN001
    _ia_passgate(archive)
    return _ia_append(
        archive,
        sample_record(
            sequence=2,
            status=AgentCandidateStatus.EVALUATED,
            static_gate_result={"passed": True},
            evaluation_result=sample_evaluation_mapping(passed=True),
        ),
    )


def _ia_bundle(record, **kw):  # noqa: ANN001
    data = dict(
        bundle_id="audit-bundle",
        candidate_id=record.candidate_id,
        objective_id=record.objective_id,
        tenant_id=record.tenant_id,
        workspace_id=record.workspace_id,
        search_run_id=record.search_run_id,
        created_at=_NOW,
        created_by="operator",
        baseline_configuration_digest=record.baseline.baseline_configuration_digest,
        candidate_configuration_digest=record.candidate_configuration_digest,
        candidate_record=record,
        assembled_at=_NOW,
        sealed=True,
    )
    data.update(kw)
    return AgentCandidateEvidenceBundle(**data)


def _ia_persist(archive, bundle):  # noqa: ANN001
    return archive.append_sealed_evidence_bundle(bundle, **_IA_S)


def _ia_select(archive, record, cost):  # noqa: ANN001
    ev = AgentCandidateEvaluationResult(candidate_id=record.candidate_id, **sample_evaluation_mapping(passed=True))
    bundle = _ia_bundle(
        record,
        static_gate_result={"passed": True},
        evaluation_result=ev,
        terminal_evidence_outcome=W1TerminalEvidenceOutcome.EVALUATION_COMPLETE,
        selection_outcome=SearchSelectionOutcome.SELECT_CANDIDATE,
        cost_report=cost,
    )
    digest = compute_evidence_bundle_digest(bundle)
    return _ia_persist(archive, bundle.model_copy(update={"bundle_digest": digest}))


@pytest.mark.parametrize(
    "bad",
    [
        {"tokens_consumed": -1},
        {"monetary_consumed": -1},
        {"root_executions_consumed": 3},
        {"root_executions_consumed": True},
        {"tokens_consumed": 0},
    ],
)
def test_ia08_selection_accounting_rejects_invalid_values(audit_archive, bad):
    r = _ia_eval_record(audit_archive)
    with pytest.raises(_IA_ERR):
        _ia_select(
            audit_archive, r, sample_w1_cost_report(**({"tokens_consumed": 100, "root_executions_consumed": 2} | bad))
        )
    assert audit_archive.get_sealed_evidence_bundle("audit-bundle", **_IA_S) is None


def test_ia09_retain_baseline_requires_passed_baseline(audit_archive):
    _ia_start(audit_archive)
    r = _ia_append(
        audit_archive,
        sample_record(
            sequence=1,
            status=AgentCandidateStatus.STATIC_GATE_FAILED,
            static_gate_result={"passed": False},
            rejection_reason="static rejected",
        ),
    )
    baseline = AgentCandidateEvaluationResult(
        candidate_id=r.candidate_id,
        baseline_agent_id="baseline-1",
        passed=False,
        execution_ref="base-exec",
        evaluation_ref="base-eval",
        failure_reasons=("baseline rejected",),
    )
    with pytest.raises(_IA_ERR):
        _ia_persist(
            audit_archive,
            _ia_bundle(
                r,
                static_gate_result={"passed": False},
                terminal_evidence_outcome=W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
                selection_outcome=SearchSelectionOutcome.RETAIN_BASELINE,
                baseline_evaluation_result=baseline,
            ),
        )
    assert audit_archive.get_sealed_evidence_bundle("audit-bundle", **_IA_S) is None


def test_ia10_bundle_stop_must_match_archived_stop(audit_archive):
    g = _ia_passgate(audit_archive)
    gate = sample_w1_execution_stop_gate(g, kind="execution_failure")
    r = _ia_append(
        audit_archive,
        sample_record(
            sequence=2,
            status=AgentCandidateStatus.REJECTED,
            static_gate_result=gate,
            rejection_reason="execution failed",
        ),
    )
    conflicting = AdasW1LabStopEvidence.model_validate(
        {
            **gate["w1_lab_stop_evidence"],
            "kind": "execution_cancellation",
            "execution_ref": {"execution_id": "other-execution"},
        }
    )
    with pytest.raises(_IA_ERR):
        _ia_persist(
            audit_archive,
            _ia_bundle(
                r,
                static_gate_result=gate,
                w1_lab_stop_evidence=conflicting,
                terminal_evidence_outcome=W1TerminalEvidenceOutcome.EXECUTION_FAILED_OR_CANCELLED,
                selection_outcome=SearchSelectionOutcome.ABSTAIN,
            ),
        )
    assert audit_archive.get_sealed_evidence_bundle("audit-bundle", **_IA_S) is None


def test_ia11_stop_scope_bound_at_archive_ingress(audit_archive):
    g = _ia_passgate(audit_archive)
    gate = sample_w1_execution_stop_gate(g)
    gate["w1_lab_stop_evidence"]["tenant_id"] = "foreign"
    with pytest.raises(_IA_ERR):
        _ia_append(
            audit_archive,
            sample_record(
                sequence=2,
                status=AgentCandidateStatus.REJECTED,
                static_gate_result=gate,
                rejection_reason="execution failed",
            ),
        )
    assert len(audit_archive.get_candidate_history("cand-1", **_IA_S)) == 2


def test_ia12_draft_must_not_contain_completed_evaluation(audit_archive):
    with pytest.raises(_IA_ERR):
        _ia_append(
            audit_archive,
            sample_record(
                static_gate_result={"passed": True}, evaluation_result=sample_evaluation_mapping(passed=True)
            ),
        )
    assert audit_archive.get_candidate_history("cand-1", **_IA_S) == []


def test_ia13_archived_must_not_gain_new_selection(audit_archive):
    r = _ia_eval_record(audit_archive)
    archived = _ia_append(
        audit_archive,
        sample_record(
            sequence=3,
            status=AgentCandidateStatus.ARCHIVED,
            static_gate_result={"passed": True},
            evaluation_result=sample_evaluation_mapping(passed=True),
            archived_at=_NOW,
        ),
    )
    ev = AgentCandidateEvaluationResult(candidate_id=r.candidate_id, **sample_evaluation_mapping(passed=True))
    with pytest.raises(_IA_ERR):
        b = _ia_bundle(
            archived,
            static_gate_result={"passed": True},
            evaluation_result=ev,
            terminal_evidence_outcome=W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
            selection_outcome=SearchSelectionOutcome.SELECT_CANDIDATE,
            cost_report=sample_w1_cost_report(tokens_consumed=100, root_executions_consumed=2),
        )
        _ia_persist(audit_archive, b.model_copy(update={"bundle_digest": compute_evidence_bundle_digest(b)}))
    assert audit_archive.get_sealed_evidence_bundle("audit-bundle", **_IA_S) is None


def test_control_valid_evaluated_selection(audit_archive):
    r = _ia_eval_record(audit_archive)
    result = _ia_select(
        audit_archive, r, sample_w1_cost_report(tokens_consumed=100, root_executions_consumed=2, monetary_consumed=0.0)
    )
    assert result.selection_outcome is SearchSelectionOutcome.SELECT_CANDIDATE
    assert result.bundle_digest is not None
    assert audit_archive.get_sealed_evidence_bundle("audit-bundle", **_IA_S) == result


def test_q09_actual_parent_child_lineage_queries(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    parent = sample_record(candidate_id="parent-1", sequence=0)
    archive.append_candidate_snapshot(parent, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    child = sample_record(candidate_id="child-1", sequence=0).model_copy(
        update={"parent_candidate_id": "parent-1", "lineage_id": "line-child"},
    )
    archive.append_candidate_snapshot(child, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    resolved_parent = archive.get_parent_candidate("child-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert resolved_parent is not None and resolved_parent.candidate_id == "parent-1"
    children = archive.get_child_candidates("parent-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert [c.candidate_id for c in children] == ["child-1"]


def test_q06_immutable_successor_positive_control(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    archive.append_candidate_snapshot(
        sample_record(sequence=0, status=AgentCandidateStatus.DRAFT),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    gate_passed = sample_record(
        sequence=1,
        status=AgentCandidateStatus.STATIC_GATE_PASSED,
        static_gate_result={"passed": True},
        snapshot_id="snap-gate-pass",
    )
    persisted = archive.append_candidate_snapshot(gate_passed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert persisted.status is AgentCandidateStatus.STATIC_GATE_PASSED
    assert len(archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)) == 2


_IMMUTABLE_SUCCESSOR_FIELDS: list[tuple[str, object]] = [
    ("objective_id", "obj-mutated"),
    ("search_run_id", "run-mutated"),
    ("lineage_id", "line-mutated"),
    ("parent_candidate_id", "parent-mutated"),
    ("baseline", sample_baseline().model_copy(update={"baseline_version_id": "baseline-v2"})),
    ("baseline", sample_baseline().model_copy(update={"baseline_configuration_digest": "f" * 64})),
    ("candidate_configuration_digest", "f" * 64),
    (
        "configuration_delta",
        ConfigurationDelta(prior_value=MergeStrategy.LAST_WINS, replacement_value=MergeStrategy.CONCAT),
    ),
    ("created_by", "other-operator"),
    ("created_at", _NOW + timedelta(minutes=5)),
    ("agent_contract_snapshot", {"role": "mutated-worker"}),
]


@pytest.mark.parametrize(
    "field_name,new_value",
    _IMMUTABLE_SUCCESSOR_FIELDS,
    ids=[
        "objective_id",
        "search_run_id",
        "lineage_id",
        "parent_candidate_id",
        "baseline_identity",
        "baseline_digest",
        "candidate_configuration_digest",
        "configuration_delta",
        "created_by",
        "created_at",
        "agent_contract_snapshot",
    ],
)
def test_q06_immutable_identity_fields_rejected_on_successor(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
    field_name: str,
    new_value: object,
) -> None:
    archive = archive_backend
    if field_name == "parent_candidate_id":
        archive.append_candidate_snapshot(
            sample_record(candidate_id=str(new_value), sequence=0, status=AgentCandidateStatus.DRAFT),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    archive.append_candidate_snapshot(
        sample_record(sequence=0, status=AgentCandidateStatus.DRAFT),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    valid_successor = sample_record(
        sequence=1,
        status=AgentCandidateStatus.STATIC_GATE_PASSED,
        static_gate_result={"passed": True},
        snapshot_id="snap-gate-pass",
    )
    mutated = valid_successor.model_copy(update={field_name: new_value})
    with pytest.raises(
        AgentDesignArchiveError,
        match="immutable candidate identity|immutable configuration lineage",
    ):
        archive.append_candidate_snapshot(mutated, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    history = archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(history) == 1
    latest = archive.get_latest_candidate_snapshot("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert latest is not None and latest.status is AgentCandidateStatus.DRAFT


def test_q06_candidate_identity_rebinding_rejected(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    archive = archive_backend
    archive.append_candidate_snapshot(
        sample_record(sequence=0, status=AgentCandidateStatus.DRAFT),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    rebound = sample_record(
        sequence=0,
        status=AgentCandidateStatus.DRAFT,
        snapshot_id="snap-foreign",
    ).model_copy(
        update={
            "tenant_id": "tenant-b",
            "envelope": sample_record().envelope.model_copy(update={"tenant_id": "tenant-b"}),
        }
    )
    with pytest.raises(AgentDesignArchiveScopeError):
        archive.append_candidate_snapshot(rebound, tenant_id="tenant-b", workspace_id=_WORKSPACE)


@pytest.mark.parametrize(
    "edge_case",
    [
        "draft_to_gate_passed",
        "draft_to_gate_failed",
        "draft_to_rejected",
        "gate_passed_to_evaluated_pos",
        "gate_passed_to_evaluated_neg",
        "gate_passed_to_rejected",
        "evaluated_to_rejected",
        "evaluated_to_archived",
        "gate_failed_to_archived",
        "rejected_to_archived",
    ],
)
def test_c08_w1_permitted_lifecycle_chain(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
    edge_case: str,
) -> None:
    archive = archive_backend
    draft = sample_record(sequence=0, status=AgentCandidateStatus.DRAFT)
    archive.append_candidate_snapshot(draft, tenant_id=_TENANT, workspace_id=_WORKSPACE)

    if edge_case == "draft_to_gate_passed":
        rec = archive.append_candidate_snapshot(
            sample_record(
                sequence=1, status=AgentCandidateStatus.STATIC_GATE_PASSED, static_gate_result={"passed": True}
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        assert rec.status is AgentCandidateStatus.STATIC_GATE_PASSED
    elif edge_case == "draft_to_gate_failed":
        rec = archive.append_candidate_snapshot(
            sample_record(
                sequence=1,
                status=AgentCandidateStatus.STATIC_GATE_FAILED,
                static_gate_result={"passed": False},
                rejection_reason="static failed",
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        assert rec.status is AgentCandidateStatus.STATIC_GATE_FAILED
    elif edge_case == "draft_to_rejected":
        stop = AdasW1LabStopEvidence.model_validate(sample_w1_lab_stop_pre_execution(draft))
        rec = archive.append_candidate_snapshot(
            sample_record(
                sequence=1,
                status=AgentCandidateStatus.REJECTED,
                w1_lab_stop_evidence=stop,
                rejection_reason="pre-gate refusal",
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        assert rec.status is AgentCandidateStatus.REJECTED
    elif edge_case in {"gate_passed_to_evaluated_pos", "gate_passed_to_evaluated_neg"}:
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1, status=AgentCandidateStatus.STATIC_GATE_PASSED, static_gate_result={"passed": True}
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        passed_flag = edge_case == "gate_passed_to_evaluated_pos"
        ev_map = (
            sample_evaluation_mapping(passed=True)
            if passed_flag
            else {"passed": False, "execution_ref": "exec-1", "evaluation_ref": "eval-1", "failure_reasons": ["policy"]}
        )
        rec = archive.append_candidate_snapshot(
            sample_record(
                sequence=2,
                status=AgentCandidateStatus.EVALUATED,
                static_gate_result={"passed": True},
                evaluation_result=ev_map,
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        assert rec.status is AgentCandidateStatus.EVALUATED
    elif edge_case == "gate_passed_to_rejected":
        gp = archive.append_candidate_snapshot(
            sample_record(
                sequence=1, status=AgentCandidateStatus.STATIC_GATE_PASSED, static_gate_result={"passed": True}
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        stop = AdasW1LabStopEvidence.model_validate(sample_w1_execution_stop_gate(gp)["w1_lab_stop_evidence"])
        rec = archive.append_candidate_snapshot(
            sample_record(
                sequence=2,
                status=AgentCandidateStatus.REJECTED,
                static_gate_result={"passed": True},
                w1_lab_stop_evidence=stop,
                rejection_reason="execution failure",
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        assert rec.status is AgentCandidateStatus.REJECTED
    elif edge_case in {"evaluated_to_rejected", "evaluated_to_archived"}:
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1, status=AgentCandidateStatus.STATIC_GATE_PASSED, static_gate_result={"passed": True}
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        archive.append_candidate_snapshot(
            sample_record(
                sequence=2,
                status=AgentCandidateStatus.EVALUATED,
                static_gate_result={"passed": True},
                evaluation_result=sample_evaluation_mapping(passed=True),
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        target = (
            AgentCandidateStatus.REJECTED if edge_case == "evaluated_to_rejected" else AgentCandidateStatus.ARCHIVED
        )
        rec = archive.append_candidate_snapshot(
            sample_record(
                sequence=3,
                status=target,
                static_gate_result={"passed": True},
                evaluation_result=sample_evaluation_mapping(passed=True),
                rejection_reason="operator rejected" if target is AgentCandidateStatus.REJECTED else None,
                archived_at=_NOW if target is AgentCandidateStatus.ARCHIVED else None,
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        assert rec.status is target
    elif edge_case == "gate_failed_to_archived":
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1,
                status=AgentCandidateStatus.STATIC_GATE_FAILED,
                static_gate_result={"passed": False},
                rejection_reason="static failed",
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        rec = archive.append_candidate_snapshot(
            sample_record(
                sequence=2,
                status=AgentCandidateStatus.ARCHIVED,
                static_gate_result={"passed": False},
                rejection_reason="static failed",
                archived_at=_NOW,
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        assert rec.status is AgentCandidateStatus.ARCHIVED
    elif edge_case == "rejected_to_archived":
        stop = AdasW1LabStopEvidence.model_validate(sample_w1_lab_stop_pre_execution(draft))
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1,
                status=AgentCandidateStatus.REJECTED,
                w1_lab_stop_evidence=stop,
                rejection_reason="pre-gate refusal",
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        rec = archive.append_candidate_snapshot(
            sample_record(
                sequence=2,
                status=AgentCandidateStatus.ARCHIVED,
                w1_lab_stop_evidence=stop,
                rejection_reason="pre-gate refusal",
                archived_at=_NOW,
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        assert rec.status is AgentCandidateStatus.ARCHIVED


@pytest.mark.parametrize(
    "forbidden_target",
    [
        AgentCandidateStatus.SCAFFOLDED,
        AgentCandidateStatus.SHADOW,
        AgentCandidateStatus.CANARY,
        AgentCandidateStatus.PROMOTED,
        AgentCandidateStatus.RETIRED,
    ],
)
def test_c08_w1_forbidden_terminal_targets_rejected(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
    forbidden_target: AgentCandidateStatus,
) -> None:
    archive = archive_backend
    archive.append_candidate_snapshot(
        sample_record(sequence=0, status=AgentCandidateStatus.DRAFT), tenant_id=_TENANT, workspace_id=_WORKSPACE
    )
    archive.append_candidate_snapshot(
        sample_record(sequence=1, status=AgentCandidateStatus.STATIC_GATE_PASSED, static_gate_result={"passed": True}),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    before = len(archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE))
    bad = sample_record(sequence=2, status=forbidden_target)
    with pytest.raises(AgentDesignArchiveError):
        archive.append_candidate_snapshot(bad, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)) == before


@pytest.mark.parametrize(
    "from_status,to_status",
    [
        (AgentCandidateStatus.DRAFT, AgentCandidateStatus.EVALUATED),
        (AgentCandidateStatus.DRAFT, AgentCandidateStatus.ARCHIVED),
        (AgentCandidateStatus.STATIC_GATE_PASSED, AgentCandidateStatus.STATIC_GATE_FAILED),
        (AgentCandidateStatus.STATIC_GATE_PASSED, AgentCandidateStatus.ARCHIVED),
        (AgentCandidateStatus.STATIC_GATE_FAILED, AgentCandidateStatus.STATIC_GATE_PASSED),
        (AgentCandidateStatus.STATIC_GATE_FAILED, AgentCandidateStatus.EVALUATED),
        (AgentCandidateStatus.STATIC_GATE_FAILED, AgentCandidateStatus.REJECTED),
        (AgentCandidateStatus.EVALUATED, AgentCandidateStatus.STATIC_GATE_PASSED),
        (AgentCandidateStatus.REJECTED, AgentCandidateStatus.EVALUATED),
        (AgentCandidateStatus.ARCHIVED, AgentCandidateStatus.EVALUATED),
        (AgentCandidateStatus.ARCHIVED, AgentCandidateStatus.REJECTED),
    ],
)
def test_c08_w1_forbidden_lifecycle_transitions_rejected(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
    from_status: AgentCandidateStatus,
    to_status: AgentCandidateStatus,
) -> None:
    archive = archive_backend
    draft = sample_record(sequence=0, status=AgentCandidateStatus.DRAFT)
    archive.append_candidate_snapshot(draft, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    if from_status is AgentCandidateStatus.STATIC_GATE_PASSED:
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1, status=AgentCandidateStatus.STATIC_GATE_PASSED, static_gate_result={"passed": True}
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    elif from_status is AgentCandidateStatus.STATIC_GATE_FAILED:
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1,
                status=AgentCandidateStatus.STATIC_GATE_FAILED,
                static_gate_result={"passed": False},
                rejection_reason="static failed",
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    elif from_status is AgentCandidateStatus.EVALUATED:
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1, status=AgentCandidateStatus.STATIC_GATE_PASSED, static_gate_result={"passed": True}
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        archive.append_candidate_snapshot(
            sample_record(
                sequence=2,
                status=AgentCandidateStatus.EVALUATED,
                static_gate_result={"passed": True},
                evaluation_result=sample_evaluation_mapping(passed=True),
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    elif from_status is AgentCandidateStatus.REJECTED:
        stop = AdasW1LabStopEvidence.model_validate(sample_w1_lab_stop_pre_execution(draft))
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1,
                status=AgentCandidateStatus.REJECTED,
                w1_lab_stop_evidence=stop,
                rejection_reason="pre-gate refusal",
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    elif from_status is AgentCandidateStatus.ARCHIVED:
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1,
                status=AgentCandidateStatus.STATIC_GATE_FAILED,
                static_gate_result={"passed": False},
                rejection_reason="static failed",
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        archive.append_candidate_snapshot(
            sample_record(
                sequence=2,
                status=AgentCandidateStatus.ARCHIVED,
                static_gate_result={"passed": False},
                rejection_reason="static failed",
                archived_at=_NOW,
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )

    before = len(archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE))
    latest = archive.get_latest_candidate_snapshot("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert latest is not None
    latest_gate = adas_json_to_plain(latest.static_gate_result) if latest.static_gate_result is not None else None
    latest_eval = adas_json_to_plain(latest.evaluation_result) if latest.evaluation_result is not None else None
    if to_status is AgentCandidateStatus.EVALUATED:
        bad = sample_record(
            sequence=before,
            status=to_status,
            static_gate_result=latest_gate or {"passed": True},
            evaluation_result=sample_evaluation_mapping(passed=True),
            rejection_reason=latest.rejection_reason,
            w1_lab_stop_evidence=latest.w1_lab_stop_evidence,
            archived_at=latest.archived_at,
        )
    elif to_status is AgentCandidateStatus.STATIC_GATE_FAILED:
        bad = sample_record(
            sequence=before,
            status=to_status,
            static_gate_result={"passed": False},
            rejection_reason="static failed",
        )
    elif to_status is AgentCandidateStatus.STATIC_GATE_PASSED:
        bad = sample_record(
            sequence=before,
            status=to_status,
            static_gate_result={"passed": True},
            evaluation_result=latest_eval,
            rejection_reason=latest.rejection_reason,
        )
    elif to_status is AgentCandidateStatus.REJECTED:
        bad = sample_record(
            sequence=before,
            status=to_status,
            static_gate_result=latest_gate,
            rejection_reason=latest.rejection_reason or "rejected",
            archived_at=latest.archived_at,
        )
    else:
        bad = sample_record(
            sequence=before,
            status=to_status,
            static_gate_result=latest_gate,
            rejection_reason=latest.rejection_reason,
            archived_at=_NOW,
        )
    with pytest.raises((AgentDesignArchiveError, ValidationError, ValueError)):
        archive.append_candidate_snapshot(bad, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)) == before


# --- C09 Invariant Qualification Cases (IA-20..IA-24) ---


def test_ia20_completed_negative_evaluation(audit_archive) -> None:
    _ia_passgate(audit_archive)
    neg_eval = {
        "passed": False,
        "execution_ref": "exec-1",
        "evaluation_ref": "eval-1",
        "failure_reasons": ["policy"],
    }
    eval_rec = _ia_append(
        audit_archive,
        sample_record(
            sequence=2,
            status=AgentCandidateStatus.EVALUATED,
            static_gate_result={"passed": True},
            evaluation_result=neg_eval,
        ),
    )
    assert eval_rec.status is AgentCandidateStatus.EVALUATED
    assert eval_rec.evaluation_result is not None
    assert eval_rec.evaluation_result["passed"] is False
    assert list(eval_rec.evaluation_result["failure_reasons"]) == ["policy"]

    ev_model = AgentCandidateEvaluationResult(candidate_id=eval_rec.candidate_id, **neg_eval)
    with pytest.raises(_IA_ERR):
        select_bundle = _ia_bundle(
            eval_rec,
            static_gate_result={"passed": True},
            evaluation_result=ev_model,
            terminal_evidence_outcome=W1TerminalEvidenceOutcome.EVALUATION_COMPLETE,
            selection_outcome=SearchSelectionOutcome.SELECT_CANDIDATE,
            cost_report=sample_w1_cost_report(tokens_consumed=100, root_executions_consumed=2),
        )
        _ia_persist(
            audit_archive,
            select_bundle.model_copy(update={"bundle_digest": compute_evidence_bundle_digest(select_bundle)}),
        )
    assert audit_archive.get_sealed_evidence_bundle("audit-bundle", **_IA_S) is None

    abstain_bundle = _ia_bundle(
        eval_rec,
        static_gate_result={"passed": True},
        evaluation_result=ev_model,
        terminal_evidence_outcome=W1TerminalEvidenceOutcome.EVALUATION_COMPLETE,
        selection_outcome=SearchSelectionOutcome.ABSTAIN,
    )
    persisted = _ia_persist(
        audit_archive,
        abstain_bundle.model_copy(update={"bundle_digest": compute_evidence_bundle_digest(abstain_bundle)}),
    )
    assert persisted.selection_outcome is SearchSelectionOutcome.ABSTAIN
    assert persisted.evaluation_result is not None and persisted.evaluation_result.passed is False

    rejected = _ia_append(
        audit_archive,
        sample_record(
            sequence=3,
            status=AgentCandidateStatus.REJECTED,
            static_gate_result={"passed": True},
            evaluation_result=neg_eval,
            rejection_reason="negative evaluation rejected",
        ),
    )
    assert rejected.status is AgentCandidateStatus.REJECTED
    assert rejected.evaluation_result is not None
    assert rejected.evaluation_result["passed"] is False
    assert list(rejected.evaluation_result["failure_reasons"]) == ["policy"]


@pytest.mark.parametrize(
    "bad",
    [
        {"tokens_consumed": True},
        {"monetary_consumed": False},
        {"root_executions_consumed": "2"},
        {"root_executions_consumed": 2.0},
    ],
    ids=["tokens_bool", "monetary_bool", "roots_str", "roots_float"],
)
def test_ia21_strict_accounting_measurements_rejected(audit_archive, bad: dict[str, object]) -> None:
    r = _ia_eval_record(audit_archive)
    with pytest.raises(_IA_ERR):
        _ia_select(
            audit_archive,
            r,
            sample_w1_cost_report(**({"tokens_consumed": 100, "root_executions_consumed": 2} | bad)),
        )
    assert audit_archive.get_sealed_evidence_bundle("audit-bundle", **_IA_S) is None


def test_ia21_strict_accounting_positive_control(audit_archive) -> None:
    r = _ia_eval_record(audit_archive)
    persisted = _ia_select(
        audit_archive,
        r,
        sample_w1_cost_report(tokens_consumed=100, monetary_consumed=0.0, root_executions_consumed=2),
    )
    assert persisted.selection_outcome is SearchSelectionOutcome.SELECT_CANDIDATE


def test_ia22_pre_execution_rejection_phase_rejects_fabricated_gate(audit_archive) -> None:
    _ia_start(audit_archive)
    with pytest.raises(_IA_ERR):
        _ia_append(
            audit_archive,
            sample_record(
                sequence=1,
                status=AgentCandidateStatus.REJECTED,
                static_gate_result={"passed": False},
                rejection_reason="pre-gate refusal",
            ),
        )
    assert len(audit_archive.get_candidate_history("cand-1", **_IA_S)) == 1


def test_ia22_pre_execution_rejection_positive_control(audit_archive) -> None:
    draft = _ia_start(audit_archive)
    stop = AdasW1LabStopEvidence.model_validate(sample_w1_lab_stop_pre_execution(draft))
    rejected = _ia_append(
        audit_archive,
        sample_record(
            sequence=1,
            status=AgentCandidateStatus.REJECTED,
            static_gate_result=None,
            evaluation_result=None,
            w1_lab_stop_evidence=stop,
            rejection_reason="pre-gate refusal",
        ),
    )
    assert rejected.status is AgentCandidateStatus.REJECTED
    assert rejected.static_gate_result is None
    assert rejected.evaluation_result is None
    assert rejected.w1_lab_stop_evidence is not None
    assert rejected.w1_lab_stop_evidence.execution_ref is None


def test_ia23_missing_typed_stop_rejected(audit_archive) -> None:
    _ia_start(audit_archive)
    with pytest.raises(_IA_ERR):
        _ia_append(
            audit_archive,
            sample_record(
                sequence=1,
                status=AgentCandidateStatus.REJECTED,
                static_gate_result=None,
                w1_lab_stop_evidence=None,
                rejection_reason="pre-gate refusal",
            ),
        )
    assert len(audit_archive.get_candidate_history("cand-1", **_IA_S)) == 1


def test_ia24_failed_gate_with_fabricated_execution_rejected(audit_archive) -> None:
    draft = _ia_start(audit_archive)
    stop_payload = sample_w1_execution_stop_gate(draft)["w1_lab_stop_evidence"]
    with pytest.raises(_IA_ERR):
        _ia_append(
            audit_archive,
            sample_record(
                sequence=1,
                status=AgentCandidateStatus.STATIC_GATE_FAILED,
                static_gate_result={"passed": False},
                w1_lab_stop_evidence=AdasW1LabStopEvidence.model_validate(stop_payload),
                rejection_reason="static rejected",
            ),
        )
    assert len(audit_archive.get_candidate_history("cand-1", **_IA_S)) == 1


def test_ia24_failed_gate_without_execution_positive_control(audit_archive) -> None:
    _ia_start(audit_archive)
    failed = _ia_append(
        audit_archive,
        sample_record(
            sequence=1,
            status=AgentCandidateStatus.STATIC_GATE_FAILED,
            static_gate_result={"passed": False},
            rejection_reason="static rejected",
        ),
    )
    assert failed.status is AgentCandidateStatus.STATIC_GATE_FAILED
    assert failed.w1_lab_stop_evidence is None


# --- C10 Invariant Qualification Cases (IA-25..IA-27) ---


def test_ia25_archival_cannot_invent_evaluation(audit_archive) -> None:
    archive = audit_archive
    _ia_append(archive, sample_record())
    _ia_append(
        archive,
        sample_record(
            sequence=1,
            status=AgentCandidateStatus.STATIC_GATE_FAILED,
            static_gate_result={"passed": False},
            rejection_reason="static failed",
        ),
    )
    good = sample_record(
        sequence=2,
        status=AgentCandidateStatus.ARCHIVED,
        static_gate_result={"passed": False},
        rejection_reason="static failed",
        archived_at=_NOW,
    )
    bad = good.model_copy(
        update={
            "evaluation_result": sample_evaluation_mapping(passed=True),
        }
    )
    with pytest.raises(
        AgentDesignArchiveError,
        match="archival must not fabricate terminal evidence",
    ):
        _ia_append(archive, bad)
    assert len(archive.get_candidate_history("cand-1", **_IA_S)) == 2
    persisted = _ia_append(archive, good)
    assert persisted.status is AgentCandidateStatus.ARCHIVED
    assert persisted.evaluation_result is None


@pytest.mark.parametrize(
    "target",
    [
        AgentCandidateStatus.ARCHIVED,
        AgentCandidateStatus.REJECTED,
    ],
)
def test_ia26_no_new_historical_selection(
    audit_archive,
    target,
) -> None:
    archive = audit_archive
    record = _ia_eval_record(archive)
    _ia_append(
        archive,
        sample_record(
            sequence=3,
            status=target,
            static_gate_result={"passed": True},
            evaluation_result=sample_evaluation_mapping(passed=True),
            rejection_reason=("operator rejected" if target is AgentCandidateStatus.REJECTED else None),
            archived_at=(_NOW if target is AgentCandidateStatus.ARCHIVED else None),
        ),
    )
    with pytest.raises(
        AgentDesignArchiveError,
        match="new selection requires current evaluated snapshot",
    ):
        _ia_select(archive, record, sample_w1_cost_report())
    assert archive.get_sealed_evidence_bundle("audit-bundle", **_IA_S) is None


@pytest.mark.parametrize(
    "target",
    [
        AgentCandidateStatus.ARCHIVED,
        AgentCandidateStatus.REJECTED,
    ],
)
def test_ia26_existing_selection_remains_idempotent(
    audit_archive,
    target,
) -> None:
    archive = audit_archive
    record = _ia_eval_record(archive)
    original = _ia_select(archive, record, sample_w1_cost_report())
    _ia_append(
        archive,
        sample_record(
            sequence=3,
            status=target,
            static_gate_result={"passed": True},
            evaluation_result=sample_evaluation_mapping(passed=True),
            rejection_reason=("operator rejected" if target is AgentCandidateStatus.REJECTED else None),
            archived_at=(_NOW if target is AgentCandidateStatus.ARCHIVED else None),
        ),
    )
    duplicate = _ia_select(archive, record, sample_w1_cost_report())
    assert duplicate.bundle_digest == original.bundle_digest
    assert duplicate.model_dump(mode="json") == original.model_dump(mode="json")


def test_ia27_foreign_bundle_collision(
    audit_archive,
    monkeypatch,
) -> None:
    import json

    archive = audit_archive

    def scoped_record(cid: str, tenant: str, seq: int, status: AgentCandidateStatus) -> AgentCandidateRecord:
        kwargs: dict[str, object] = (
            {}
            if seq == 0
            else {
                "static_gate_result": {"passed": False},
                "rejection_reason": "static failed",
            }
        )
        record = sample_record(
            candidate_id=cid,
            sequence=seq,
            status=status,
            **kwargs,
        )
        return record.model_copy(
            update={
                "tenant_id": tenant,
                "envelope": record.envelope.model_copy(update={"tenant_id": tenant}),
            }
        )

    def build(cid: str, tenant: str) -> tuple[AgentCandidateEvidenceBundle, dict[str, str]]:
        scope = {
            "tenant_id": tenant,
            "workspace_id": "ws-a",
        }
        archive.append_candidate_snapshot(
            scoped_record(cid, tenant, 0, AgentCandidateStatus.DRAFT),
            **scope,
        )
        record = archive.append_candidate_snapshot(
            scoped_record(cid, tenant, 1, AgentCandidateStatus.STATIC_GATE_FAILED),
            **scope,
        )
        bundle = _ia_bundle(
            record,
            static_gate_result={"passed": False},
            terminal_evidence_outcome=W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
            selection_outcome=SearchSelectionOutcome.ABSTAIN,
        )
        return bundle, scope

    foreign, foreign_scope = build("foreign-cand", "tenant-b")
    archive.append_sealed_evidence_bundle(foreign, **foreign_scope)
    local, local_scope = build("local-cand", "tenant-a")
    original = AgentCandidateEvidenceBundle.model_validate_json
    seen: list[str] = []

    def guard(data: str, *args, **kwargs):  # noqa: ANN002, ANN003
        if json.loads(data)["tenant_id"] == "tenant-b":
            seen.append("foreign")
            raise AssertionError("foreign bundle deserialization")
        return original(data, *args, **kwargs)

    monkeypatch.setattr(
        AgentCandidateEvidenceBundle,
        "model_validate_json",
        guard,
    )
    with pytest.raises(
        AgentDesignArchiveScopeError,
        match="bundle identity scope mismatch",
    ):
        archive.append_sealed_evidence_bundle(local, **local_scope)
    assert seen == []
    assert archive.get_sealed_evidence_bundle("audit-bundle", **local_scope) is None


_W1_LIFECYCLE_STATES = frozenset(
    {
        AgentCandidateStatus.DRAFT,
        AgentCandidateStatus.STATIC_GATE_PASSED,
        AgentCandidateStatus.STATIC_GATE_FAILED,
        AgentCandidateStatus.EVALUATED,
        AgentCandidateStatus.REJECTED,
        AgentCandidateStatus.ARCHIVED,
    }
)

_W1_PERMITTED_EDGES = frozenset(
    {
        (AgentCandidateStatus.DRAFT, AgentCandidateStatus.STATIC_GATE_PASSED),
        (AgentCandidateStatus.DRAFT, AgentCandidateStatus.STATIC_GATE_FAILED),
        (AgentCandidateStatus.DRAFT, AgentCandidateStatus.REJECTED),
        (AgentCandidateStatus.STATIC_GATE_PASSED, AgentCandidateStatus.EVALUATED),
        (AgentCandidateStatus.STATIC_GATE_PASSED, AgentCandidateStatus.REJECTED),
        (AgentCandidateStatus.EVALUATED, AgentCandidateStatus.REJECTED),
        (AgentCandidateStatus.EVALUATED, AgentCandidateStatus.ARCHIVED),
        (AgentCandidateStatus.STATIC_GATE_FAILED, AgentCandidateStatus.ARCHIVED),
        (AgentCandidateStatus.REJECTED, AgentCandidateStatus.ARCHIVED),
    }
)


def _seed_archive_w1_status(
    archive: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
    from_status: AgentCandidateStatus,
) -> None:
    draft = sample_record(sequence=0, status=AgentCandidateStatus.DRAFT)
    archive.append_candidate_snapshot(draft, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    if from_status is AgentCandidateStatus.DRAFT:
        return
    if from_status is AgentCandidateStatus.STATIC_GATE_PASSED:
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1, status=AgentCandidateStatus.STATIC_GATE_PASSED, static_gate_result={"passed": True}
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        return
    if from_status is AgentCandidateStatus.STATIC_GATE_FAILED:
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1,
                status=AgentCandidateStatus.STATIC_GATE_FAILED,
                static_gate_result={"passed": False},
                rejection_reason="static failed",
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        return
    if from_status is AgentCandidateStatus.EVALUATED:
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1, status=AgentCandidateStatus.STATIC_GATE_PASSED, static_gate_result={"passed": True}
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        archive.append_candidate_snapshot(
            sample_record(
                sequence=2,
                status=AgentCandidateStatus.EVALUATED,
                static_gate_result={"passed": True},
                evaluation_result=sample_evaluation_mapping(passed=True),
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        return
    if from_status is AgentCandidateStatus.REJECTED:
        stop = AdasW1LabStopEvidence.model_validate(sample_w1_lab_stop_pre_execution(draft))
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1,
                status=AgentCandidateStatus.REJECTED,
                w1_lab_stop_evidence=stop,
                rejection_reason="pre-gate refusal",
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        return
    if from_status is AgentCandidateStatus.ARCHIVED:
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1,
                status=AgentCandidateStatus.STATIC_GATE_FAILED,
                static_gate_result={"passed": False},
                rejection_reason="static failed",
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
        archive.append_candidate_snapshot(
            sample_record(
                sequence=2,
                status=AgentCandidateStatus.ARCHIVED,
                static_gate_result={"passed": False},
                rejection_reason="static failed",
                archived_at=_NOW,
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )


_Q07_02_FORBIDDEN_EDGES = tuple(
    sorted((a, b) for a in _W1_LIFECYCLE_STATES for b in _W1_LIFECYCLE_STATES if (a, b) not in _W1_PERMITTED_EDGES)
)


def _q07_02_build_forbidden_successor(
    *,
    sequence: int,
    to_status: AgentCandidateStatus,
    latest: AgentCandidateRecord,
) -> AgentCandidateRecord:
    latest_gate = adas_json_to_plain(latest.static_gate_result) if latest.static_gate_result is not None else None
    latest_eval = adas_json_to_plain(latest.evaluation_result) if latest.evaluation_result is not None else None
    if to_status is AgentCandidateStatus.EVALUATED:
        return sample_record(
            sequence=sequence,
            status=to_status,
            static_gate_result=latest_gate or {"passed": True},
            evaluation_result=sample_evaluation_mapping(passed=True),
            rejection_reason=latest.rejection_reason,
            w1_lab_stop_evidence=latest.w1_lab_stop_evidence,
            archived_at=latest.archived_at,
        )
    if to_status is AgentCandidateStatus.STATIC_GATE_FAILED:
        return sample_record(
            sequence=sequence,
            status=to_status,
            static_gate_result={"passed": False},
            rejection_reason="static failed",
        )
    if to_status is AgentCandidateStatus.STATIC_GATE_PASSED:
        return sample_record(
            sequence=sequence,
            status=to_status,
            static_gate_result={"passed": True},
            evaluation_result=latest_eval,
            rejection_reason=latest.rejection_reason,
        )
    if to_status is AgentCandidateStatus.REJECTED:
        return sample_record(
            sequence=sequence,
            status=to_status,
            static_gate_result=latest_gate,
            rejection_reason=latest.rejection_reason or "rejected",
            archived_at=latest.archived_at,
        )
    return sample_record(
        sequence=sequence,
        status=to_status,
        static_gate_result=latest_gate,
        rejection_reason=latest.rejection_reason,
        archived_at=_NOW,
    )


def test_q07_02_w1_lifecycle_transition_table_matches_cartesian_matrix(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
) -> None:
    from intergrax.runtime.adaptive.agent_design_search.archive import _W1_STATUS_TRANSITIONS

    derived: set[tuple[AgentCandidateStatus, AgentCandidateStatus]] = set()
    for prior_status, targets in _W1_STATUS_TRANSITIONS.items():
        for target in targets:
            derived.add((prior_status, target))
    assert derived == _W1_PERMITTED_EDGES


@pytest.mark.parametrize(
    "from_status,to_status",
    _Q07_02_FORBIDDEN_EDGES,
    ids=[f"{left.value}-to-{right.value}" for left, right in _Q07_02_FORBIDDEN_EDGES],
)
def test_q07_02_forbidden_successor_append_rejected(
    archive_backend: InMemoryAgentDesignArchive | SQLiteAgentDesignArchive,
    from_status: AgentCandidateStatus,
    to_status: AgentCandidateStatus,
) -> None:
    archive = archive_backend
    _seed_archive_w1_status(archive, from_status)
    before = len(archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE))
    latest = archive.get_latest_candidate_snapshot("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert latest is not None
    bad = _q07_02_build_forbidden_successor(sequence=before, to_status=to_status, latest=latest)
    with pytest.raises(AgentDesignArchiveError):
        archive.append_candidate_snapshot(bad, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)) == before


_Q13_01_Q05_Q11_ARCHIVE_WITNESSES: dict[str, tuple[str, ...]] = {
    "Q05": ("test_q05_unknown_parent_same_as_foreign_parent",),
    "Q06": (
        "test_q06_immutable_successor_positive_control",
        "test_q06_immutable_identity_fields_rejected_on_successor",
        "test_q06_candidate_identity_rebinding_rejected",
    ),
    "Q07": (
        "test_q07_rejects_successor_omitting_prior_evaluation",
        "test_q07_02_w1_lifecycle_transition_table_matches_cartesian_matrix",
        "test_q07_02_forbidden_successor_append_rejected",
    ),
    "Q08": ("test_q08_compatible_additive_nested_evidence",),
    "Q09": ("test_q09_actual_parent_child_lineage_queries",),
    "Q11": ("test_q11_evidence_binds_exact_historical_snapshot",),
}


def test_q13_01_backend_parity_witness_collects_memory_and_sqlite(tmp_path: Path) -> None:
    import subprocess
    import sys
    from pathlib import Path as PathLib

    witness_tests = sorted({name for names in _Q13_01_Q05_Q11_ARCHIVE_WITNESSES.values() for name in names})
    assert witness_tests
    repo_root = PathLib(__file__).resolve().parents[5]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/unit/runtime/adaptive/agent_design_search/test_archive.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    collected = proc.stdout.splitlines()
    for test_name in witness_tests:
        matching = [line for line in collected if f"::{test_name}" in line]
        assert matching, test_name
        assert any("[memory" in line for line in matching), test_name
        assert any("[sqlite" in line for line in matching), test_name

    memory = InMemoryAgentDesignArchive()
    sqlite = SQLiteAgentDesignArchive(tmp_path / "parity.sqlite3")
    for archive in (memory, sqlite):
        draft = sample_record(sequence=0, status=AgentCandidateStatus.DRAFT)
        archive.append_candidate_snapshot(draft, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        latest = archive.get_latest_candidate_snapshot("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert latest is not None and latest.status is AgentCandidateStatus.DRAFT
    sqlite.close()
