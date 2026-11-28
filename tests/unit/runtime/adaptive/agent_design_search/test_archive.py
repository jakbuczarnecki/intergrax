# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from intergrax.runtime.adaptive.agent_design_search.archive import (
    AgentDesignArchiveError,
    AgentDesignArchiveScopeError,
    InMemoryAgentDesignArchive,
)
from intergrax.runtime.adaptive.agent_design_search.contracts import (
    AgentCandidateEvaluationResult,
    AgentCandidateEvidenceBundle,
    AgentCandidateStatus,
    SearchSelectionOutcome,
    W1TerminalEvidenceOutcome,
    compute_evidence_bundle_digest,
)
from intergrax.runtime.adaptive.agent_design_search.sqlite_archive import SQLiteAgentDesignArchive
from pydantic import ValidationError
from tests.unit.runtime.adaptive.agent_design_search.test_contracts import (
    _DIGEST,
    sample_evaluation_mapping,
    sample_record,
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
