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
    AgentCandidateEvidenceBundle,
    AgentCandidateStatus,
    compute_evidence_bundle_digest,
)
from intergrax.runtime.adaptive.agent_design_search.sqlite_archive import SQLiteAgentDesignArchive
from pydantic import ValidationError
from tests.unit.runtime.adaptive.agent_design_search.test_contracts import _DIGEST, sample_record

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


def test_snapshot_append_history_and_latest() -> None:
    archive = InMemoryAgentDesignArchive()
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


def test_idempotent_and_conflicting_duplicates() -> None:
    archive = InMemoryAgentDesignArchive()
    record = sample_record()
    archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    conflict = record.model_copy(update={"lineage_id": "line-2"})
    with pytest.raises(AgentDesignArchiveError):
        archive.append_candidate_snapshot(conflict, tenant_id=_TENANT, workspace_id=_WORKSPACE)


def test_invalid_transition_and_lineage_checks() -> None:
    archive = InMemoryAgentDesignArchive()
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


def test_cross_scope_rejection() -> None:
    archive = InMemoryAgentDesignArchive()
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


def test_scoped_listing_parent_child_and_sealed_bundle() -> None:
    archive = InMemoryAgentDesignArchive()
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


def test_mutation_escape_blocked() -> None:
    archive = InMemoryAgentDesignArchive()
    record = sample_record().model_copy(update={"agent_contract_snapshot": {"layer": {"k": "v"}}})
    stored = archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    with pytest.raises(TypeError):
        stored.agent_contract_snapshot["layer"]["k"] = "mutated"  # type: ignore[index]


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
    evaluated = sample_record(
        sequence=2,
        status=AgentCandidateStatus.EVALUATED,
        snapshot_id="snap-eval-2",
        static_gate_result={"passed": True},
        evaluation_result={
            "passed": True,
            "execution_ref": "exec-1",
            "evaluation_ref": "eval-1",
        },
    )
    archive.append_candidate_snapshot(draft, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(gate_passed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(evaluated, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    rejected = sample_record(
        sequence=3,
        status=AgentCandidateStatus.REJECTED,
        snapshot_id="snap-eval-3",
        static_gate_result={"passed": True},
        evaluation_result={
            "passed": True,
            "execution_ref": "exec-1",
            "evaluation_ref": "eval-1",
        },
        rejection_reason="policy",
    )
    archive.append_candidate_snapshot(rejected, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    bundle = AgentCandidateEvidenceBundle(
        bundle_id="bundle-historical",
        candidate_id=draft.candidate_id,
        objective_id=draft.objective_id,
        tenant_id=draft.tenant_id,
        workspace_id=draft.workspace_id,
        created_at=datetime.now(UTC),
        created_by="operator",
        baseline_configuration_digest=_DIGEST,
        candidate_configuration_digest=evaluated.candidate_configuration_digest,
        candidate_record=evaluated,
        static_gate_result={"passed": True},
        assembled_at=datetime.now(UTC),
        sealed=True,
    )
    digest = compute_evidence_bundle_digest(bundle)
    sealed = bundle.model_copy(update={"bundle_digest": digest})
    archive.append_sealed_evidence_bundle(sealed, tenant_id=_TENANT, workspace_id=_WORKSPACE)


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
        evaluation_result={"passed": True, "execution_ref": "exec-1", "evaluation_ref": "eval-1"},
    )
    archive.append_candidate_snapshot(gate_passed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.append_candidate_snapshot(evaluated, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    omitted = sample_record(
        sequence=3,
        status=AgentCandidateStatus.REJECTED,
        static_gate_result={"passed": False, "reason": "later"},
        rejection_reason="later",
        evaluation_result=None,
    )
    with pytest.raises(AgentDesignArchiveError):
        archive.append_candidate_snapshot(omitted, tenant_id=_TENANT, workspace_id=_WORKSPACE)


def test_q05_unknown_parent_same_as_foreign_parent() -> None:
    archive = InMemoryAgentDesignArchive()
    child = sample_record(candidate_id="child-2", sequence=0).model_copy(
        update={"parent_candidate_id": "missing-parent"},
    )
    with pytest.raises(AgentDesignArchiveError) as exc:
        archive.append_candidate_snapshot(child, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert "foreign" not in str(exc.value).lower()
