# © Artur Czarnecki. All rights reserved.

"""Append-only in-memory ADAS candidate archive (AHI-ADAS-10)."""

from __future__ import annotations

import copy
from typing import Protocol

from intergrax.runtime.adaptive.agent_design_search.contracts import (
    AgentCandidateEvidenceBundle,
    AgentCandidateRecord,
    AgentCandidateStatus,
    compute_evidence_bundle_digest,
    verify_evidence_bundle_digest,
)

_W1_STATUS_TRANSITIONS: dict[AgentCandidateStatus, frozenset[AgentCandidateStatus]] = {
    AgentCandidateStatus.DRAFT: frozenset(
        {
            AgentCandidateStatus.STATIC_GATE_PASSED,
            AgentCandidateStatus.STATIC_GATE_FAILED,
            AgentCandidateStatus.REJECTED,
        }
    ),
    AgentCandidateStatus.STATIC_GATE_PASSED: frozenset({AgentCandidateStatus.EVALUATED, AgentCandidateStatus.REJECTED}),
    AgentCandidateStatus.EVALUATED: frozenset({AgentCandidateStatus.REJECTED, AgentCandidateStatus.ARCHIVED}),
    AgentCandidateStatus.STATIC_GATE_FAILED: frozenset({AgentCandidateStatus.ARCHIVED}),
    AgentCandidateStatus.REJECTED: frozenset({AgentCandidateStatus.ARCHIVED}),
}

_FORBIDDEN_W1_TARGETS = frozenset(
    {
        AgentCandidateStatus.SCAFFOLDED,
        AgentCandidateStatus.SHADOW,
        AgentCandidateStatus.CANARY,
        AgentCandidateStatus.PROMOTED,
        AgentCandidateStatus.RETIRED,
    }
)


class AgentDesignArchiveError(Exception):
    """Archive operation rejected."""


class AgentDesignArchiveScopeError(AgentDesignArchiveError):
    """Scope or access violation without foreign detail."""


class AgentDesignArchive(Protocol):
    def append_candidate_snapshot(
        self,
        record: AgentCandidateRecord,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord: ...

    def get_latest_candidate_snapshot(
        self,
        candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord | None: ...

    def get_candidate_history(
        self,
        candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[AgentCandidateRecord]: ...

    def list_candidates_for_objective(
        self,
        objective_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
        status: AgentCandidateStatus | None = None,
    ) -> list[AgentCandidateRecord]: ...

    def get_child_candidates(
        self,
        parent_candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[AgentCandidateRecord]: ...

    def get_parent_candidate(
        self,
        candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord | None: ...

    def append_sealed_evidence_bundle(
        self,
        bundle: AgentCandidateEvidenceBundle,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateEvidenceBundle: ...

    def get_sealed_evidence_bundle(
        self,
        bundle_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateEvidenceBundle | None: ...


def _freeze_record(record: AgentCandidateRecord) -> AgentCandidateRecord:
    payload = record.model_dump(mode="python")
    payload["agent_contract_snapshot"] = copy.deepcopy(payload.get("agent_contract_snapshot", {}))
    if payload.get("static_gate_result") is not None:
        payload["static_gate_result"] = copy.deepcopy(payload["static_gate_result"])
    if payload.get("evaluation_result") is not None:
        payload["evaluation_result"] = copy.deepcopy(payload["evaluation_result"])
    return AgentCandidateRecord.model_validate(payload)


def _assert_scope(record: AgentCandidateRecord, *, tenant_id: str, workspace_id: str) -> None:
    if record.tenant_id != tenant_id or record.workspace_id != workspace_id:
        raise AgentDesignArchiveScopeError("record scope mismatch")


def _identity_tuple(record: AgentCandidateRecord) -> tuple[str, str, str, str, str]:
    return (
        record.candidate_id,
        record.objective_id,
        record.tenant_id,
        record.workspace_id,
        record.lineage_id,
    )


def _validate_parent_link(
    store: dict[str, list[AgentCandidateRecord]],
    record: AgentCandidateRecord,
) -> None:
    parent_id = record.parent_candidate_id
    if parent_id is None:
        return
    if parent_id == record.candidate_id:
        raise AgentDesignArchiveError("candidate cannot be its own parent")
    history = store.get(parent_id)
    if not history:
        raise AgentDesignArchiveError("unknown parent candidate")
    parent = history[-1]
    if parent.tenant_id != record.tenant_id or parent.workspace_id != record.workspace_id:
        raise AgentDesignArchiveScopeError("cross-scope lineage rejected")
    if parent.objective_id != record.objective_id:
        raise AgentDesignArchiveError("cross-objective lineage rejected")
    if _would_cycle(store, record.candidate_id, parent_id):
        raise AgentDesignArchiveError("cyclic lineage rejected")


def _would_cycle(
    store: dict[str, list[AgentCandidateRecord]],
    candidate_id: str,
    parent_id: str,
) -> bool:
    seen: set[str] = {candidate_id}
    current = parent_id
    while current is not None:
        if current in seen:
            return True
        seen.add(current)
        history = store.get(current)
        if not history:
            break
        current = history[-1].parent_candidate_id
    return False


def _validate_transition(
    prior: AgentCandidateRecord | None,
    incoming: AgentCandidateRecord,
) -> None:
    if incoming.status in _FORBIDDEN_W1_TARGETS:
        raise AgentDesignArchiveError("W1 archive profile forbids this status transition target")
    if prior is None:
        if incoming.status is not AgentCandidateStatus.DRAFT or incoming.sequence != 0:
            raise AgentDesignArchiveError("first snapshot must be DRAFT at sequence 0")
        return
    if _identity_tuple(prior) != _identity_tuple(incoming):
        raise AgentDesignArchiveError("immutable candidate identity violation")
    baseline_changed = prior.baseline != incoming.baseline
    digest_changed = prior.candidate_configuration_digest != incoming.candidate_configuration_digest
    if baseline_changed or digest_changed:
        raise AgentDesignArchiveError("immutable configuration lineage violation")
    if incoming.sequence != prior.sequence + 1:
        raise AgentDesignArchiveError("snapshot sequence must increase by one")
    allowed = _W1_STATUS_TRANSITIONS.get(prior.status, frozenset())
    if incoming.status not in allowed:
        raise AgentDesignArchiveError("invalid lifecycle transition")
    if incoming.status is AgentCandidateStatus.ARCHIVED and incoming.archived_at is None:
        raise AgentDesignArchiveError("archived snapshots require archived_at")


class InMemoryAgentDesignArchive:
    """Process-local append-only archive."""

    def __init__(self) -> None:
        self._history: dict[str, list[AgentCandidateRecord]] = {}
        self._snapshots: dict[str, AgentCandidateRecord] = {}
        self._bundles: dict[str, AgentCandidateEvidenceBundle] = {}

    def append_candidate_snapshot(
        self,
        record: AgentCandidateRecord,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord:
        _assert_scope(record, tenant_id=tenant_id, workspace_id=workspace_id)
        frozen = _freeze_record(record)
        prior = self._history.get(frozen.candidate_id, [None])[-1]
        if frozen.snapshot_id in self._snapshots:
            existing = self._snapshots[frozen.snapshot_id]
            if existing.model_dump(mode="json") == frozen.model_dump(mode="json"):
                return existing
            raise AgentDesignArchiveError("conflicting snapshot identity")
        _validate_parent_link(self._history, frozen)
        _validate_transition(prior, frozen)
        self._snapshots[frozen.snapshot_id] = frozen
        self._history.setdefault(frozen.candidate_id, []).append(frozen)
        return frozen

    def get_latest_candidate_snapshot(
        self,
        candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord | None:
        history = self._history.get(candidate_id)
        if not history:
            return None
        latest = history[-1]
        if latest.tenant_id != tenant_id or latest.workspace_id != workspace_id:
            raise AgentDesignArchiveScopeError("record not found")
        return latest

    def get_candidate_history(
        self,
        candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[AgentCandidateRecord]:
        history = self._history.get(candidate_id, [])
        if not history:
            return []
        if history[0].tenant_id != tenant_id or history[0].workspace_id != workspace_id:
            raise AgentDesignArchiveScopeError("record not found")
        return list(history)

    def list_candidates_for_objective(
        self,
        objective_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
        status: AgentCandidateStatus | None = None,
    ) -> list[AgentCandidateRecord]:
        latest: dict[str, AgentCandidateRecord] = {}
        for records in self._history.values():
            for record in records:
                if record.tenant_id != tenant_id or record.workspace_id != workspace_id:
                    continue
                if record.objective_id != objective_id:
                    continue
                prior = latest.get(record.candidate_id)
                if prior is None or record.sequence > prior.sequence:
                    latest[record.candidate_id] = record
        values = sorted(latest.values(), key=lambda item: (item.created_at, item.candidate_id, item.sequence))
        if status is None:
            return values
        return [item for item in values if item.status is status]

    def get_child_candidates(
        self,
        parent_candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[AgentCandidateRecord]:
        parent = self.get_latest_candidate_snapshot(parent_candidate_id, tenant_id=tenant_id, workspace_id=workspace_id)
        if parent is None:
            raise AgentDesignArchiveScopeError("record not found")
        children: list[AgentCandidateRecord] = []
        for records in self._history.values():
            latest = records[-1]
            if latest.parent_candidate_id != parent_candidate_id:
                continue
            if latest.tenant_id != tenant_id or latest.workspace_id != workspace_id:
                continue
            children.append(latest)
        return sorted(children, key=lambda item: item.candidate_id)

    def get_parent_candidate(
        self,
        candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord | None:
        latest = self.get_latest_candidate_snapshot(candidate_id, tenant_id=tenant_id, workspace_id=workspace_id)
        if latest is None or latest.parent_candidate_id is None:
            return None
        return self.get_latest_candidate_snapshot(
            latest.parent_candidate_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )

    def append_sealed_evidence_bundle(
        self,
        bundle: AgentCandidateEvidenceBundle,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateEvidenceBundle:
        if bundle.tenant_id != tenant_id or bundle.workspace_id != workspace_id:
            raise AgentDesignArchiveScopeError("bundle scope mismatch")
        if not bundle.sealed:
            raise AgentDesignArchiveError("bundle must be sealed before append")
        digest = bundle.bundle_digest or compute_evidence_bundle_digest(bundle)
        payload = bundle.model_dump(mode="python")
        payload["bundle_digest"] = digest
        frozen = AgentCandidateEvidenceBundle.model_validate(payload)
        verify_evidence_bundle_digest(frozen)
        if bundle.bundle_id in self._bundles:
            existing = self._bundles[bundle.bundle_id]
            if existing.model_dump(mode="json") == frozen.model_dump(mode="json"):
                return existing
            raise AgentDesignArchiveError("conflicting evidence bundle identity")
        self._bundles[bundle.bundle_id] = frozen
        return frozen

    def get_sealed_evidence_bundle(
        self,
        bundle_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateEvidenceBundle | None:
        bundle = self._bundles.get(bundle_id)
        if bundle is None:
            return None
        if bundle.tenant_id != tenant_id or bundle.workspace_id != workspace_id:
            raise AgentDesignArchiveScopeError("bundle not found")
        return bundle
