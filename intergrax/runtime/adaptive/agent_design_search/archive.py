# © Artur Czarnecki. All rights reserved.

"""Append-only in-memory ADAS candidate archive (AHI-ADAS-10)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol

from pydantic import ValidationError

from intergrax.runtime.adaptive.agent_design_search.contracts import (
    AdasJsonValue,
    AgentCandidateEvidenceBundle,
    AgentCandidateRecord,
    AgentCandidateStatus,
    SearchSelectionOutcome,
    adas_json_to_plain,
    adas_static_gate_verdict,
    compute_evidence_bundle_digest,
    validate_evaluation_mapping_for_record,
    validate_sealed_terminal_evidence_bundle,
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


def require_archive_read_scope(*, tenant_id: str, workspace_id: str) -> None:
    if not tenant_id.strip() or not workspace_id.strip():
        raise AgentDesignArchiveScopeError("tenant and workspace scope required")


def freeze_candidate_record(record: AgentCandidateRecord) -> AgentCandidateRecord:
    payload = json.loads(record.model_dump_json())
    detached = AgentCandidateRecord.model_validate(payload)
    return detached


def _assert_record_write_scope(record: AgentCandidateRecord, *, tenant_id: str, workspace_id: str) -> None:
    if record.tenant_id != tenant_id or record.workspace_id != workspace_id:
        raise AgentDesignArchiveScopeError("record scope mismatch")
    if record.envelope.tenant_id != tenant_id or record.envelope.workspace_id != workspace_id:
        raise AgentDesignArchiveScopeError("record scope mismatch")


def _identity_tuple(record: AgentCandidateRecord) -> tuple[str, str, str, str, str, str, str]:
    return (
        record.candidate_id,
        record.objective_id,
        record.search_run_id,
        record.tenant_id,
        record.workspace_id,
        record.lineage_id,
        record.parent_candidate_id or "",
    )


def _immutable_provenance_tuple(record: AgentCandidateRecord) -> tuple[object, ...]:
    return (
        record.baseline.model_dump(mode="json"),
        record.candidate_configuration_digest,
        record.configuration_delta.model_dump(mode="json"),
        record.created_by,
        record.created_at.isoformat(),
        adas_json_to_plain(record.agent_contract_snapshot),
    )


def validate_parent_link(
    records_by_candidate: dict[str, list[AgentCandidateRecord]],
    record: AgentCandidateRecord,
) -> None:
    parent_id = record.parent_candidate_id
    if parent_id is None:
        return
    if parent_id == record.candidate_id:
        raise AgentDesignArchiveError("candidate cannot be its own parent")
    history = records_by_candidate.get(parent_id)
    if not history:
        raise AgentDesignArchiveError("parent candidate unavailable")
    parent = history[-1]
    if parent.tenant_id != record.tenant_id or parent.workspace_id != record.workspace_id:
        raise AgentDesignArchiveError("parent candidate unavailable")
    if parent.objective_id != record.objective_id:
        raise AgentDesignArchiveError("cross-objective lineage rejected")
    if would_create_lineage_cycle(records_by_candidate, record.candidate_id, parent_id):
        raise AgentDesignArchiveError("cyclic lineage rejected")


def would_create_lineage_cycle(
    records_by_candidate: dict[str, list[AgentCandidateRecord]],
    candidate_id: str,
    parent_id: str,
) -> bool:
    seen: set[str] = {candidate_id}
    current: str | None = parent_id
    while current is not None:
        if current in seen:
            return True
        seen.add(current)
        history = records_by_candidate.get(current)
        if not history:
            break
        current = history[-1].parent_candidate_id
    return False


def _gate_passed_payload(record: AgentCandidateRecord) -> bool:
    payload = record.static_gate_result
    if payload is None:
        return False
    plain = adas_json_to_plain(payload)
    try:
        return adas_static_gate_verdict(plain) is True
    except ValueError:
        return False


def _gate_failed_payload(record: AgentCandidateRecord) -> bool:
    payload = record.static_gate_result
    if payload is None:
        return False
    plain = adas_json_to_plain(payload)
    try:
        return adas_static_gate_verdict(plain) is False
    except ValueError:
        return False


def _has_rejection_evidence(record: AgentCandidateRecord) -> bool:
    if record.rejection_reason and record.rejection_reason.strip():
        return True
    plain = adas_json_to_plain(record.static_gate_result) if record.static_gate_result is not None else {}
    reason = plain.get("reason") or plain.get("rejection_reason")
    return isinstance(reason, str) and bool(reason.strip())


def _has_evaluation_evidence(record: AgentCandidateRecord) -> bool:
    if record.evaluation_result is None:
        return False
    try:
        result = validate_evaluation_mapping_for_record(
            record.evaluation_result,
            candidate_id=record.candidate_id,
        )
    except (ValueError, ValidationError):
        return False
    return bool(result.execution_ref or result.evaluation_ref or result.failure_reasons)


def _validate_accumulated_evidence(
    *,
    prior: dict[str, AdasJsonValue] | None,
    incoming: dict[str, AdasJsonValue] | None,
    label: str,
) -> None:
    if prior is None:
        return
    if incoming is None:
        raise AgentDesignArchiveError(f"successor must preserve {label}")
    for key, value in prior.items():
        if key not in incoming:
            raise AgentDesignArchiveError(f"successor removed established {label} field")
        if incoming[key] != value:
            raise AgentDesignArchiveError(f"contradictory {label} evidence")


def validate_status_evidence(record: AgentCandidateRecord) -> None:
    if record.status is AgentCandidateStatus.STATIC_GATE_PASSED and not _gate_passed_payload(record):
        raise AgentDesignArchiveError("STATIC_GATE_PASSED requires successful static gate evidence")
    if record.status is AgentCandidateStatus.STATIC_GATE_FAILED and (
        not _gate_failed_payload(record) or not _has_rejection_evidence(record)
    ):
        raise AgentDesignArchiveError("STATIC_GATE_FAILED requires failed static gate evidence and rejection reason")
    if record.status is AgentCandidateStatus.EVALUATED:
        if record.evaluation_result is None:
            raise AgentDesignArchiveError("EVALUATED requires evaluation evidence references")
        try:
            validate_evaluation_mapping_for_record(
                record.evaluation_result,
                candidate_id=record.candidate_id,
            )
        except (ValueError, ValidationError) as exc:
            raise AgentDesignArchiveError("EVALUATED requires valid typed evaluation evidence") from exc
    if record.static_gate_result is not None:
        try:
            adas_static_gate_verdict(adas_json_to_plain(record.static_gate_result))
        except ValueError as exc:
            raise AgentDesignArchiveError("static gate evidence must declare boolean verdict") from exc
    if record.status is AgentCandidateStatus.REJECTED and not _has_rejection_evidence(record):
        raise AgentDesignArchiveError("REJECTED requires explicit rejection evidence")
    if record.status is AgentCandidateStatus.ARCHIVED and record.archived_at is None:
        raise AgentDesignArchiveError("ARCHIVED requires archival timestamp")


def _frozen_evidence_plain(value: Mapping[str, AdasJsonValue] | None) -> dict[str, AdasJsonValue] | None:
    if value is None:
        return None
    return adas_json_to_plain(value)


def validate_transition(
    prior: AgentCandidateRecord | None,
    incoming: AgentCandidateRecord,
) -> None:
    if incoming.status in _FORBIDDEN_W1_TARGETS:
        raise AgentDesignArchiveError("W1 archive profile forbids this status transition target")
    if prior is not None and prior.static_gate_result is not None:
        if incoming.static_gate_result is None:
            raise AgentDesignArchiveError("successor must preserve static gate evidence")
        _validate_accumulated_evidence(
            prior=_frozen_evidence_plain(prior.static_gate_result),
            incoming=_frozen_evidence_plain(incoming.static_gate_result),
            label="static gate",
        )
    if prior is not None and prior.evaluation_result is not None:
        if incoming.evaluation_result is None:
            if prior.status is AgentCandidateStatus.EVALUATED and incoming.status is AgentCandidateStatus.REJECTED:
                raise AgentDesignArchiveError("rejected successor must preserve evaluation evidence")
            raise AgentDesignArchiveError("successor must preserve evaluation evidence")
        _validate_accumulated_evidence(
            prior=_frozen_evidence_plain(prior.evaluation_result),
            incoming=_frozen_evidence_plain(incoming.evaluation_result),
            label="evaluation",
        )
    if prior is None:
        if incoming.status is not AgentCandidateStatus.DRAFT or incoming.sequence != 0:
            raise AgentDesignArchiveError("first snapshot must be DRAFT at sequence 0")
        validate_status_evidence(incoming)
        return
    if _identity_tuple(prior) != _identity_tuple(incoming):
        raise AgentDesignArchiveError("immutable candidate identity violation")
    if _immutable_provenance_tuple(prior) != _immutable_provenance_tuple(incoming):
        raise AgentDesignArchiveError("immutable configuration lineage violation")
    if incoming.sequence != prior.sequence + 1:
        raise AgentDesignArchiveError("snapshot sequence must increase by one")
    allowed = _W1_STATUS_TRANSITIONS.get(prior.status, frozenset())
    if incoming.status not in allowed:
        raise AgentDesignArchiveError("invalid lifecycle transition")
    validate_status_evidence(incoming)


def validate_evidence_bundle_against_archive(
    bundle: AgentCandidateEvidenceBundle,
    record: AgentCandidateRecord | None,
) -> None:
    if record is None:
        raise AgentDesignArchiveError("orphan evidence bundle rejected")
    if (
        record.candidate_id != bundle.candidate_id
        or record.objective_id != bundle.objective_id
        or record.tenant_id != bundle.tenant_id
        or record.workspace_id != bundle.workspace_id
    ):
        raise AgentDesignArchiveError("evidence bundle candidate mismatch")
    if record.model_dump(mode="json") != bundle.candidate_record.model_dump(mode="json"):
        raise AgentDesignArchiveError("evidence bundle must bind to archived candidate snapshot")
    if bundle.search_run_id is not None and bundle.search_run_id != record.search_run_id:
        raise AgentDesignArchiveError("evidence bundle search run mismatch")
    terminal = record.status in {
        AgentCandidateStatus.STATIC_GATE_FAILED,
        AgentCandidateStatus.REJECTED,
        AgentCandidateStatus.EVALUATED,
        AgentCandidateStatus.ARCHIVED,
    }
    if bundle.sealed and not terminal:
        raise AgentDesignArchiveError("sealed bundle requires terminal candidate status")
    if bundle.sealed:
        try:
            validate_sealed_terminal_evidence_bundle(bundle)
        except ValueError as exc:
            raise AgentDesignArchiveError(str(exc)) from exc
        if bundle.selection_outcome is SearchSelectionOutcome.SELECT_CANDIDATE:
            if record.status in {AgentCandidateStatus.STATIC_GATE_FAILED, AgentCandidateStatus.REJECTED}:
                raise AgentDesignArchiveError("failed or rejected candidates cannot be selected")


def freeze_evidence_bundle(bundle: AgentCandidateEvidenceBundle) -> AgentCandidateEvidenceBundle:
    payload = json.loads(bundle.model_dump_json())
    return AgentCandidateEvidenceBundle.model_validate(payload)


def _ensure_global_candidate_scope(
    history: dict[str, list[AgentCandidateRecord]],
    record: AgentCandidateRecord,
    *,
    tenant_id: str,
    workspace_id: str,
) -> None:
    existing = history.get(record.candidate_id)
    if not existing:
        return
    anchor = existing[0]
    if anchor.tenant_id != tenant_id or anchor.workspace_id != workspace_id:
        raise AgentDesignArchiveScopeError("candidate identity scope mismatch")


class InMemoryAgentDesignArchive:
    """Process-local append-only archive."""

    def __init__(self) -> None:
        self._history: dict[str, list[AgentCandidateRecord]] = {}
        self._snapshots: dict[str, AgentCandidateRecord] = {}
        self._bundles: dict[str, AgentCandidateEvidenceBundle] = {}

    def _get_scoped_snapshot(
        self,
        snapshot_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord | None:
        record = self._snapshots.get(snapshot_id)
        if record is None:
            return None
        if record.tenant_id != tenant_id or record.workspace_id != workspace_id:
            return None
        return record

    def append_candidate_snapshot(
        self,
        record: AgentCandidateRecord,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord:
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
        _assert_record_write_scope(record, tenant_id=tenant_id, workspace_id=workspace_id)
        frozen = freeze_candidate_record(record)
        _ensure_global_candidate_scope(self._history, frozen, tenant_id=tenant_id, workspace_id=workspace_id)
        prior = self._history.get(frozen.candidate_id, [None])[-1]  # type: ignore[list-item]
        if frozen.snapshot_id in self._snapshots:
            existing = self._snapshots[frozen.snapshot_id]
            if existing.model_dump(mode="json") == frozen.model_dump(mode="json"):
                return existing
            raise AgentDesignArchiveError("conflicting snapshot identity")
        validate_parent_link(self._history, frozen)
        validate_transition(prior, frozen)
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
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
        history = self._history.get(candidate_id)
        if not history:
            return None
        latest = history[-1]
        if latest.tenant_id != tenant_id or latest.workspace_id != workspace_id:
            return None
        return latest

    def get_candidate_history(
        self,
        candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[AgentCandidateRecord]:
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
        history = self._history.get(candidate_id, [])
        if not history:
            return []
        if history[0].tenant_id != tenant_id or history[0].workspace_id != workspace_id:
            return []
        return list(history)

    def list_candidates_for_objective(
        self,
        objective_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
        status: AgentCandidateStatus | None = None,
    ) -> list[AgentCandidateRecord]:
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
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
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
        parent = self.get_latest_candidate_snapshot(parent_candidate_id, tenant_id=tenant_id, workspace_id=workspace_id)
        if parent is None:
            return []
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
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
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
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
        if bundle.tenant_id != tenant_id or bundle.workspace_id != workspace_id:
            raise AgentDesignArchiveScopeError("bundle scope mismatch")
        if not bundle.sealed:
            raise AgentDesignArchiveError("bundle must be sealed before append")
        archived = self._get_scoped_snapshot(
            bundle.candidate_record.snapshot_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        validate_evidence_bundle_against_archive(bundle, archived)
        digest = bundle.bundle_digest or compute_evidence_bundle_digest(bundle)
        payload = bundle.model_dump(mode="python")
        payload["bundle_digest"] = digest
        frozen = freeze_evidence_bundle(AgentCandidateEvidenceBundle.model_validate(payload))
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
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
        bundle = self._bundles.get(bundle_id)
        if bundle is None:
            return None
        if bundle.tenant_id != tenant_id or bundle.workspace_id != workspace_id:
            return None
        if bundle.bundle_digest is not None:
            verify_evidence_bundle_digest(bundle)
        return bundle
