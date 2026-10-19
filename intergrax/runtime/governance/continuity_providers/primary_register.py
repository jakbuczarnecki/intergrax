# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Spanner-shaped primary register: transactional compare-and-advance.

The state machine runs against a transaction port. The offline port is
in-process. The Google Cloud client is a separate module and is not opened
here. Rows are isolated by application, environment, and tenant.
"""

from __future__ import annotations

import copy
import hashlib
import threading
from dataclasses import dataclass, field
from enum import StrEnum

from intergrax.runtime.governance.continuity_providers.errors import ProviderFault

_COMMITTED = "committed"
_PREPARED = "prepared"
_WITNESS_SEALED = "witness_sealed"


class PrimaryCommitKind(StrEnum):
    COMMITTED = "committed"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


class PrimaryConflict(Exception):
    def __init__(self, fault: ProviderFault) -> None:
        super().__init__(fault.value)
        self.fault = fault


@dataclass
class HeadRecord:
    application_id: str
    environment_id: str
    tenant_id: str
    revision: int
    fencing_token: str
    status: str
    pis_history_id: str
    pis_watermark: int
    gca_history_id: str
    gca_watermark: int
    identity_epoch: int
    proof_revision: int
    phase: str
    transition_id: str
    request_digest: str
    authority_incarnation: str
    witness_segment: int | None = None
    witness_offset: int | None = None
    seal_digest: str | None = None


@dataclass
class TransitionRecord:
    application_id: str
    environment_id: str
    tenant_id: str
    transition_id: str
    request_digest: str
    revision: int
    fencing_token: str
    binding_digest: str
    status: str
    identity_epoch: int
    phase: str
    pis_history_id: str
    pis_watermark: int
    gca_history_id: str
    gca_watermark: int
    witness_segment: int | None = None
    witness_offset: int | None = None
    seal_digest: str | None = None


def request_digest(
    *,
    application_id: str,
    environment_id: str,
    tenant_id: str,
    revision: int,
    fencing_token: str,
    transition_id: str,
    binding_digest: str,
    status: str,
    identity_epoch: int,
    pis_history_id: str,
    pis_watermark: int,
    gca_history_id: str,
    gca_watermark: int,
) -> str:
    """Immutable digest of one transition request. Same bytes resume; others deny."""

    parts = (
        application_id,
        environment_id,
        tenant_id,
        str(revision),
        fencing_token,
        transition_id,
        binding_digest,
        status,
        str(identity_epoch),
        pis_history_id,
        str(pis_watermark),
        gca_history_id,
        str(gca_watermark),
    )
    body = bytearray(len(parts).to_bytes(4, "big"))
    for part in parts:
        encoded = part.encode("utf-8")
        body += len(encoded).to_bytes(4, "big")
        body += encoded
    return hashlib.sha256(bytes(body)).hexdigest()


ScopeKey = tuple[str, str, str]


def _scope_key(application_id: str, environment_id: str, tenant_id: str) -> ScopeKey:
    return (application_id, environment_id, tenant_id)


@dataclass
class _TxnState:
    heads: dict[ScopeKey, HeadRecord] = field(default_factory=dict)
    transitions: dict[tuple[ScopeKey, str], TransitionRecord] = field(default_factory=dict)


class PrimaryTransaction:
    """One serializable attempt. Conflicts raise before the caller commits."""

    def __init__(self, state: _TxnState) -> None:
        self._state = state

    def read_head(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
    ) -> HeadRecord | None:
        found = self._state.heads.get(_scope_key(application_id, environment_id, tenant_id))
        return None if found is None else copy.deepcopy(found)

    def read_transition(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
        transition_id: str,
    ) -> TransitionRecord | None:
        key = (_scope_key(application_id, environment_id, tenant_id), transition_id)
        found = self._state.transitions.get(key)
        return None if found is None else copy.deepcopy(found)

    def insert_transition(self, record: TransitionRecord) -> None:
        key = (
            _scope_key(record.application_id, record.environment_id, record.tenant_id),
            record.transition_id,
        )
        existing = self._state.transitions.get(key)
        if existing is not None and existing.request_digest != record.request_digest:
            raise PrimaryConflict(ProviderFault.DIGEST_MISMATCH)
        self._state.transitions[key] = copy.deepcopy(record)

    def put_head(self, record: HeadRecord) -> None:
        self._state.heads[_scope_key(record.application_id, record.environment_id, record.tenant_id)] = (
            copy.deepcopy(record)
        )


@dataclass(frozen=True, slots=True)
class PrimaryStep:
    kind: PrimaryCommitKind
    fault: ProviderFault | None = None
    head: HeadRecord | None = None
    transition: TransitionRecord | None = None


class InMemoryPrimaryDatabase:
    """Offline transaction port. ``offline_contract_transport`` marks this double.

    Faults are one-shot and explicit. An unknown outcome after apply leaves the
    write durable. An unknown outcome before apply leaves it absent.
    """

    offline_contract_transport = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = _TxnState()
        self._history: dict[ScopeKey, dict[int, HeadRecord]] = {}
        self._calls = 0
        self._fault: str | None = None
        self._fault_call = 1

    def arm_fault(self, mode: str, *, call_index: int = 1) -> None:
        if mode not in {"unknown_before", "unknown_after", "unavailable"}:
            raise ValueError("unsupported primary fault")
        if call_index < 1:
            raise ValueError("fault call index starts at 1")
        with self._lock:
            self._fault = mode
            self._fault_call = call_index
            self._calls = 0

    def clear_fault(self) -> None:
        with self._lock:
            self._fault = None
            self._calls = 0

    def read_head(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
    ) -> HeadRecord | None:
        with self._lock:
            found = self._state.heads.get(_scope_key(application_id, environment_id, tenant_id))
            return None if found is None else copy.deepcopy(found)

    def read_transition(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
        transition_id: str,
    ) -> TransitionRecord | None:
        with self._lock:
            key = (_scope_key(application_id, environment_id, tenant_id), transition_id)
            found = self._state.transitions.get(key)
            return None if found is None else copy.deepcopy(found)

    def testing_rollback_head(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
    ) -> None:
        """Move the committed head to the previous revision. The witness is untouched."""

        key = _scope_key(application_id, environment_id, tenant_id)
        with self._lock:
            current = self._state.heads.get(key)
            if current is None or current.phase != _COMMITTED or current.revision < 2:
                raise RuntimeError("primary rollback requires a committed revision above 1")
            previous = self._history.get(key, {}).get(current.revision - 1)
            if previous is None:
                raise RuntimeError("previous committed head is absent")
            self._state.heads[key] = copy.deepcopy(previous)

    def run(self, work) -> PrimaryStep:
        with self._lock:
            self._calls += 1
            snapshot = copy.deepcopy(self._state)
            txn = PrimaryTransaction(snapshot)
            try:
                work(txn)
            except PrimaryConflict as exc:
                return PrimaryStep(kind=PrimaryCommitKind.CONFLICT, fault=exc.fault)
            fault = self._fault if self._calls == self._fault_call else None
            if fault is not None:
                self._fault = None
            if fault == "unavailable":
                return PrimaryStep(kind=PrimaryCommitKind.UNAVAILABLE, fault=ProviderFault.UNAVAILABLE)
            if fault == "unknown_before":
                return PrimaryStep(kind=PrimaryCommitKind.UNKNOWN, fault=ProviderFault.OUTCOME_UNKNOWN)
            self._state = snapshot
            self._remember_committed_heads()
            head = _only_changed_head(snapshot)
            transition = _only_changed_transition(snapshot)
            if fault == "unknown_after":
                return PrimaryStep(
                    kind=PrimaryCommitKind.UNKNOWN,
                    fault=ProviderFault.OUTCOME_UNKNOWN,
                    head=head,
                    transition=transition,
                )
            return PrimaryStep(
                kind=PrimaryCommitKind.COMMITTED,
                head=head,
                transition=transition,
            )

    def _remember_committed_heads(self) -> None:
        for key, head in self._state.heads.items():
            if head.phase == _COMMITTED:
                self._history.setdefault(key, {})[head.revision] = copy.deepcopy(head)


def _only_changed_head(state: _TxnState) -> HeadRecord | None:
    if len(state.heads) == 1:
        return copy.deepcopy(next(iter(state.heads.values())))
    return None


def _only_changed_transition(state: _TxnState) -> TransitionRecord | None:
    if len(state.transitions) == 1:
        return copy.deepcopy(next(iter(state.transitions.values())))
    return None


def _require_scope(record_app: str, record_env: str, record_tenant: str, **scope: str) -> None:
    if (
        record_app != scope["application_id"]
        or record_env != scope["environment_id"]
        or record_tenant != scope["tenant_id"]
    ):
        raise PrimaryConflict(ProviderFault.SCOPE_MISMATCH)


class SpannerPrimaryRegister:
    """Compare-and-advance rules for one scoped head. Vendor SQL is not this class."""

    def prepare(
        self,
        database: InMemoryPrimaryDatabase,
        *,
        application_id: str,
        environment_id: str,
        tenant_id: str,
        expected_revision: int,
        expected_fencing_token: str | None,
        transition: TransitionRecord,
        authority_incarnation: str,
    ) -> PrimaryStep:
        observed: dict[str, HeadRecord | TransitionRecord | None] = {}

        def work(txn: PrimaryTransaction) -> None:
            head = txn.read_head(application_id, environment_id, tenant_id)
            existing = txn.read_transition(
                application_id,
                environment_id,
                tenant_id,
                transition.transition_id,
            )
            if existing is not None:
                _require_scope(
                    existing.application_id,
                    existing.environment_id,
                    existing.tenant_id,
                    application_id=application_id,
                    environment_id=environment_id,
                    tenant_id=tenant_id,
                )
                if existing.request_digest != transition.request_digest:
                    raise PrimaryConflict(ProviderFault.DIGEST_MISMATCH)
                observed["head"] = head
                observed["transition"] = existing
                return
            if expected_revision == 0:
                if head is not None:
                    raise PrimaryConflict(ProviderFault.ALREADY_INITIALIZED)
            else:
                if head is None:
                    raise PrimaryConflict(ProviderFault.NOT_INITIALIZED)
                _require_scope(
                    head.application_id,
                    head.environment_id,
                    head.tenant_id,
                    application_id=application_id,
                    environment_id=environment_id,
                    tenant_id=tenant_id,
                )
                if head.phase != _COMMITTED:
                    raise PrimaryConflict(ProviderFault.PENDING)
                if head.revision != expected_revision:
                    raise PrimaryConflict(ProviderFault.REVISION_CONFLICT)
                if head.fencing_token != expected_fencing_token:
                    raise PrimaryConflict(ProviderFault.FENCE_CONFLICT)
            txn.insert_transition(transition)
            base = head
            prepared_head = HeadRecord(
                application_id=application_id,
                environment_id=environment_id,
                tenant_id=tenant_id,
                revision=0 if base is None else base.revision,
                fencing_token="" if base is None else base.fencing_token,
                status="" if base is None else base.status,
                pis_history_id="" if base is None else base.pis_history_id,
                pis_watermark=0 if base is None else base.pis_watermark,
                gca_history_id="" if base is None else base.gca_history_id,
                gca_watermark=0 if base is None else base.gca_watermark,
                identity_epoch=0 if base is None else base.identity_epoch,
                proof_revision=0 if base is None else base.proof_revision,
                phase=_PREPARED,
                transition_id=transition.transition_id,
                request_digest=transition.request_digest,
                authority_incarnation=authority_incarnation
                if base is None
                else base.authority_incarnation,
                witness_segment=None if base is None else base.witness_segment,
                witness_offset=None if base is None else base.witness_offset,
                seal_digest=None if base is None else base.seal_digest,
            )
            if base is None:
                prepared_head.authority_incarnation = authority_incarnation
            txn.put_head(prepared_head)

        step = database.run(work)
        if step.kind is PrimaryCommitKind.CONFLICT:
            return step
        if step.kind is PrimaryCommitKind.UNAVAILABLE:
            return step
        if step.kind is PrimaryCommitKind.UNKNOWN and step.head is None:
            return step
        fresh_head = database.read_head(application_id, environment_id, tenant_id)
        fresh_transition = database.read_transition(
            application_id,
            environment_id,
            tenant_id,
            transition.transition_id,
        )
        if fresh_transition is None or fresh_transition.request_digest != transition.request_digest:
            return PrimaryStep(
                kind=PrimaryCommitKind.UNKNOWN,
                fault=ProviderFault.OUTCOME_UNKNOWN,
            )
        return PrimaryStep(
            kind=PrimaryCommitKind.COMMITTED,
            head=fresh_head,
            transition=fresh_transition,
        )

    def attach_receipt(
        self,
        database: InMemoryPrimaryDatabase,
        *,
        application_id: str,
        environment_id: str,
        tenant_id: str,
        transition_id: str,
        request_digest_value: str,
        witness_segment: int,
        witness_offset: int,
        seal_digest: str,
    ) -> PrimaryStep:
        def work(txn: PrimaryTransaction) -> None:
            head = txn.read_head(application_id, environment_id, tenant_id)
            transition = txn.read_transition(
                application_id,
                environment_id,
                tenant_id,
                transition_id,
            )
            if head is None or transition is None:
                raise PrimaryConflict(ProviderFault.NOT_INITIALIZED)
            if transition.request_digest != request_digest_value:
                raise PrimaryConflict(ProviderFault.DIGEST_MISMATCH)
            if head.phase == _WITNESS_SEALED and transition.phase == _WITNESS_SEALED:
                if (
                    transition.seal_digest == seal_digest
                    and transition.witness_offset == witness_offset
                    and transition.witness_segment == witness_segment
                ):
                    return
                raise PrimaryConflict(ProviderFault.DIGEST_MISMATCH)
            if head.phase != _PREPARED or transition.phase != _PREPARED:
                raise PrimaryConflict(ProviderFault.PENDING)
            if head.transition_id != transition_id or head.request_digest != request_digest_value:
                raise PrimaryConflict(ProviderFault.DIGEST_MISMATCH)
            transition.phase = _WITNESS_SEALED
            transition.witness_segment = witness_segment
            transition.witness_offset = witness_offset
            transition.seal_digest = seal_digest
            head.phase = _WITNESS_SEALED
            head.witness_segment = witness_segment
            head.witness_offset = witness_offset
            head.seal_digest = seal_digest
            txn.insert_transition(transition)
            txn.put_head(head)

        return self._reread(
            database,
            work,
            application_id=application_id,
            environment_id=environment_id,
            tenant_id=tenant_id,
            transition_id=transition_id,
            expected_phase=_WITNESS_SEALED,
        )

    def commit_transition(
        self,
        database: InMemoryPrimaryDatabase,
        *,
        application_id: str,
        environment_id: str,
        tenant_id: str,
        transition_id: str,
        request_digest_value: str,
    ) -> PrimaryStep:
        def work(txn: PrimaryTransaction) -> None:
            head = txn.read_head(application_id, environment_id, tenant_id)
            transition = txn.read_transition(
                application_id,
                environment_id,
                tenant_id,
                transition_id,
            )
            if head is None or transition is None:
                raise PrimaryConflict(ProviderFault.NOT_INITIALIZED)
            if transition.request_digest != request_digest_value:
                raise PrimaryConflict(ProviderFault.DIGEST_MISMATCH)
            if head.phase == _COMMITTED and head.revision == transition.revision:
                if head.request_digest != request_digest_value:
                    raise PrimaryConflict(ProviderFault.DIGEST_MISMATCH)
                return
            if head.phase != _WITNESS_SEALED or transition.phase != _WITNESS_SEALED:
                raise PrimaryConflict(ProviderFault.PENDING)
            if (
                head.seal_digest != transition.seal_digest
                or transition.seal_digest is None
                or head.transition_id != transition_id
            ):
                raise PrimaryConflict(ProviderFault.DIGEST_MISMATCH)
            head.revision = transition.revision
            head.fencing_token = transition.fencing_token
            head.status = transition.status
            head.pis_history_id = transition.pis_history_id
            head.pis_watermark = transition.pis_watermark
            head.gca_history_id = transition.gca_history_id
            head.gca_watermark = transition.gca_watermark
            head.identity_epoch = transition.identity_epoch
            head.proof_revision = transition.revision
            head.phase = _COMMITTED
            head.transition_id = transition.transition_id
            head.request_digest = transition.request_digest
            transition.phase = _COMMITTED
            txn.insert_transition(transition)
            txn.put_head(head)

        return self._reread(
            database,
            work,
            application_id=application_id,
            environment_id=environment_id,
            tenant_id=tenant_id,
            transition_id=transition_id,
            expected_phase=_COMMITTED,
        )

    def _reread(
        self,
        database: InMemoryPrimaryDatabase,
        work,
        *,
        application_id: str,
        environment_id: str,
        tenant_id: str,
        transition_id: str,
        expected_phase: str,
    ) -> PrimaryStep:
        step = database.run(work)
        if step.kind is PrimaryCommitKind.CONFLICT or step.kind is PrimaryCommitKind.UNAVAILABLE:
            return step
        head = database.read_head(application_id, environment_id, tenant_id)
        transition = database.read_transition(
            application_id,
            environment_id,
            tenant_id,
            transition_id,
        )
        if (
            head is None
            or transition is None
            or head.phase != expected_phase
            or transition.phase != expected_phase
        ):
            return PrimaryStep(kind=PrimaryCommitKind.UNKNOWN, fault=ProviderFault.OUTCOME_UNKNOWN)
        return PrimaryStep(kind=PrimaryCommitKind.COMMITTED, head=head, transition=transition)
