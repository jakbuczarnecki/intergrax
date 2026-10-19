# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Spanner primary transport.

``open_spanner_sdk_database`` always refuses a live session. It has no switch
that can import or call a vendor client. ``SpannerPrimaryTransport`` executes
the documented transaction argument shapes against the primary row image.
``SpannerSdkPrimaryBridge`` translates those same shapes onto an injected
database session. Constructing the bridge does not open a session, acquire a
credential, or contact a network. Neither path qualifies the vendor.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import NoReturn, Protocol

from intergrax.contracts.governance.history_continuity import HistoryContinuityIdentityBinding
from intergrax.runtime.governance.continuity_providers.credentials import (
    OfflineProviderRefusal,
    evaluate_continuity_credentials,
)
from intergrax.runtime.governance.continuity_providers.errors import (
    ProviderFault,
    map_transport_exception,
)
from intergrax.runtime.governance.continuity_providers.primary_register import (
    HeadRecord,
    PrimaryCommitKind,
    PrimaryConflict,
    PrimaryStep,
    PrimaryTransactionPort,
    TransitionRecord,
    _TxnState,
    _only_changed_head,
    _only_changed_transition,
    put_scoped_head,
    put_scoped_transition,
    read_scoped_head,
    read_scoped_transition,
)
from intergrax.runtime.governance.continuity_providers.targets import SpannerPrimaryTarget

CONTINUITY_HEAD_TABLE = "continuity_head"
CONTINUITY_TRANSITION_TABLE = "continuity_transition"
CONTINUITY_HEAD_COLUMNS = (
    "application_id",
    "environment_id",
    "tenant_id",
    "revision",
    "fencing_token",
    "status",
    "pis_history_id",
    "pis_watermark",
    "gca_history_id",
    "gca_watermark",
    "identity_epoch",
    "proof_revision",
    "phase",
    "transition_id",
    "request_digest",
    "authority_incarnation",
    "witness_segment",
    "witness_offset",
    "seal_digest",
)
CONTINUITY_TRANSITION_COLUMNS = (
    "application_id",
    "environment_id",
    "tenant_id",
    "transition_id",
    "request_digest",
    "revision",
    "fencing_token",
    "binding_digest",
    "status",
    "identity_epoch",
    "phase",
    "pis_history_id",
    "pis_watermark",
    "gca_history_id",
    "gca_watermark",
    "witness_segment",
    "witness_offset",
    "seal_digest",
)
_HEAD_PREDICATES = ("application_id", "environment_id", "tenant_id")
_TRANSITION_PREDICATES = ("application_id", "environment_id", "tenant_id", "transition_id")
_SpannerScalar = str | int | None


@dataclass(frozen=True, slots=True)
class ExecuteSql:
    """Arguments of ``Transaction.execute_sql``. No client is constructed."""

    sql: str
    params: tuple[tuple[str, str], ...]
    param_types: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class InsertOrUpdate:
    """Arguments of ``Transaction.insert_or_update``. No client is constructed."""

    table: str
    columns: tuple[str, ...]
    values: tuple[tuple[_SpannerScalar, ...], ...]


def cloud_execution_authorized() -> bool:
    return False


def open_spanner_sdk_database(
    target: SpannerPrimaryTarget,
    binding: HistoryContinuityIdentityBinding | None,
) -> NoReturn:
    """Refuse a live session before any client import or network call."""

    decision = evaluate_continuity_credentials(binding)
    if type(target) is not SpannerPrimaryTarget:
        raise OfflineProviderRefusal(decision.refused.value)
    raise OfflineProviderRefusal(decision.refused.value)


def _as_str(value: _SpannerScalar) -> str:
    if type(value) is not str:
        raise PrimaryConflict(ProviderFault.CORRUPT)
    return value


def _as_int(value: _SpannerScalar) -> int:
    if type(value) is not int:
        raise PrimaryConflict(ProviderFault.CORRUPT)
    return value


def _as_optional_int(value: _SpannerScalar) -> int | None:
    if value is None:
        return None
    return _as_int(value)


def _as_optional_str(value: _SpannerScalar) -> str | None:
    if value is None:
        return None
    return _as_str(value)


def _select_sql(table: str, columns: tuple[str, ...], predicates: tuple[str, ...]) -> str:
    selected = ", ".join(columns)
    where = " AND ".join(f"{name} = @{name}" for name in predicates)
    return f"SELECT {selected} FROM {table} WHERE {where}"


def _string_params(pairs: tuple[tuple[str, str], ...], sql: str) -> ExecuteSql:
    for name, _value in pairs:
        if f"@{name}" not in sql:
            raise PrimaryConflict(ProviderFault.CORRUPT)
    return ExecuteSql(
        sql=sql,
        params=pairs,
        param_types=tuple((name, "STRING") for name, _value in pairs),
    )


def encode_head(record: HeadRecord) -> tuple[_SpannerScalar, ...]:
    return (
        record.application_id,
        record.environment_id,
        record.tenant_id,
        record.revision,
        record.fencing_token,
        record.status,
        record.pis_history_id,
        record.pis_watermark,
        record.gca_history_id,
        record.gca_watermark,
        record.identity_epoch,
        record.proof_revision,
        record.phase,
        record.transition_id,
        record.request_digest,
        record.authority_incarnation,
        record.witness_segment,
        record.witness_offset,
        record.seal_digest,
    )


def decode_head(columns: tuple[str, ...], row: tuple[_SpannerScalar, ...]) -> HeadRecord:
    if columns != CONTINUITY_HEAD_COLUMNS or len(row) != len(columns):
        raise PrimaryConflict(ProviderFault.CORRUPT)
    return HeadRecord(
        application_id=_as_str(row[0]),
        environment_id=_as_str(row[1]),
        tenant_id=_as_str(row[2]),
        revision=_as_int(row[3]),
        fencing_token=_as_str(row[4]),
        status=_as_str(row[5]),
        pis_history_id=_as_str(row[6]),
        pis_watermark=_as_int(row[7]),
        gca_history_id=_as_str(row[8]),
        gca_watermark=_as_int(row[9]),
        identity_epoch=_as_int(row[10]),
        proof_revision=_as_int(row[11]),
        phase=_as_str(row[12]),
        transition_id=_as_str(row[13]),
        request_digest=_as_str(row[14]),
        authority_incarnation=_as_str(row[15]),
        witness_segment=_as_optional_int(row[16]),
        witness_offset=_as_optional_int(row[17]),
        seal_digest=_as_optional_str(row[18]),
    )


def encode_transition(record: TransitionRecord) -> tuple[_SpannerScalar, ...]:
    return (
        record.application_id,
        record.environment_id,
        record.tenant_id,
        record.transition_id,
        record.request_digest,
        record.revision,
        record.fencing_token,
        record.binding_digest,
        record.status,
        record.identity_epoch,
        record.phase,
        record.pis_history_id,
        record.pis_watermark,
        record.gca_history_id,
        record.gca_watermark,
        record.witness_segment,
        record.witness_offset,
        record.seal_digest,
    )


def decode_transition(columns: tuple[str, ...], row: tuple[_SpannerScalar, ...]) -> TransitionRecord:
    if columns != CONTINUITY_TRANSITION_COLUMNS or len(row) != len(columns):
        raise PrimaryConflict(ProviderFault.CORRUPT)
    return TransitionRecord(
        application_id=_as_str(row[0]),
        environment_id=_as_str(row[1]),
        tenant_id=_as_str(row[2]),
        transition_id=_as_str(row[3]),
        request_digest=_as_str(row[4]),
        revision=_as_int(row[5]),
        fencing_token=_as_str(row[6]),
        binding_digest=_as_str(row[7]),
        status=_as_str(row[8]),
        identity_epoch=_as_int(row[9]),
        phase=_as_str(row[10]),
        pis_history_id=_as_str(row[11]),
        pis_watermark=_as_int(row[12]),
        gca_history_id=_as_str(row[13]),
        gca_watermark=_as_int(row[14]),
        witness_segment=_as_optional_int(row[15]),
        witness_offset=_as_optional_int(row[16]),
        seal_digest=_as_optional_str(row[17]),
    )


def apply_insert_or_update(state: _TxnState, call: InsertOrUpdate) -> None:
    """Execute one logged mutation against the primary row image."""

    if len(call.values) != 1:
        raise PrimaryConflict(ProviderFault.CORRUPT)
    row = call.values[0]
    if call.table == CONTINUITY_HEAD_TABLE and call.columns == CONTINUITY_HEAD_COLUMNS:
        put_scoped_head(state, decode_head(call.columns, row))
        return
    if call.table == CONTINUITY_TRANSITION_TABLE and call.columns == CONTINUITY_TRANSITION_COLUMNS:
        put_scoped_transition(state, decode_transition(call.columns, row))
        return
    raise PrimaryConflict(ProviderFault.CORRUPT)


def apply_spanner_calls(state: _TxnState, calls: tuple[ExecuteSql | InsertOrUpdate, ...]) -> _TxnState:
    """Replay logged calls. Reads do not change the row image."""

    replay = copy.deepcopy(state)
    for call in calls:
        if type(call) is ExecuteSql:
            continue
        if type(call) is InsertOrUpdate:
            apply_insert_or_update(replay, call)
            continue
        raise PrimaryConflict(ProviderFault.CORRUPT)
    return replay


class _EncodingTransaction:
    def __init__(self, state: _TxnState) -> None:
        self._state = state
        self.calls: list[ExecuteSql | InsertOrUpdate] = []

    def read_head(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
    ) -> HeadRecord | None:
        sql = _select_sql(CONTINUITY_HEAD_TABLE, CONTINUITY_HEAD_COLUMNS, _HEAD_PREDICATES)
        self.calls.append(
            _string_params(
                (
                    ("application_id", application_id),
                    ("environment_id", environment_id),
                    ("tenant_id", tenant_id),
                ),
                sql,
            )
        )
        return read_scoped_head(self._state, application_id, environment_id, tenant_id)

    def read_transition(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
        transition_id: str,
    ) -> TransitionRecord | None:
        sql = _select_sql(
            CONTINUITY_TRANSITION_TABLE,
            CONTINUITY_TRANSITION_COLUMNS,
            _TRANSITION_PREDICATES,
        )
        self.calls.append(
            _string_params(
                (
                    ("application_id", application_id),
                    ("environment_id", environment_id),
                    ("tenant_id", tenant_id),
                    ("transition_id", transition_id),
                ),
                sql,
            )
        )
        return read_scoped_transition(
            self._state,
            application_id,
            environment_id,
            tenant_id,
            transition_id,
        )

    def insert_transition(self, record: TransitionRecord) -> None:
        row = encode_transition(record)
        if decode_transition(CONTINUITY_TRANSITION_COLUMNS, row) != record:
            raise PrimaryConflict(ProviderFault.CORRUPT)
        call = InsertOrUpdate(CONTINUITY_TRANSITION_TABLE, CONTINUITY_TRANSITION_COLUMNS, (row,))
        apply_insert_or_update(self._state, call)
        self.calls.append(call)

    def put_head(self, record: HeadRecord) -> None:
        row = encode_head(record)
        if decode_head(CONTINUITY_HEAD_COLUMNS, row) != record:
            raise PrimaryConflict(ProviderFault.CORRUPT)
        call = InsertOrUpdate(CONTINUITY_HEAD_TABLE, CONTINUITY_HEAD_COLUMNS, (row,))
        apply_insert_or_update(self._state, call)
        self.calls.append(call)


class SpannerPrimaryTransport:
    """Offline executor for the Spanner call shapes. Not a live database client."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = _TxnState()
        self.operations: list[ExecuteSql | InsertOrUpdate] = []

    def read_head(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
    ) -> HeadRecord | None:
        with self._lock:
            sql = _select_sql(CONTINUITY_HEAD_TABLE, CONTINUITY_HEAD_COLUMNS, _HEAD_PREDICATES)
            self.operations.append(
                _string_params(
                    (
                        ("application_id", application_id),
                        ("environment_id", environment_id),
                        ("tenant_id", tenant_id),
                    ),
                    sql,
                )
            )
            return read_scoped_head(self._state, application_id, environment_id, tenant_id)

    def read_transition(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
        transition_id: str,
    ) -> TransitionRecord | None:
        with self._lock:
            sql = _select_sql(
                CONTINUITY_TRANSITION_TABLE,
                CONTINUITY_TRANSITION_COLUMNS,
                _TRANSITION_PREDICATES,
            )
            self.operations.append(
                _string_params(
                    (
                        ("application_id", application_id),
                        ("environment_id", environment_id),
                        ("tenant_id", tenant_id),
                        ("transition_id", transition_id),
                    ),
                    sql,
                )
            )
            return read_scoped_transition(
                self._state,
                application_id,
                environment_id,
                tenant_id,
                transition_id,
            )

    def run(self, work: Callable[[PrimaryTransactionPort], None]) -> PrimaryStep:
        with self._lock:
            txn = _EncodingTransaction(copy.deepcopy(self._state))
            try:
                work(txn)
            except PrimaryConflict as exc:
                return PrimaryStep(kind=PrimaryCommitKind.CONFLICT, fault=exc.fault)
            try:
                replayed = apply_spanner_calls(self._state, tuple(txn.calls))
            except PrimaryConflict as exc:
                return PrimaryStep(kind=PrimaryCommitKind.CONFLICT, fault=exc.fault)
            if replayed != txn._state:
                return PrimaryStep(
                    kind=PrimaryCommitKind.UNKNOWN,
                    fault=ProviderFault.OUTCOME_UNKNOWN,
                )
            self._state = txn._state
            self.operations.extend(txn.calls)
            return PrimaryStep(
                kind=PrimaryCommitKind.COMMITTED,
                head=_only_changed_head(txn._state),
                transition=_only_changed_transition(txn._state),
            )


class SpannerTransportError(Exception):
    """A strong read failed. The absence of a row is not this error."""

    def __init__(self, fault: ProviderFault) -> None:
        super().__init__(fault.value)
        self.fault = fault


class SpannerReadSession(Protocol):
    """Read methods of a transaction or a strong snapshot."""

    def execute_sql(
        self,
        sql: str,
        params: Mapping[str, str] | None = None,
        param_types: Mapping[str, Mapping[str, str]] | None = None,
    ) -> Iterable[Sequence[_SpannerScalar]]:
        """Return rows for one scoped query. Parameter types are SDK messages."""
        ...


class SpannerTransactionSession(SpannerReadSession, Protocol):
    """Transaction callback argument. Mutations stay on this object."""

    def insert_or_update(
        self,
        table: str,
        columns: Sequence[str],
        values: Sequence[Sequence[_SpannerScalar]],
    ) -> None:
        """Buffer one mutation until the database transaction commits."""
        ...


class SpannerDatabaseSession(Protocol):
    """Injected database. ``snapshot`` is strong: no staleness arguments."""

    def snapshot(self) -> AbstractContextManager[SpannerReadSession]:
        """Open one strong read. Stale and directed reads are not this method."""
        ...

    def run_in_transaction(self, func: Callable[..., None]) -> None:
        """Run one read-write attempt. The callback receives the transaction."""
        ...


@dataclass(slots=True)
class _WrittenImage:
    head: HeadRecord | None = None
    transition: TransitionRecord | None = None


def _param_type_message(code: str) -> dict[str, str]:
    if code != "STRING" and code != "INT64":
        raise PrimaryConflict(ProviderFault.CORRUPT)
    return {"code": code}


def _one_row(result: Iterable[Sequence[_SpannerScalar]]) -> tuple[_SpannerScalar, ...] | None:
    found: tuple[_SpannerScalar, ...] | None = None
    for row in result:
        if found is not None:
            raise PrimaryConflict(ProviderFault.CORRUPT)
        found = tuple(row)
    return found


def _invoke_execute_sql(
    session: SpannerReadSession,
    call: ExecuteSql,
) -> tuple[_SpannerScalar, ...] | None:
    params = {name: value for name, value in call.params}
    param_types = {name: _param_type_message(code) for name, code in call.param_types}
    return _one_row(session.execute_sql(call.sql, params=params, param_types=param_types))


def _invoke_insert_or_update(session: SpannerTransactionSession, call: InsertOrUpdate) -> None:
    if len(call.values) != 1:
        raise PrimaryConflict(ProviderFault.CORRUPT)
    session.insert_or_update(
        call.table,
        list(call.columns),
        [list(row) for row in call.values],
    )


def _head_query(application_id: str, environment_id: str, tenant_id: str) -> ExecuteSql:
    return _string_params(
        (
            ("application_id", application_id),
            ("environment_id", environment_id),
            ("tenant_id", tenant_id),
        ),
        _select_sql(CONTINUITY_HEAD_TABLE, CONTINUITY_HEAD_COLUMNS, _HEAD_PREDICATES),
    )


def _transition_query(
    application_id: str,
    environment_id: str,
    tenant_id: str,
    transition_id: str,
) -> ExecuteSql:
    return _string_params(
        (
            ("application_id", application_id),
            ("environment_id", environment_id),
            ("tenant_id", tenant_id),
            ("transition_id", transition_id),
        ),
        _select_sql(
            CONTINUITY_TRANSITION_TABLE,
            CONTINUITY_TRANSITION_COLUMNS,
            _TRANSITION_PREDICATES,
        ),
    )


class _SdkPrimaryTransaction:
    """Translate one register attempt. Continuity rules stay in the register."""

    def __init__(self, session: SpannerTransactionSession, image: _WrittenImage) -> None:
        self._session = session
        self._image = image

    def read_head(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
    ) -> HeadRecord | None:
        row = _invoke_execute_sql(self._session, _head_query(application_id, environment_id, tenant_id))
        if row is None:
            return None
        return decode_head(CONTINUITY_HEAD_COLUMNS, row)

    def read_transition(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
        transition_id: str,
    ) -> TransitionRecord | None:
        row = _invoke_execute_sql(
            self._session,
            _transition_query(application_id, environment_id, tenant_id, transition_id),
        )
        if row is None:
            return None
        return decode_transition(CONTINUITY_TRANSITION_COLUMNS, row)

    def insert_transition(self, record: TransitionRecord) -> None:
        row = encode_transition(record)
        if decode_transition(CONTINUITY_TRANSITION_COLUMNS, row) != record:
            raise PrimaryConflict(ProviderFault.CORRUPT)
        _invoke_insert_or_update(
            self._session,
            InsertOrUpdate(CONTINUITY_TRANSITION_TABLE, CONTINUITY_TRANSITION_COLUMNS, (row,)),
        )
        self._image.transition = decode_transition(CONTINUITY_TRANSITION_COLUMNS, row)

    def put_head(self, record: HeadRecord) -> None:
        row = encode_head(record)
        if decode_head(CONTINUITY_HEAD_COLUMNS, row) != record:
            raise PrimaryConflict(ProviderFault.CORRUPT)
        _invoke_insert_or_update(
            self._session,
            InsertOrUpdate(CONTINUITY_HEAD_TABLE, CONTINUITY_HEAD_COLUMNS, (row,)),
        )
        self._image.head = decode_head(CONTINUITY_HEAD_COLUMNS, row)


class SpannerSdkPrimaryBridge:
    """Primary port over an injected database session. Not a session factory."""

    def __init__(self, database: SpannerDatabaseSession) -> None:
        self._database = database
        self._lock = threading.RLock()

    def read_head(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
    ) -> HeadRecord | None:
        with self._lock:
            return self._read_head(application_id, environment_id, tenant_id)

    def read_transition(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
        transition_id: str,
    ) -> TransitionRecord | None:
        with self._lock:
            return self._read_transition(application_id, environment_id, tenant_id, transition_id)

    def run(self, work: Callable[[PrimaryTransactionPort], None]) -> PrimaryStep:
        with self._lock:
            image = _WrittenImage()

            def attempt(transaction: SpannerTransactionSession) -> None:
                work(_SdkPrimaryTransaction(transaction, image))

            try:
                self._database.run_in_transaction(attempt)
            except PrimaryConflict as exc:
                return PrimaryStep(kind=PrimaryCommitKind.CONFLICT, fault=exc.fault)
            except Exception as exc:
                return self._step_for_transport_failure(exc, image)
            return PrimaryStep(
                kind=PrimaryCommitKind.COMMITTED,
                head=image.head,
                transition=image.transition,
            )

    def _step_for_transport_failure(self, exc: BaseException, image: _WrittenImage) -> PrimaryStep:
        if isinstance(exc, SpannerTransportError):
            fault = exc.fault
        else:
            fault = map_transport_exception(exc)
        if fault is ProviderFault.OUTCOME_UNKNOWN:
            return self._confirm_committed(image)
        if fault is ProviderFault.REVISION_CONFLICT:
            return PrimaryStep(kind=PrimaryCommitKind.CONFLICT, fault=fault)
        return PrimaryStep(kind=PrimaryCommitKind.UNAVAILABLE, fault=ProviderFault.UNAVAILABLE)

    def _confirm_committed(self, image: _WrittenImage) -> PrimaryStep:
        unknown = PrimaryStep(
            kind=PrimaryCommitKind.UNKNOWN,
            fault=ProviderFault.OUTCOME_UNKNOWN,
        )
        if image.head is None and image.transition is None:
            return unknown
        try:
            if image.head is not None and self._read_head(
                image.head.application_id,
                image.head.environment_id,
                image.head.tenant_id,
            ) != image.head:
                return unknown
            if image.transition is not None and self._read_transition(
                image.transition.application_id,
                image.transition.environment_id,
                image.transition.tenant_id,
                image.transition.transition_id,
            ) != image.transition:
                return unknown
        except (SpannerTransportError, PrimaryConflict):
            return unknown
        return PrimaryStep(
            kind=PrimaryCommitKind.COMMITTED,
            head=image.head,
            transition=image.transition,
        )

    def _read_head(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
    ) -> HeadRecord | None:
        row = self._strong_read(_head_query(application_id, environment_id, tenant_id))
        if row is None:
            return None
        return decode_head(CONTINUITY_HEAD_COLUMNS, row)

    def _read_transition(
        self,
        application_id: str,
        environment_id: str,
        tenant_id: str,
        transition_id: str,
    ) -> TransitionRecord | None:
        row = self._strong_read(
            _transition_query(application_id, environment_id, tenant_id, transition_id)
        )
        if row is None:
            return None
        return decode_transition(CONTINUITY_TRANSITION_COLUMNS, row)

    def _strong_read(self, call: ExecuteSql) -> tuple[_SpannerScalar, ...] | None:
        try:
            with self._database.snapshot() as snapshot:
                return _invoke_execute_sql(snapshot, call)
        except PrimaryConflict:
            raise
        except Exception as exc:
            raise SpannerTransportError(map_transport_exception(exc)) from exc
