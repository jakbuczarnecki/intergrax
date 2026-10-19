# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SDK call bridges for the offline Spanner and Azure providers.

Recording doubles use the vendor method signatures. The suite does not import a
vendor SDK, open a session, or contact a network.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
from typing import cast

import pytest

from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.applications._shared.history_continuity_composition import (
    compose_history_continuity_authority,
)
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.governance import history_continuity as continuity_contract
from intergrax.contracts.governance.history_continuity import (
    ContinuityPermission,
    HistoryBinding,
    HistoryContinuityAdapterKind,
    HistoryContinuityAdministrativeIdentities,
    HistoryContinuityCloudQualification,
    HistoryContinuityExecutionDomain,
    HistoryContinuityIdentityBinding,
    HistoryContinuityProviderSelection,
    HistoryContinuityRetentionPolicy,
    HistoryContinuityRuntimeCredentialMode,
)
from intergrax.contracts.runtime_policy_bundle import (
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.runtime.governance.continuity_providers import azure_sdk, spanner_sdk
from intergrax.runtime.governance.continuity_providers.azure_sdk import (
    AzureSdkAppendBridge,
    WitnessStorageUnavailable,
    witness_blob_name,
)
from intergrax.runtime.governance.continuity_providers.credentials import OfflineProviderRefusal
from intergrax.runtime.governance.continuity_providers.errors import ProviderFault
from intergrax.runtime.governance.continuity_providers.primary_register import (
    HeadRecord,
    InMemoryPrimaryDatabase,
    PrimaryCommitKind,
    PrimaryConflict,
    PrimaryStep,
    SpannerPrimaryRegister,
    TransitionRecord,
    request_digest,
)
from intergrax.runtime.governance.continuity_providers.spanner_sdk import (
    CONTINUITY_HEAD_COLUMNS,
    SpannerSdkPrimaryBridge,
    open_spanner_sdk_database,
)
from intergrax.runtime.governance.continuity_providers.targets import (
    AzureWitnessTarget,
    SpannerPrimaryTarget,
)
from intergrax.runtime.governance.continuity_providers.witness_log import (
    AzureAppendWitness,
    InMemoryAppendWitness,
    block_digest,
    decode_block,
    encode_block,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.control_plane_mutation_policy import (
    bundle_backed_control_plane_mutation_evaluator,
)
from intergrax.runtime.governance.local_history_continuity import (
    InactiveHistoryContinuityAuthority,
    continuity_authorization_action,
    continuity_binding_digest,
)
from intergrax.runtime.governance.offline_history_continuity import OfflineProviderBundle
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import RuntimePolicyBundleEvaluator

_APP = "app-1"
_ENV = "env-1"
_BLOCK = 4096
_Scalar = str | int | None
_Row = dict[str, _Scalar]


class DeadlineExceeded(Exception):
    pass


class Aborted(Exception):
    pass


class ServiceUnavailable(Exception):
    pass


class HttpResponseError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("storage")
        self.status_code = status_code


class ResourceNotFoundError(HttpResponseError):
    def __init__(self) -> None:
        super().__init__(404)


class _SqlCall:
    def __init__(
        self,
        sql: str,
        params: dict[str, str] | None,
        param_types: dict[str, dict[str, str]] | None,
        *,
        query_mode: object,
        directed_read_options: object,
        data_boost_enabled: bool,
    ) -> None:
        self.sql = sql
        self.params = params
        self.param_types = param_types
        self.query_mode = query_mode
        self.directed_read_options = directed_read_options
        self.data_boost_enabled = data_boost_enabled


class _MutationCall:
    def __init__(self, table: str, columns: list[str], values: list[list[_Scalar]]) -> None:
        self.table = table
        self.columns = columns
        self.values = values


class _TransactionLog:
    def __init__(self) -> None:
        self.sql: list[_SqlCall] = []
        self.mutations: list[_MutationCall] = []
        self.committed = False


class _RecordingTransaction:
    def __init__(self, database: RecordingSpannerDatabase, log: _TransactionLog) -> None:
        self._database = database
        self._log = log
        self._heads = {key: dict(row) for key, row in database.heads.items()}
        self._transitions = {key: dict(row) for key, row in database.transitions.items()}

    def execute_sql(
        self,
        sql: str,
        params: Mapping[str, str] | None = None,
        param_types: Mapping[str, Mapping[str, str]] | None = None,
    ) -> list[list[_Scalar]]:
        call = _SqlCall(
            sql,
            None if params is None else dict(params),
            None if param_types is None else {name: dict(value) for name, value in param_types.items()},
            query_mode=None,
            directed_read_options=None,
            data_boost_enabled=False,
        )
        self._log.sql.append(call)
        self._database.events.append("transaction-sql")
        return _project(sql, params, self._heads, self._transitions)

    def insert_or_update(
        self,
        table: str,
        columns: list[str],
        values: list[list[_Scalar]],
    ) -> None:
        self._log.mutations.append(_MutationCall(table, list(columns), values))
        self._database.events.append("mutation")
        _store_row(table, columns, values, self._heads, self._transitions)


class _SnapshotReader:
    def __init__(self, database: RecordingSpannerDatabase) -> None:
        self._database = database

    def execute_sql(
        self,
        sql: str,
        params: Mapping[str, str] | None = None,
        param_types: Mapping[str, Mapping[str, str]] | None = None,
    ) -> list[list[_Scalar]]:
        self._database.snapshot_sql.append(
            _SqlCall(
                sql,
                None if params is None else dict(params),
                None if param_types is None else {name: dict(value) for name, value in param_types.items()},
                query_mode=None,
                directed_read_options=None,
                data_boost_enabled=False,
            )
        )
        return _project(sql, params, self._database.heads, self._database.transitions)

    def insert_or_update(self, table: str, columns: list[str], values: list[list[_Scalar]]) -> None:
        raise AssertionError(f"snapshot mutated {table}")


class RecordingSpannerDatabase:
    """In-process double for Database.snapshot and Database.run_in_transaction."""

    def __init__(self) -> None:
        self.heads: dict[tuple[str, str, str], _Row] = {}
        self.transitions: dict[tuple[str, str, str, str], _Row] = {}
        self.transactions: list[_TransactionLog] = []
        self.snapshot_sql: list[_SqlCall] = []
        self.events: list[str] = []
        self.snapshot_starts = 0
        self.fault: str | None = None
        self.snapshot_failures = 0

    def snapshot(self) -> AbstractContextManager[_SnapshotReader]:
        return self._snapshot()

    @contextmanager
    def _snapshot(self) -> Iterator[_SnapshotReader]:
        self.snapshot_starts += 1
        self.events.append("snapshot")
        if self.snapshot_failures:
            self.snapshot_failures -= 1
            raise ServiceUnavailable()
        yield _SnapshotReader(self)

    def run_in_transaction(self, func: object, *args: object, **kw: object) -> None:
        if args or kw:
            raise AssertionError("bridge passed session or retry arguments")
        if not callable(func):
            raise AssertionError("transaction function is required")
        log = _TransactionLog()
        self.transactions.append(log)
        self.events.append("transaction")
        txn = _RecordingTransaction(self, log)
        try:
            func(txn)
        except Exception:
            self.events.append("rollback")
            raise
        fault = self.fault
        self.fault = None
        if fault == "unknown_before":
            self.events.append("commit-before")
            raise DeadlineExceeded()
        if fault == "aborted":
            self.events.append("commit-aborted")
            raise Aborted()
        self.heads = txn._heads
        self.transitions = txn._transitions
        log.committed = True
        self.events.append("commit-applied")
        if fault == "unknown_after":
            raise DeadlineExceeded()


def _selected_columns(sql: str) -> tuple[str, ...]:
    selected = sql.split(" FROM ", 1)[0].removeprefix("SELECT ")
    return tuple(part.strip() for part in selected.split(","))


def _require_predicates(
    sql: str,
    params: Mapping[str, str] | None,
    names: tuple[str, ...],
) -> Mapping[str, str]:
    if params is None:
        raise AssertionError("scoped query has no parameters")
    for name in names:
        if f"@{name}" not in sql or name not in params:
            raise AssertionError(name)
    return params


def _project(
    sql: str,
    params: Mapping[str, str] | None,
    heads: dict[tuple[str, str, str], _Row],
    transitions: dict[tuple[str, str, str, str], _Row],
) -> list[list[_Scalar]]:
    columns = _selected_columns(sql)
    if "FROM continuity_head" in sql:
        bound = _require_predicates(sql, params, ("application_id", "environment_id", "tenant_id"))
        row = heads.get((bound["application_id"], bound["environment_id"], bound["tenant_id"]))
    elif "FROM continuity_transition" in sql:
        bound = _require_predicates(
            sql,
            params,
            ("application_id", "environment_id", "tenant_id", "transition_id"),
        )
        row = transitions.get(
            (
                bound["application_id"],
                bound["environment_id"],
                bound["tenant_id"],
                bound["transition_id"],
            )
        )
    else:
        raise AssertionError(sql)
    if row is None:
        return []
    return [[row[column] for column in columns]]


def _store_row(
    table: str,
    columns: list[str],
    values: list[list[_Scalar]],
    heads: dict[tuple[str, str, str], _Row],
    transitions: dict[tuple[str, str, str, str], _Row],
) -> None:
    if len(values) != 1:
        raise AssertionError("one mutation row")
    record = {column: values[0][index] for index, column in enumerate(columns)}
    app = record["application_id"]
    env = record["environment_id"]
    tenant = record["tenant_id"]
    if type(app) is not str or type(env) is not str or type(tenant) is not str:
        raise AssertionError("scope values must stay strings")
    if table == "continuity_head":
        heads[(app, env, tenant)] = record
        return
    if table == "continuity_transition":
        transition_id = record["transition_id"]
        if type(transition_id) is not str:
            raise AssertionError("transition id")
        transitions[(app, env, tenant, transition_id)] = record
        return
    raise AssertionError(table)


def _head(**overrides: object) -> HeadRecord:
    values: dict[str, object] = {
        "application_id": _APP,
        "environment_id": _ENV,
        "tenant_id": "tenant-a",
        "revision": 1,
        "fencing_token": "1",
        "status": "active",
        "pis_history_id": "pis-h",
        "pis_watermark": 1,
        "gca_history_id": "gca-h",
        "gca_watermark": 1,
        "identity_epoch": 1,
        "proof_revision": 1,
        "phase": "committed",
        "transition_id": "env-1:1:1",
        "request_digest": "digest",
        "authority_incarnation": "inc",
    }
    values.update(overrides)
    return HeadRecord(**values)  # type: ignore[arg-type]


def _transition(*, revision: int, fence: str, tenant_id: str = "tenant-a") -> TransitionRecord:
    binding = HistoryBinding(
        pis_history_id="pis-h",
        pis_committed_watermark=revision,
        gca_history_id="gca-h",
        gca_committed_watermark=revision,
    )
    digest = continuity_binding_digest(binding)
    transition_id = f"{_ENV}:{revision}:{fence}"
    return TransitionRecord(
        application_id=_APP,
        environment_id=_ENV,
        tenant_id=tenant_id,
        transition_id=transition_id,
        request_digest=request_digest(
            application_id=_APP,
            environment_id=_ENV,
            tenant_id=tenant_id,
            revision=revision,
            fencing_token=fence,
            transition_id=transition_id,
            binding_digest=digest,
            status="active",
            identity_epoch=1,
            pis_history_id=binding.pis_history_id,
            pis_watermark=binding.pis_committed_watermark,
            gca_history_id=binding.gca_history_id,
            gca_watermark=binding.gca_committed_watermark,
        ),
        revision=revision,
        fencing_token=fence,
        binding_digest=digest,
        status="active",
        identity_epoch=1,
        phase="prepared",
        pis_history_id=binding.pis_history_id,
        pis_watermark=binding.pis_committed_watermark,
        gca_history_id=binding.gca_history_id,
        gca_watermark=binding.gca_committed_watermark,
    )


def _bridge() -> tuple[RecordingSpannerDatabase, SpannerSdkPrimaryBridge]:
    database = RecordingSpannerDatabase()
    return database, SpannerSdkPrimaryBridge(database)


def _prepare(
    bridge: SpannerSdkPrimaryBridge,
    transition: TransitionRecord,
    *,
    expected_revision: int,
    expected_fencing_token: str | None,
) -> PrimaryStep:
    return SpannerPrimaryRegister().prepare(
        bridge,
        application_id=transition.application_id,
        environment_id=transition.environment_id,
        tenant_id=transition.tenant_id,
        expected_revision=expected_revision,
        expected_fencing_token=expected_fencing_token,
        transition=transition,
        authority_incarnation="offline-incarnation:1",
    )


def test_spanner_bridge_strong_read_decodes_typed_rows_and_sdk_arguments() -> None:
    database, bridge = _bridge()
    assert database.events == []
    stored = _head(revision=4, fencing_token="fence-4")

    def seed(txn: _RecordingTransaction) -> None:
        txn.insert_or_update(
            "continuity_head",
            list(CONTINUITY_HEAD_COLUMNS),
            [list(spanner_sdk.encode_head(stored))],
        )

    database.run_in_transaction(seed)
    database.events.clear()
    database.snapshot_sql.clear()
    found = bridge.read_head(_APP, _ENV, "tenant-a")
    assert found is not None
    assert found == stored
    assert type(found.revision) is int
    assert database.transactions[-1].mutations
    assert database.events[0] == "snapshot"
    assert "transaction" not in database.events
    call = database.snapshot_sql[-1]
    assert call.params == {
        "application_id": _APP,
        "environment_id": _ENV,
        "tenant_id": "tenant-a",
    }
    assert call.param_types == {name: {"code": "STRING"} for name in call.params or {}}
    assert call.sql.startswith("SELECT " + ", ".join(CONTINUITY_HEAD_COLUMNS))
    assert "tenant_id = @tenant_id" in call.sql
    assert call.query_mode is None
    assert call.directed_read_options is None
    assert call.data_boost_enabled is False
    signature = inspect.signature(RecordingSpannerDatabase.snapshot)
    assert "exact_staleness" not in signature.parameters
    assert "read_timestamp" not in signature.parameters
    sql_signature = inspect.signature(_SnapshotReader.execute_sql)
    assert list(sql_signature.parameters)[1:4] == ["sql", "params", "param_types"]


def test_spanner_bridge_commits_one_transaction_and_rolls_back_partial_work() -> None:
    database, bridge = _bridge()
    record = _head()

    def partial(txn: object) -> None:
        txn.put_head(record)  # type: ignore[attr-defined]
        assert database.heads == {}
        raise PrimaryConflict(ProviderFault.REVISION_CONFLICT)

    rejected = bridge.run(partial)
    assert rejected.kind is PrimaryCommitKind.CONFLICT
    assert bridge.read_head(_APP, _ENV, "tenant-a") is None
    assert database.transactions[0].committed is False

    def complete(txn: object) -> None:
        txn.insert_transition(_transition(revision=1, fence="1"))  # type: ignore[attr-defined]
        txn.put_head(record)  # type: ignore[attr-defined]
        assert database.heads == {}

    committed = bridge.run(complete)
    assert committed.kind is PrimaryCommitKind.COMMITTED
    assert bridge.read_head(_APP, _ENV, "tenant-a") == record
    mutation = database.transactions[-1].mutations[0]
    assert mutation.table == "continuity_transition"
    revision_index = mutation.columns.index("revision")
    assert type(mutation.values[0][revision_index]) is int
    assert database.transactions[-1].sql == []
    assert len(database.transactions) == 2


def test_spanner_register_keeps_cas_fence_idempotency_and_tenant_scope() -> None:
    database, bridge = _bridge()
    transition = _transition(revision=1, fence="1")
    first = _prepare(bridge, transition, expected_revision=0, expected_fencing_token=None)
    second = _prepare(bridge, transition, expected_revision=0, expected_fencing_token=None)
    assert first.kind is PrimaryCommitKind.COMMITTED
    assert second.kind is PrimaryCommitKind.COMMITTED
    assert len(database.transactions[0].mutations) == 2
    assert database.transactions[1].mutations == []
    assert [call.table for call in database.transactions[0].mutations] == [
        "continuity_transition",
        "continuity_head",
    ]
    assert database.transactions[0].committed is True

    seeded = _head(revision=2, fencing_token="9", phase="committed")

    def replace(txn: object) -> None:
        txn.put_head(seeded)  # type: ignore[attr-defined]

    assert bridge.run(replace).kind is PrimaryCommitKind.COMMITTED
    stale = _prepare(
        bridge,
        _transition(revision=3, fence="9"),
        expected_revision=1,
        expected_fencing_token="9",
    )
    assert stale.fault is ProviderFault.REVISION_CONFLICT
    fence = _prepare(
        bridge,
        _transition(revision=2, fence="1"),
        expected_revision=2,
        expected_fencing_token="1",
    )
    assert fence.fault is ProviderFault.FENCE_CONFLICT
    assert bridge.read_head(_APP, _ENV, "tenant-b") is None
    other = _prepare(
        bridge,
        _transition(revision=1, fence="1", tenant_id="tenant-b"),
        expected_revision=0,
        expected_fencing_token=None,
    )
    assert other.kind is PrimaryCommitKind.COMMITTED
    home = bridge.read_head(_APP, _ENV, "tenant-a")
    isolated = bridge.read_head(_APP, _ENV, "tenant-b")
    assert home is not None and home.tenant_id == "tenant-a" and home.revision == 2
    assert isolated is not None and isolated.tenant_id == "tenant-b"


def test_spanner_unknown_commit_is_success_only_after_strong_read_back() -> None:
    database, bridge = _bridge()
    record = _head()
    database.fault = "unknown_before"
    before = bridge.run(lambda txn: txn.put_head(record))
    assert before.kind is PrimaryCommitKind.UNKNOWN
    assert before.fault is ProviderFault.OUTCOME_UNKNOWN
    assert bridge.read_head(_APP, _ENV, "tenant-a") is None
    assert "commit-before" in database.events
    assert database.events.index("commit-before") < database.events.index("snapshot")

    database.fault = "unknown_after"
    after = bridge.run(lambda txn: txn.put_head(record))
    assert after.kind is PrimaryCommitKind.COMMITTED
    assert after.head == record
    assert "commit-applied" in database.events
    applied = database.events.index("commit-applied")
    assert "snapshot" in database.events[applied + 1 :]

    database.fault = "unknown_after"
    database.snapshot_failures = 1
    hidden = bridge.run(lambda txn: txn.put_head(_head(revision=5)))
    assert hidden.kind is PrimaryCommitKind.UNKNOWN
    assert bridge.read_head(_APP, _ENV, "tenant-a") == _head(revision=5)

    database.fault = "aborted"
    aborted = bridge.run(lambda txn: txn.put_head(_head(revision=8)))
    assert aborted.kind is PrimaryCommitKind.CONFLICT
    assert aborted.fault is ProviderFault.REVISION_CONFLICT
    abort_at = database.events.index("commit-aborted")
    assert "snapshot" not in database.events[abort_at:]
    assert bridge.read_head(_APP, _ENV, "tenant-a") == _head(revision=5)


class _Download:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def readall(self) -> bytes:
        return self._payload


class _Properties:
    def __init__(self, size: int) -> None:
        self._size = size

    @property
    def size(self) -> int:
        return self._size


class RecordingBlob:
    def __init__(self, directory: RecordingBlobDirectory, name: str) -> None:
        self._directory = directory
        self.name = name
        self.creates: list[dict[str, object]] = []
        self.appends: list[tuple[bytes, int | None, int]] = []
        self.downloads: list[tuple[int | None, int | None]] = []
        self.property_reads = 0

    @property
    def url(self) -> str:
        host = "account-secondary.blob.core.windows.net" if self._directory.secondary else "account.blob.core.windows.net"
        return f"https://{host}/witness/{self.name}"

    def create_append_blob(self) -> dict[str, str]:
        self.creates.append({"content_settings": None, "metadata": None, "kwargs": {}})
        self._directory.blobs.setdefault(self.name, b"")
        return {}

    def append_block(
        self,
        data: bytes,
        length: int | None = None,
        *,
        appendpos_condition: int,
    ) -> dict[str, int]:
        self.appends.append((data, length, appendpos_condition))
        fault = self._directory.take_append_fault()
        if fault == "before":
            raise DeadlineExceeded()
        if fault == "unavailable":
            raise HttpResponseError(500)
        if fault == "conflict":
            raise HttpResponseError(412)
        current = self._directory.blobs.get(self.name)
        if current is None:
            raise ResourceNotFoundError()
        if length != len(data) or appendpos_condition != len(current):
            raise HttpResponseError(412)
        self._directory.blobs[self.name] = current + data
        if fault == "after":
            raise DeadlineExceeded()
        return {"append_offset": appendpos_condition}

    def download_blob(
        self,
        offset: int | None = None,
        length: int | None = None,
    ) -> _Download:
        self.downloads.append((offset, length))
        if self._directory.download_failures:
            self._directory.download_failures -= 1
            raise ServiceUnavailable()
        current = self._directory.blobs.get(self.name)
        if current is None:
            raise ResourceNotFoundError()
        start = 0 if offset is None else offset
        end = len(current) if length is None else start + length
        return _Download(current[start:end])

    def get_blob_properties(self) -> _Properties:
        self.property_reads += 1
        if self._directory.property_fault == "missing":
            raise ResourceNotFoundError()
        if self._directory.property_fault == "unavailable":
            raise HttpResponseError(503)
        current = self._directory.blobs.get(self.name)
        if current is None:
            raise ResourceNotFoundError()
        return _Properties(len(current))


class RecordingBlobDirectory:
    def __init__(self, *, secondary: bool = False) -> None:
        self.secondary = secondary
        self.blobs: dict[str, bytes] = {}
        self.clients: list[RecordingBlob] = []
        self.opened: list[tuple[str, object]] = []
        self.append_fault: str | None = None
        self.property_fault: str | None = None
        self.download_failures = 0

    def get_blob_client(self, blob: str) -> RecordingBlob:
        self.opened.append((blob, None))
        client = RecordingBlob(self, blob)
        self.clients.append(client)
        return client

    def take_append_fault(self) -> str | None:
        fault = self.append_fault
        self.append_fault = None
        return fault


def _witness_bridge(*, secondary: bool = False) -> tuple[RecordingBlobDirectory, AzureSdkAppendBridge]:
    directory = RecordingBlobDirectory(secondary=secondary)
    bridge = AzureSdkAppendBridge(directory)
    return directory, bridge


def _seal_bytes(revision: int, predecessor: str) -> bytes:
    payload = {
        "kind": "seal",
        "application_id": _APP,
        "environment_id": _ENV,
        "tenant_id": "tenant-a",
        "revision": revision,
        "fencing_token": "1",
        "transition_id": f"{_ENV}:{revision}:1",
        "binding_digest": f"digest-{revision}",
        "status": "active",
        "identity_epoch": 1,
        "predecessor": predecessor,
        "segment": 0,
        "record_ordinal": revision - 1,
    }
    return encode_block(payload)


def test_azure_bridge_names_blobs_and_appends_exact_conditional_bytes() -> None:
    directory, bridge = _witness_bridge()
    assert directory.opened == []
    bridge.open_segment(0)
    bridge.open_segment(0)
    assert directory.opened[0][0] == witness_blob_name(0)
    created = [client for client in directory.clients if client.creates]
    assert len(created) == 1
    assert created[0].creates[0]["kwargs"] == {}
    block = _seal_bytes(1, "predecessor")
    step = bridge.append(0, 0, block)
    assert step.kind.value == "committed"
    assert directory.blobs[witness_blob_name(0)] == block
    appends = [item for client in directory.clients for item in client.appends]
    assert appends == [(block, _BLOCK, 0)]
    assert bridge.read_range(0, 0, _BLOCK) == block
    assert bridge.length(0) == _BLOCK
    signature = inspect.signature(RecordingBlob.append_block)
    assert list(signature.parameters)[1:3] == ["data", "length"]
    assert signature.parameters["appendpos_condition"].kind is inspect.Parameter.KEYWORD_ONLY
    download = inspect.signature(RecordingBlob.download_blob)
    assert list(download.parameters)[1:3] == ["offset", "length"]
    assert "appendpos_condition" not in download.parameters


def test_azure_bridge_separates_conflict_timeout_and_read_back() -> None:
    directory, bridge = _witness_bridge()
    bridge.open_segment(0)
    block = _seal_bytes(1, "predecessor")
    other = _seal_bytes(2, "other")
    assert bridge.append(0, 0, block).kind.value == "committed"
    conflict = bridge.append(0, 0, other)
    assert conflict.kind.value == "conflict"
    assert conflict.fault is ProviderFault.APPEND_POSITION_CONFLICT
    assert directory.blobs[witness_blob_name(0)] == block

    directory.append_fault = "before"
    before = bridge.append(0, _BLOCK, other)
    assert before.kind.value == "unknown"
    assert directory.blobs[witness_blob_name(0)] == block

    directory.append_fault = "after"
    after = bridge.append(0, _BLOCK, other)
    assert after.kind.value == "committed"
    assert directory.blobs[witness_blob_name(0)] == block + other

    directory.append_fault = "after"
    directory.download_failures = 1
    hidden = bridge.append(0, 2 * _BLOCK, block)
    assert hidden.kind.value == "unknown"
    assert directory.blobs[witness_blob_name(0)] == block + other + block

    missing, bare = _witness_bridge()
    assert bare.length(0) == -1
    assert bare.append(0, 0, block).fault is ProviderFault.PREFIX_MISSING
    missing.property_fault = "unavailable"
    bridge_unavailable = AzureSdkAppendBridge(missing)
    try:
        bridge_unavailable.length(0)
    except WitnessStorageUnavailable as exc:
        assert exc.fault is ProviderFault.UNAVAILABLE
    else:
        raise AssertionError("unavailable storage looked like a missing blob")


def test_azure_witness_rejects_bad_bytes_rollover_and_secondary_reads() -> None:
    directory, bridge = _witness_bridge()
    bridge.open_segment(0)
    witness = AzureAppendWitness(bridge, segment_block_limit=3, recovery_block_bound=16)
    common = {
        "application_id": _APP,
        "environment_id": _ENV,
        "tenant_id": "tenant-a",
        "fencing_token": "1",
        "status": "active",
        "identity_epoch": 1,
    }
    assert witness.append_seal(revision=1, transition_id="env-1:1:1", binding_digest="d1", **common).kind.value == "committed"
    second = witness.append_seal(revision=2, transition_id="env-1:2:1", binding_digest="d2", **common)
    assert second.kind.value == "committed"
    name = witness_blob_name(0)
    raw = bytearray(directory.blobs[name])
    raw[_BLOCK] = 0
    directory.blobs[name] = bytes(raw)
    assert witness.verify_bounded(full_chain=False).fault is ProviderFault.CORRUPT

    directory.blobs[name] = directory.blobs[name][_BLOCK:]
    assert witness.verify_bounded(full_chain=True).fault is not None

    fresh_dir, fresh = _witness_bridge()
    fresh.open_segment(0)
    chain = AzureAppendWitness(fresh, segment_block_limit=3, recovery_block_bound=16)
    for revision, digest in ((1, "d1"), (2, "d2"), (3, "d3")):
        assert (
            chain.append_seal(
                revision=revision,
                transition_id=f"{_ENV}:{revision}:1",
                binding_digest=digest,
                **common,
            ).kind.value
            == "committed"
        )
    assert witness_blob_name(0) in fresh_dir.blobs
    assert witness_blob_name(1) in fresh_dir.blobs
    genesis = decode_block(fresh.read_range(1, 0, _BLOCK) or b"")
    assert genesis is not None and genesis["kind"] == "segment_genesis"

    short_dir, short = _witness_bridge()
    short.open_segment(0)
    short_dir.blobs[witness_blob_name(0)] = b"truncated"
    assert short.length(0) == len(b"truncated")
    assert AzureAppendWitness(short).verify_bounded(full_chain=True).fault is ProviderFault.CORRUPT

    absent, absent_bridge = _witness_bridge()
    del absent
    assert AzureAppendWitness(absent_bridge).verify_bounded(full_chain=False).fault is ProviderFault.PREFIX_MISSING

    secondary_dir, secondary = _witness_bridge(secondary=True)
    block = _seal_bytes(1, block_digest({"kind": "unused"}))
    secondary_dir.blobs[witness_blob_name(0)] = block
    view = AzureAppendWitness(secondary).verify_bounded(full_chain=False)
    assert view.fault is ProviderFault.PREFIX_MISSING
    assert view.high_revision == 0
    denied = AzureAppendWitness(secondary).append_seal(
        revision=1,
        transition_id="env-1:1:1",
        binding_digest="d1",
        **common,
    )
    assert denied.kind.value != "committed"
    assert secondary_dir.blobs[witness_blob_name(0)] == block


def test_sdk_bridges_stay_offline_and_outside_composition() -> None:
    assert spanner_sdk.cloud_execution_authorized() is False
    assert azure_sdk.cloud_execution_authorized() is False
    primary = SpannerPrimaryTarget(
        project_id="offline-project",
        instance_id="offline-instance",
        database_id="offline-database",
        edition="enterprise_plus",
        instance_configuration="eur3",
    )
    witness = AzureWitnessTarget(
        region="northeurope",
        account_sku="Standard_RAGZRS",
        account_kind="StorageV2",
        access_tier="Hot",
    )
    with pytest.raises(OfflineProviderRefusal):
        open_spanner_sdk_database(primary, None)
    with pytest.raises(OfflineProviderRefusal):
        azure_sdk.open_azure_append_client(witness, None)
    database, bridge = _bridge()
    directory, append_bridge = _witness_bridge()
    assert database.events == []
    assert directory.opened == []
    assert "google.cloud" not in sys.modules
    assert "azure.storage" not in sys.modules
    for module in (spanner_sdk, azure_sdk):
        source = inspect.getsource(module)
        assert "google.cloud" not in source
        assert "azure.storage" not in source
        assert "importlib" not in source
        assert "getenv" not in source
        assert "os.environ" not in source
        assert "getattr" not in source
    contract = inspect.getsource(continuity_contract)
    assert "SpannerSdkPrimaryBridge" not in contract
    assert "AzureSdkAppendBridge" not in contract
    assert "google.cloud" not in contract
    assert "azure.storage" not in contract
    rules = [
        PolicyBundleRule(
            rule_id=f"history_continuity.{permission.value}",
            description=f"Allow {permission.value}",
            match_action=continuity_authorization_action(permission),
            effect="allow",
        )
        for permission in sorted(ContinuityPermission, key=lambda item: item.value)
    ]
    bundle = build_immutable_runtime_policy_bundle(
        bundle_id="d03a.sdk.bridge",
        version="1.0.0",
        rules=rules,
        issued_at=datetime(2026, 10, 19, tzinfo=timezone.utc),
    )
    governance = ProductionControlPlaneGovernance(
        principal=RequestIdentity(
            tenant_id="tenant-a",
            user_id="continuity-owner",
            principal_type=PrincipalType.SERVICE,
            auth_subject="continuity-owner",
        ),
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=bundle_backed_control_plane_mutation_evaluator(
                RuntimePolicyBundleEvaluator(bundle),
            ),
        ),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(tenant_id="tenant-a"),
    )
    selection = HistoryContinuityProviderSelection(
        adapter=HistoryContinuityAdapterKind.OFFLINE_EXTERNAL_PAIR,
        execution_domain=HistoryContinuityExecutionDomain.OFFLINE_CONTRACT,
        cloud_qualification=HistoryContinuityCloudQualification.UNQUALIFIED,
        application_id=_APP,
        environment_id=_ENV,
        retention=HistoryContinuityRetentionPolicy(
            retention_seconds=3600,
            ceiling_seconds=7200,
            legal_hold=True,
        ),
        administrators=HistoryContinuityAdministrativeIdentities(
            primary_administrator_id="primary-admin",
            witness_administrator_id="witness-admin",
        ),
        identity_binding=HistoryContinuityIdentityBinding(
            binding_id="offline-binding-1",
            credential_mode=HistoryContinuityRuntimeCredentialMode.WORKLOAD_FEDERATION,
            subject="offline-subject",
            audience="offline-audience",
        ),
    )
    denied_primary = compose_history_continuity_authority(
        governance,
        None,
        provider_selection=selection,
        offline_bundle=OfflineProviderBundle(
            primary=cast(InMemoryPrimaryDatabase, bridge),
            witness=InMemoryAppendWitness(),
            primary_target=primary,
            witness_target=witness,
        ),
    )
    denied_witness = compose_history_continuity_authority(
        governance,
        None,
        provider_selection=selection,
        offline_bundle=OfflineProviderBundle(
            primary=InMemoryPrimaryDatabase(),
            witness=cast(InMemoryAppendWitness, append_bridge),
            primary_target=primary,
            witness_target=witness,
        ),
    )
    assert isinstance(denied_primary, InactiveHistoryContinuityAuthority)
    assert isinstance(denied_witness, InactiveHistoryContinuityAuthority)
    assert "GovernedProviderInvocationDispatchGate" not in inspect.getsource(spanner_sdk)
    assert "GovernedProviderInvocationDispatchGate" not in inspect.getsource(azure_sdk)


def test_azure_predecessor_mismatch_stays_a_witness_decision() -> None:
    directory, bridge = _witness_bridge()
    bridge.open_segment(0)
    witness = AzureAppendWitness(bridge, segment_block_limit=8, recovery_block_bound=16)
    common = {
        "application_id": _APP,
        "environment_id": _ENV,
        "tenant_id": "tenant-a",
        "fencing_token": "1",
        "status": "active",
        "identity_epoch": 1,
    }
    assert witness.append_seal(revision=1, transition_id="env-1:1:1", binding_digest="d1", **common).kind.value == "committed"
    assert witness.append_seal(revision=2, transition_id="env-1:2:1", binding_digest="d2", **common).kind.value == "committed"
    name = witness_blob_name(0)
    first = decode_block(directory.blobs[name][:_BLOCK])
    assert first is not None
    replacement = encode_block(
        {
            "kind": "seal",
            "application_id": _APP,
            "environment_id": _ENV,
            "tenant_id": "tenant-a",
            "revision": 2,
            "fencing_token": "1",
            "transition_id": "env-1:2:1",
            "binding_digest": "d2",
            "status": "active",
            "identity_epoch": 1,
            "predecessor": "0" * 64,
            "segment": 0,
            "record_ordinal": 1,
        }
    )
    directory.blobs[name] = directory.blobs[name][:_BLOCK] + replacement
    assert witness.verify_bounded(full_chain=False).fault is ProviderFault.CORRUPT
