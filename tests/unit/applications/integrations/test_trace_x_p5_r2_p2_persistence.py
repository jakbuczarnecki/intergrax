# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P2 durable opportunity and provenance persistence tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk
from intergrax.contracts.execution_identity import (
    ExecutionId,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    validate_execution_id,
)
from intergrax.integrations.contracts.execution_integration_configuration_pin_record import (
    ExecutionIntegrationConfigurationRequirementRecoveryStaging,
)
from intergrax.contracts.execution_integration_configuration_provenance import (
    ConfiguredIntegrationProvenanceSlice,
    ExecutionIntegrationConfigurationProvenance,
    ExecutionIntegrationConfigurationProvenanceMode,
    IntegrationConfigurationSubject,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.execution_integration_configuration import (
    EffectiveIntegrationIdentity,
    IntegrationMaterializationKind,
)
from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
    ExecutionIntegrationConfigurationPinningError,
    ExecutionIntegrationConfigurationPinningFailureReason,
)
from intergrax.integrations.contracts.existing_capability_configuration_opportunity import (
    ConfigurationOpportunityRef,
    ExistingCapabilityConfigurationOpportunity,
    ExistingCapabilityConfigurationOpportunityLookupError,
    ExistingCapabilityConfigurationOpportunityLookupFailureReason,
    validate_configuration_opportunity_ref,
)
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentQueryPageV1,
    DocumentRecord,
)
from intergrax.integrations.providers.relational_store.sqlite.configuration_payload_codec import (
    sqlite_relational_store_configuration_payload_codec,
)
from intergrax.integrations.providers.relational_store.sqlite.configuration_realization import (
    SQLiteRelationalStoreConfigurationPayload,
)
from intergrax.integrations.contracts.integration_configuration_payload_codec import (
    integration_configuration_payload_codec_registry,
)
from intergrax.applications._shared.integrations.persistence import (
    DocumentStoreExistingCapabilityConfigurationOpportunityStore,
    DocumentStoreExecutionIntegrationConfigurationPinningStore,
    InMemoryExistingCapabilityConfigurationOpportunityStore,
    InMemoryExecutionIntegrationConfigurationPinningStore,
    KvExistingCapabilityConfigurationOpportunityStore,
    KvExecutionIntegrationConfigurationPinningStore,
    _provenance_index_kv_key,
    _provenance_kv_key,
    _subject_row_key,
    decode_configuration_opportunity,
    encode_configuration_opportunity,
    encode_integration_configuration_provenance,
    wire_existing_capability_configuration_opportunity_store,
)
from intergrax.applications._shared.integrations.integration_configuration_provenance_reader import (
    PinningStoreExecutionIntegrationConfigurationProvenanceReader,
)
from testing_support.integration_configuration_payload_codecs import (
    TEST_CONFIGURATION_PAYLOAD_TYPE,
    QualificationIntegrationConfigurationPayload,
    qualification_integration_configuration_payload_codec_registry,
)

pytestmark = pytest.mark.unit

_EXEC = validate_execution_id("exec_01234567890123456789012345678901")


class SimulatedProcessCrash(RuntimeError):
    """Deterministic crash injection for KV provenance pinning tests."""


class InMemoryKVStore(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}
        self.abort_before_provenance_record_cas = False

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        del ttl_seconds
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        del ttl_seconds
        if (
            self.abort_before_provenance_record_cas
            and key.startswith("integration_config_provenance:")
            and expected is None
        ):
            raise SimulatedProcessCrash("simulated crash before provenance record CAS")
        current = self._data.get((tenant_id, key))
        if current != expected:
            return False
        self._data[(tenant_id, key)] = new_value
        return True


class InMemoryConditionalDocumentStore(ConditionalDocumentStore):
    def __init__(self, *, max_page_size: int | None = None) -> None:
        self._records: dict[tuple[str, str], DocumentRecord] = {}
        self._max_page_size = max_page_size
        self.query_call_count = 0

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        return self._records.get((partition_key, row_key))

    def put(self, document: DocumentRecord) -> None:
        self._records[(document.partition_key, document.row_key)] = document

    def delete(self, partition_key: str, row_key: str) -> None:
        self._records.pop((partition_key, row_key), None)

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
        cursor: str | None = None,
        row_key_upper_bound: str | None = None,
        data_equalities=(),
        sort=(),
    ) -> DocumentQueryPageV1:
        del row_key_upper_bound, data_equalities, sort
        self.query_call_count += 1
        docs = [
            record
            for (partition, _), record in self._records.items()
            if partition == partition_key
            and (row_key_prefix is None or record.row_key.startswith(row_key_prefix))
        ]
        docs.sort(key=lambda d: d.row_key)
        start = 0
        if cursor is not None:
            if not cursor.startswith("offset:"):
                raise ValueError(f"unsupported test cursor: {cursor}")
            start = int(cursor.split(":", 1)[1])
        page_size = limit
        if self._max_page_size is not None:
            page_size = min(page_size, self._max_page_size)
        page_docs = docs[start : start + page_size]
        next_offset = start + len(page_docs)
        next_cursor = f"offset:{next_offset}" if next_offset < len(docs) else None
        return DocumentQueryPageV1(documents=tuple(page_docs), next_cursor=next_cursor)

    def close(self) -> None:
        return None

    def put_if_absent(self, document: DocumentRecord) -> bool:
        key = (document.partition_key, document.row_key)
        if key in self._records:
            return False
        self._records[key] = document
        return True

    def replace_if_match(self, *, expected: DocumentRecord, replacement: DocumentRecord) -> bool:
        del expected, replacement
        return False


class ScriptedQueryDocumentStore(InMemoryConditionalDocumentStore):
    """Test double that can return scripted query pages before falling back to storage."""

    def __init__(
        self,
        *,
        scripted_pages: tuple[DocumentQueryPageV1, ...] = (),
        share_records_from: InMemoryConditionalDocumentStore | None = None,
    ) -> None:
        super().__init__()
        if share_records_from is not None:
            self._records = share_records_from._records
        self._scripted_pages = list(scripted_pages)
        self._scripted_page_index = 0

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: str | None = None,
        cursor: str | None = None,
        row_key_upper_bound: str | None = None,
        data_equalities=(),
        sort=(),
    ) -> DocumentQueryPageV1:
        if self._scripted_page_index < len(self._scripted_pages):
            page = self._scripted_pages[self._scripted_page_index]
            self._scripted_page_index += 1
            del partition_key, limit, row_key_prefix, cursor, row_key_upper_bound, data_equalities, sort
            self.query_call_count += 1
            return page
        return super().query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
            row_key_upper_bound=row_key_upper_bound,
            data_equalities=data_equalities,
            sort=sort,
        )


def _pin_distinct_provenance_records(
    store: DocumentStoreExecutionIntegrationConfigurationPinningStore | KvExecutionIntegrationConfigurationPinningStore,
    count: int,
    *,
    tenant_id: str = "tenant-a",
) -> list[ExecutionIntegrationConfigurationProvenance]:
    base = _provenance_configured_adopted()
    if tenant_id != "tenant-a":
        base = replace(base, tenant_id=tenant_id, configured=_configured_slice(tenant_id=tenant_id))
    records: list[ExecutionIntegrationConfigurationProvenance] = []
    for index in range(count):
        provider_id = f"prov-{index:04d}"
        subject = _subject(provider_id=provider_id)
        record = replace(
            base,
            configured=_configured_slice(provider_id=provider_id, tenant_id=tenant_id),
            effective=EffectiveIntegrationIdentity(
                integration_category=IntegrationCategory.RELATIONAL_STORE,
                provider_id=provider_id,
                materialization_kind=IntegrationMaterializationKind.CATALOG_FACTORY,
            ),
        )
        _pin_configured(store, subject=subject, provenance=record)
        records.append(record)
    def _configured_sort_key(
        item: ExecutionIntegrationConfigurationProvenance,
    ) -> tuple[str, str, str, str]:
        configured = item.configured
        assert configured is not None
        return (
            configured.integration_category.value,
            configured.provider_id,
            configured.resource_scope,
            configured.configuration_type,
        )

    records.sort(key=_configured_sort_key)
    return records


def _test_codecs():
    return qualification_integration_configuration_payload_codec_registry()


def _sqlite_codecs():
    return integration_configuration_payload_codec_registry(
        codecs=(sqlite_relational_store_configuration_payload_codec(),),
    )


def _opportunity(
    *,
    tenant_id: str = "tenant-a",
    configuration_ref: str = "opp-ref-001",
    fingerprint: str = "fp-test-001",
) -> ExistingCapabilityConfigurationOpportunity:
    return ExistingCapabilityConfigurationOpportunity(
        configuration_ref=validate_configuration_opportunity_ref(configuration_ref),
        tenant_id=tenant_id,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="scope-a",
        current_revision="rev-1",
        configuration=QualificationIntegrationConfigurationPayload(
            _configuration_type=TEST_CONFIGURATION_PAYLOAD_TYPE,
            _configuration_version="1",
            _configuration_fingerprint=fingerprint,
        ),
        configuration_fingerprint=fingerprint,
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


def _subject(**overrides: str) -> IntegrationConfigurationSubject:
    return IntegrationConfigurationSubject(
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id=overrides.get("provider_id", "sqlite"),
        resource_scope=overrides.get("resource_scope", "scope-a"),
        configuration_type=overrides.get("configuration_type", TEST_CONFIGURATION_PAYLOAD_TYPE),
    )


def _configured_slice(**overrides: str) -> ConfiguredIntegrationProvenanceSlice:
    return ConfiguredIntegrationProvenanceSlice(
        tenant_id=overrides.get("tenant_id", "tenant-a"),
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id=overrides.get("provider_id", "sqlite"),
        resource_scope=overrides.get("resource_scope", "scope-a"),
        configuration_type=overrides.get("configuration_type", TEST_CONFIGURATION_PAYLOAD_TYPE),
        configuration_version="1",
        configuration_fingerprint=overrides.get("configuration_fingerprint", "fp-test-001"),
    )


def _recovery_staging() -> ExecutionIntegrationConfigurationRequirementRecoveryStaging:
    return ExecutionIntegrationConfigurationRequirementRecoveryStaging(
        requirement_boundary_prepared_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )


def _pin_configured(
    store: object,
    *,
    subject: IntegrationConfigurationSubject,
    provenance: ExecutionIntegrationConfigurationProvenance,
) -> None:
    pin = getattr(store, "pin")
    pin(
        subject=subject,
        provenance=provenance,
        requirement_recovery_staging=_recovery_staging(),
    )


def _provenance_configured_adopted() -> ExecutionIntegrationConfigurationProvenance:
    return ExecutionIntegrationConfigurationProvenance(
        tenant_id="tenant-a",
        execution_id=_EXEC,
        mode=ExecutionIntegrationConfigurationProvenanceMode.CONFIGURED_ADOPTED,
        effective=EffectiveIntegrationIdentity(
            integration_category=IntegrationCategory.RELATIONAL_STORE,
            provider_id="sqlite",
            materialization_kind=IntegrationMaterializationKind.CATALOG_FACTORY,
        ),
        configured=_configured_slice(),
    )


@pytest.mark.parametrize("use_kv", [True, False])
def test_opportunity_first_persist_and_restart_read(use_kv: bool) -> None:
    backing_kv = InMemoryKVStore()
    backing_doc = InMemoryConditionalDocumentStore()
    if use_kv:
        store_a = KvExistingCapabilityConfigurationOpportunityStore(
            backing_kv,
            payload_codecs=_test_codecs(),
        )
        store_b = KvExistingCapabilityConfigurationOpportunityStore(
            backing_kv,
            payload_codecs=_test_codecs(),
        )
    else:
        store_a = DocumentStoreExistingCapabilityConfigurationOpportunityStore(
            backing_doc,
            payload_codecs=_test_codecs(),
        )
        store_b = DocumentStoreExistingCapabilityConfigurationOpportunityStore(
            backing_doc,
            payload_codecs=_test_codecs(),
        )
    opportunity = _opportunity()
    store_a.persist(opportunity)
    store_a.persist(opportunity)
    read = store_b.read_exact(
        tenant_id="tenant-a",
        configuration_ref=opportunity.configuration_ref,
    )
    assert read == opportunity


def test_opportunity_conflict_and_cross_tenant() -> None:
    store = KvExistingCapabilityConfigurationOpportunityStore(
        InMemoryKVStore(),
        payload_codecs=_test_codecs(),
    )
    first = _opportunity()
    second = _opportunity(fingerprint="fp-other-002")
    store.persist(first)
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError) as exc:
        store.persist(second)
    assert exc.value.reason == ExistingCapabilityConfigurationOpportunityLookupFailureReason.CONFLICT
    store.persist(_opportunity(tenant_id="tenant-b", configuration_ref="opp-ref-001"))
    assert (
        store.read_exact(
            tenant_id="tenant-a",
            configuration_ref=validate_configuration_opportunity_ref("opp-ref-001"),
        ).tenant_id
        == "tenant-a"
    )
    assert (
        store.read_exact(
            tenant_id="tenant-b",
            configuration_ref=validate_configuration_opportunity_ref("opp-ref-001"),
        ).tenant_id
        == "tenant-b"
    )
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError) as missing:
        store.read_exact(
            tenant_id="tenant-b",
            configuration_ref=validate_configuration_opportunity_ref("missing-ref"),
        )
    assert missing.value.reason == ExistingCapabilityConfigurationOpportunityLookupFailureReason.NOT_FOUND


def test_opportunity_unsupported_schema_and_corrupt_record() -> None:
    codecs = _test_codecs()
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError) as schema_exc:
        decode_configuration_opportunity(b"{\"schema_version\":999,\"record\":{}}", payload_codecs=codecs)
    assert (
        schema_exc.value.reason
        == ExistingCapabilityConfigurationOpportunityLookupFailureReason.UNSUPPORTED_SCHEMA_VERSION
    )
    with pytest.raises(ExistingCapabilityConfigurationOpportunityLookupError) as corrupt_exc:
        decode_configuration_opportunity(b"not-json", payload_codecs=codecs)
    assert corrupt_exc.value.reason == ExistingCapabilityConfigurationOpportunityLookupFailureReason.CORRUPT_RECORD


def test_sqlite_payload_round_trip_via_kv_store() -> None:
    payload = SQLiteRelationalStoreConfigurationPayload(
        data_dir=Path("/tmp/data"),
        relational_db=Path("/tmp/data/app.db"),
    )
    fp = payload.configuration_fingerprint
    opportunity = ExistingCapabilityConfigurationOpportunity(
        configuration_ref=validate_configuration_opportunity_ref("sqlite-opp-1"),
        tenant_id="tenant-a",
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="scope",
        current_revision="rev",
        configuration=payload,
        configuration_fingerprint=fp,
        risk_classification=ControlPlaneMutationRisk.MEDIUM,
    )
    store = KvExistingCapabilityConfigurationOpportunityStore(
        InMemoryKVStore(),
        payload_codecs=_sqlite_codecs(),
    )
    store.persist(opportunity)
    assert store.read_exact(
        tenant_id="tenant-a",
        configuration_ref=opportunity.configuration_ref,
    ).configuration.configuration_fingerprint == fp


@pytest.mark.parametrize("use_kv", [True, False])
def test_provenance_pin_restart_and_idempotent(use_kv: bool) -> None:
    backing_kv = InMemoryKVStore()
    backing_doc = InMemoryConditionalDocumentStore()
    if use_kv:
        store_a = KvExecutionIntegrationConfigurationPinningStore(backing_kv)
        store_b = KvExecutionIntegrationConfigurationPinningStore(backing_kv)
    else:
        store_a = DocumentStoreExecutionIntegrationConfigurationPinningStore(backing_doc)
        store_b = DocumentStoreExecutionIntegrationConfigurationPinningStore(backing_doc)
    subject = _subject()
    record = _provenance_configured_adopted()
    staging = _recovery_staging()
    store_a.pin(subject=subject, provenance=record, requirement_recovery_staging=staging)
    store_a.pin(subject=subject, provenance=record, requirement_recovery_staging=staging)
    read = store_b.read_all(tenant_id="tenant-a", execution_id=_EXEC)
    assert read == (record,)


def test_provenance_multiple_subjects_deterministic_order() -> None:
    store = KvExecutionIntegrationConfigurationPinningStore(InMemoryKVStore())
    base = _provenance_configured_adopted()
    subject_b = _subject(provider_id="postgres", configuration_type="other.type")
    record_b = replace(
        base,
        configured=_configured_slice(provider_id="postgres", configuration_type="other.type"),
        effective=EffectiveIntegrationIdentity(
            integration_category=IntegrationCategory.RELATIONAL_STORE,
            provider_id="postgres",
            materialization_kind=IntegrationMaterializationKind.CATALOG_FACTORY,
        ),
    )
    _pin_configured(store, subject=_subject(), provenance=base)
    _pin_configured(store, subject=subject_b, provenance=record_b)
    records = store.read_all(tenant_id="tenant-a", execution_id=_EXEC)
    assert records == (record_b, base)


def test_provenance_conflict_and_invalid_execution_id() -> None:
    store = InMemoryExecutionIntegrationConfigurationPinningStore()
    subject = _subject()
    record = _provenance_configured_adopted()
    _pin_configured(store, subject=subject, provenance=record)
    conflicting = replace(
        record,
        configured=_configured_slice(configuration_fingerprint="fp-changed"),
    )
    with pytest.raises(ExecutionIntegrationConfigurationPinningError) as exc:
        _pin_configured(store, subject=subject, provenance=conflicting)
    assert exc.value.reason == ExecutionIntegrationConfigurationPinningFailureReason.CONFLICT
    with pytest.raises(ValueError):
        store.read_all(tenant_id="tenant-a", execution_id=ExecutionId("bad"))


def test_provenance_effective_only_survives_codec() -> None:
    store = KvExecutionIntegrationConfigurationPinningStore(InMemoryKVStore())
    subject = _subject()
    record = ExecutionIntegrationConfigurationProvenance(
        tenant_id="tenant-a",
        execution_id=_EXEC,
        mode=ExecutionIntegrationConfigurationProvenanceMode.EFFECTIVE_ONLY,
        effective=EffectiveIntegrationIdentity(
            integration_category=IntegrationCategory.RELATIONAL_STORE,
            provider_id="sqlite",
            materialization_kind=IntegrationMaterializationKind.PROFILE_PREBUILT,
        ),
        configured=None,
    )
    _pin_configured(store, subject=subject, provenance=record)
    assert store.read_all(tenant_id="tenant-a", execution_id=_EXEC) == (record,)


def test_neutral_reader_is_read_only_projection() -> None:
    pinning = InMemoryExecutionIntegrationConfigurationPinningStore()
    subject = _subject()
    record = _provenance_configured_adopted()
    _pin_configured(pinning, subject=subject, provenance=record)
    reader = PinningStoreExecutionIntegrationConfigurationProvenanceReader(pinning)
    assert reader.read_all(tenant_id="tenant-a", execution_id=_EXEC) == (record,)
    assert not hasattr(reader, "pin")


def test_in_memory_stores_are_non_durable() -> None:
    assert InMemoryExistingCapabilityConfigurationOpportunityStore(
        payload_codecs=_test_codecs(),
    ).is_durable is False
    assert InMemoryExecutionIntegrationConfigurationPinningStore().is_durable is False


def test_kv_provenance_crash_after_index_before_record_fails_closed_then_repair() -> None:
    backing = InMemoryKVStore()
    backing.abort_before_provenance_record_cas = True
    store_a = KvExecutionIntegrationConfigurationPinningStore(backing)
    subject = _subject()
    record = _provenance_configured_adopted()
    with pytest.raises(SimulatedProcessCrash):
        _pin_configured(store_a, subject=subject, provenance=record)
    store_b = KvExecutionIntegrationConfigurationPinningStore(backing)
    with pytest.raises(ExecutionIntegrationConfigurationPinningError) as incomplete:
        store_b.read_all(tenant_id="tenant-a", execution_id=_EXEC)
    assert incomplete.value.reason == ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD
    backing.abort_before_provenance_record_cas = False
    store_c = KvExecutionIntegrationConfigurationPinningStore(backing)
    _pin_configured(store_c, subject=subject, provenance=record)
    store_d = KvExecutionIntegrationConfigurationPinningStore(backing)
    assert store_d.read_all(tenant_id="tenant-a", execution_id=_EXEC) == (record,)


def test_kv_provenance_legacy_orphan_record_without_index_repaired_on_retry() -> None:
    backing = InMemoryKVStore()
    subject = _subject()
    record = _provenance_configured_adopted()
    encoded = encode_integration_configuration_provenance(record, subject=subject)
    backing.set(
        tenant_id="tenant-a",
        key=_provenance_kv_key(_EXEC, subject),
        value=encoded,
    )
    assert backing.get(tenant_id="tenant-a", key=_provenance_index_kv_key(_EXEC)) is None
    store = KvExecutionIntegrationConfigurationPinningStore(backing)
    assert store.read_all(tenant_id="tenant-a", execution_id=_EXEC) == ()
    store.pin(subject=subject, provenance=record, requirement_recovery_staging=None)
    restarted = KvExecutionIntegrationConfigurationPinningStore(backing)
    assert restarted.read_all(tenant_id="tenant-a", execution_id=_EXEC) == (record,)


def test_document_provenance_read_all_paginates_all_records() -> None:
    backing = InMemoryConditionalDocumentStore(max_page_size=2)
    store = DocumentStoreExecutionIntegrationConfigurationPinningStore(backing)
    expected = _pin_distinct_provenance_records(store, 5)
    read = store.read_all(tenant_id="tenant-a", execution_id=_EXEC)
    assert read == tuple(expected)
    assert backing.query_call_count >= 3


def test_document_provenance_provider_caps_page_below_requested_limit() -> None:
    backing = InMemoryConditionalDocumentStore(max_page_size=2)
    store = DocumentStoreExecutionIntegrationConfigurationPinningStore(backing)
    expected = _pin_distinct_provenance_records(store, 5)
    read = store.read_all(tenant_id="tenant-a", execution_id=_EXEC)
    assert read == tuple(expected)
    assert backing.query_call_count >= 3


def test_document_provenance_read_all_matches_kv_with_pagination() -> None:
    kv_backing = InMemoryKVStore()
    doc_backing = InMemoryConditionalDocumentStore(max_page_size=2)
    kv_store = KvExecutionIntegrationConfigurationPinningStore(kv_backing)
    doc_store = DocumentStoreExecutionIntegrationConfigurationPinningStore(doc_backing)
    expected = _pin_distinct_provenance_records(kv_store, 5)
    _pin_distinct_provenance_records(doc_store, 5)
    assert doc_store.read_all(tenant_id="tenant-a", execution_id=_EXEC) == kv_store.read_all(
        tenant_id="tenant-a",
        execution_id=_EXEC,
    )
    assert expected  # records were pinned


def test_document_provenance_pagination_cross_tenant_isolation() -> None:
    backing = InMemoryConditionalDocumentStore(max_page_size=2)
    store = DocumentStoreExecutionIntegrationConfigurationPinningStore(backing)
    tenant_a = _pin_distinct_provenance_records(store, 3, tenant_id="tenant-a")
    tenant_b = _pin_distinct_provenance_records(store, 3, tenant_id="tenant-b")
    assert store.read_all(tenant_id="tenant-a", execution_id=_EXEC) == tuple(tenant_a)
    assert store.read_all(tenant_id="tenant-b", execution_id=_EXEC) == tuple(tenant_b)


def test_document_provenance_empty_partition() -> None:
    store = DocumentStoreExecutionIntegrationConfigurationPinningStore(
        InMemoryConditionalDocumentStore(),
    )
    assert store.read_all(tenant_id="tenant-a", execution_id=_EXEC) == ()


def test_document_provenance_empty_intermediate_page_continues() -> None:
    backing = InMemoryConditionalDocumentStore(max_page_size=2)
    store = DocumentStoreExecutionIntegrationConfigurationPinningStore(backing)
    expected = _pin_distinct_provenance_records(store, 2)
    scripted = ScriptedQueryDocumentStore(
        scripted_pages=(
            DocumentQueryPageV1(documents=(), next_cursor="offset:0"),
        ),
        share_records_from=backing,
    )
    reader = DocumentStoreExecutionIntegrationConfigurationPinningStore(scripted)
    assert reader.read_all(tenant_id="tenant-a", execution_id=_EXEC) == tuple(expected)


def test_document_provenance_repeated_cursor_fail_closed() -> None:
    store = DocumentStoreExecutionIntegrationConfigurationPinningStore(
        ScriptedQueryDocumentStore(
            scripted_pages=(
                DocumentQueryPageV1(documents=(), next_cursor="cursor-a"),
                DocumentQueryPageV1(documents=(), next_cursor="cursor-a"),
            ),
        ),
    )
    with pytest.raises(ExecutionIntegrationConfigurationPinningError) as exc:
        store.read_all(tenant_id="tenant-a", execution_id=_EXEC)
    assert exc.value.reason == ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD


def test_document_provenance_cursor_cycle_fail_closed() -> None:
    store = DocumentStoreExecutionIntegrationConfigurationPinningStore(
        ScriptedQueryDocumentStore(
            scripted_pages=(
                DocumentQueryPageV1(documents=(), next_cursor="cursor-a"),
                DocumentQueryPageV1(documents=(), next_cursor="cursor-b"),
                DocumentQueryPageV1(documents=(), next_cursor="cursor-a"),
            ),
        ),
    )
    with pytest.raises(ExecutionIntegrationConfigurationPinningError) as exc:
        store.read_all(tenant_id="tenant-a", execution_id=_EXEC)
    assert exc.value.reason == ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD


def test_document_provenance_duplicate_row_key_across_pages_fail_closed() -> None:
    subject = _subject()
    record = _provenance_configured_adopted()
    row_key = _subject_row_key(subject)
    partition = f"intergrax.integration_config_provenance_pinning.v1:tenant-a:{_EXEC}"
    payload = encode_integration_configuration_provenance(record, subject=subject).decode("utf-8")
    document = DocumentRecord(partition_key=partition, row_key=row_key, data={"provenance": payload})
    store = DocumentStoreExecutionIntegrationConfigurationPinningStore(
        ScriptedQueryDocumentStore(
            scripted_pages=(
                DocumentQueryPageV1(documents=(document,), next_cursor="more"),
                DocumentQueryPageV1(documents=(document,), next_cursor=None),
            ),
        ),
    )
    with pytest.raises(ExecutionIntegrationConfigurationPinningError) as exc:
        store.read_all(tenant_id="tenant-a", execution_id=_EXEC)
    assert exc.value.reason == ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD


def test_document_provenance_row_key_subject_mismatch_fail_closed() -> None:
    subject = _subject()
    record = _provenance_configured_adopted()
    partition = f"intergrax.integration_config_provenance_pinning.v1:tenant-a:{_EXEC}"
    payload = encode_integration_configuration_provenance(record, subject=subject).decode("utf-8")
    document = DocumentRecord(
        partition_key=partition,
        row_key="mismatched-row-key",
        data={"provenance": payload},
    )
    store = DocumentStoreExecutionIntegrationConfigurationPinningStore(
        ScriptedQueryDocumentStore(scripted_pages=(DocumentQueryPageV1(documents=(document,), next_cursor=None),)),
    )
    with pytest.raises(ExecutionIntegrationConfigurationPinningError) as exc:
        store.read_all(tenant_id="tenant-a", execution_id=_EXEC)
    assert exc.value.reason == ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD


def test_document_provenance_corrupt_record_on_later_page_fail_closed() -> None:
    subject = _subject()
    record = _provenance_configured_adopted()
    partition = f"intergrax.integration_config_provenance_pinning.v1:tenant-a:{_EXEC}"
    valid = DocumentRecord(
        partition_key=partition,
        row_key=_subject_row_key(subject),
        data={
            "provenance": encode_integration_configuration_provenance(record, subject=subject).decode("utf-8"),
        },
    )
    corrupt = DocumentRecord(
        partition_key=partition,
        row_key=_subject_row_key(_subject(provider_id="other")),
        data={"provenance": "not-valid-json"},
    )
    store = DocumentStoreExecutionIntegrationConfigurationPinningStore(
        ScriptedQueryDocumentStore(
            scripted_pages=(
                DocumentQueryPageV1(documents=(valid,), next_cursor="more"),
                DocumentQueryPageV1(documents=(corrupt,), next_cursor=None),
            ),
        ),
    )
    with pytest.raises(ExecutionIntegrationConfigurationPinningError):
        store.read_all(tenant_id="tenant-a", execution_id=_EXEC)


def test_wire_opportunity_store_requires_explicit_payload_codecs() -> None:
    with pytest.raises(TypeError):
        wire_existing_capability_configuration_opportunity_store(kv_store=InMemoryKVStore())
    wired = wire_existing_capability_configuration_opportunity_store(
        payload_codecs=_test_codecs(),
        kv_store=InMemoryKVStore(),
    )
    assert wired.is_durable is True
