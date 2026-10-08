# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P2 durable opportunity and provenance persistence tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
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
    decode_configuration_opportunity,
    encode_configuration_opportunity,
    encode_integration_configuration_provenance,
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


class InMemoryKVStore(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

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
        current = self._data.get((tenant_id, key))
        if current != expected:
            return False
        self._data[(tenant_id, key)] = new_value
        return True


class InMemoryConditionalDocumentStore(ConditionalDocumentStore):
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], DocumentRecord] = {}

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
        del cursor, row_key_upper_bound, data_equalities, sort
        docs = [
            record
            for (partition, _), record in self._records.items()
            if partition == partition_key
            and (row_key_prefix is None or record.row_key.startswith(row_key_prefix))
        ]
        docs.sort(key=lambda d: d.row_key)
        return DocumentQueryPageV1(documents=tuple(docs[:limit]))

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
    store_a.pin(subject=subject, provenance=record)
    store_a.pin(subject=subject, provenance=record)
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
    store.pin(subject=_subject(), provenance=base)
    store.pin(subject=subject_b, provenance=record_b)
    records = store.read_all(tenant_id="tenant-a", execution_id=_EXEC)
    assert records == (record_b, base)


def test_provenance_conflict_and_invalid_execution_id() -> None:
    store = InMemoryExecutionIntegrationConfigurationPinningStore()
    subject = _subject()
    record = _provenance_configured_adopted()
    store.pin(subject=subject, provenance=record)
    conflicting = replace(
        record,
        configured=_configured_slice(configuration_fingerprint="fp-changed"),
    )
    with pytest.raises(ExecutionIntegrationConfigurationPinningError) as exc:
        store.pin(subject=subject, provenance=conflicting)
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
    store.pin(subject=subject, provenance=record)
    assert store.read_all(tenant_id="tenant-a", execution_id=_EXEC) == (record,)


def test_neutral_reader_is_read_only_projection() -> None:
    pinning = InMemoryExecutionIntegrationConfigurationPinningStore()
    subject = _subject()
    record = _provenance_configured_adopted()
    pinning.pin(subject=subject, provenance=record)
    reader = PinningStoreExecutionIntegrationConfigurationProvenanceReader(pinning)
    assert reader.read_all(tenant_id="tenant-a", execution_id=_EXEC) == (record,)
    assert not hasattr(reader, "pin")


def test_in_memory_stores_are_non_durable() -> None:
    assert InMemoryExistingCapabilityConfigurationOpportunityStore(
        payload_codecs=_test_codecs(),
    ).is_durable is False
    assert InMemoryExecutionIntegrationConfigurationPinningStore().is_durable is False
