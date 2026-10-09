# © Artur Czarnecki. All rights reserved.

"""Provider-backed configuration opportunity and integration provenance persistence (TRACE-X-P5-R2-P2)."""

from __future__ import annotations

import json
from typing import Any

from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.contracts.execution_integration_configuration_provenance import (
    ConfiguredIntegrationProvenanceSlice,
    ExecutionIntegrationConfigurationProvenance,
    ExecutionIntegrationConfigurationProvenanceMode,
    IntegrationConfigurationSubject,
    require_tenant_id_for_integration_configuration_provenance,
    validate_execution_integration_configuration_provenance_record,
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
    ExecutionIntegrationConfigurationPinningStore,
    validate_pin_subject_against_provenance,
)
from intergrax.integrations.contracts.existing_capability_configuration_opportunity import (
    ConfigurationOpportunityRef,
    ExistingCapabilityConfigurationOpportunity,
    ExistingCapabilityConfigurationOpportunityLookupError,
    ExistingCapabilityConfigurationOpportunityLookupFailureReason,
    ExistingCapabilityConfigurationOpportunityStore,
    validate_configuration_opportunity_ref,
    validate_existing_capability_configuration_opportunity,
)
from intergrax.integrations.contracts.integration_configuration_payload_codec import (
    IntegrationConfigurationPayloadCodecRegistry,
)
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)

_OPPORTUNITY_KV_PREFIX = "configuration_opportunity"
_OPPORTUNITY_DOCUMENT_PARTITION_PREFIX = "intergrax.configuration_opportunity.v1"
_OPPORTUNITY_SCHEMA_VERSION = 1

_PROVENANCE_KV_PREFIX = "integration_config_provenance"
_PROVENANCE_INDEX_KV_PREFIX = "integration_config_provenance_index"
_PROVENANCE_DOCUMENT_PARTITION_PREFIX = "intergrax.integration_config_provenance_pinning.v1"
_PROVENANCE_SCHEMA_VERSION = 1
_PROVENANCE_DOCUMENT_QUERY_PAGE_SIZE = 1000


def _opportunity_kv_key(configuration_ref: ConfigurationOpportunityRef) -> str:
    return f"{_OPPORTUNITY_KV_PREFIX}:{configuration_ref}"


def _opportunity_document_partition(tenant_id: str) -> str:
    return f"{_OPPORTUNITY_DOCUMENT_PARTITION_PREFIX}:{tenant_id}"


def _subject_row_key(subject: IntegrationConfigurationSubject) -> str:
    return (
        f"{subject.integration_category.value}\x1f"
        f"{subject.provider_id}\x1f"
        f"{subject.resource_scope}\x1f"
        f"{subject.configuration_type}"
    )


def _provenance_kv_key(execution_id: ExecutionId, subject: IntegrationConfigurationSubject) -> str:
    return f"{_PROVENANCE_KV_PREFIX}:{execution_id}:{_subject_row_key(subject)}"


def _provenance_index_kv_key(execution_id: ExecutionId) -> str:
    return f"{_PROVENANCE_INDEX_KV_PREFIX}:{execution_id}"


def _provenance_document_partition(tenant_id: str, execution_id: ExecutionId) -> str:
    return f"{_PROVENANCE_DOCUMENT_PARTITION_PREFIX}:{tenant_id}:{execution_id}"


def encode_configuration_opportunity(
    opportunity: ExistingCapabilityConfigurationOpportunity,
    *,
    payload_codecs: IntegrationConfigurationPayloadCodecRegistry,
) -> bytes:
    validate_existing_capability_configuration_opportunity(opportunity)
    config_type, payload_record = payload_codecs.encode(opportunity.configuration)
    envelope: dict[str, Any] = {
        "schema_version": _OPPORTUNITY_SCHEMA_VERSION,
        "record": {
            "configuration_ref": opportunity.configuration_ref,
            "tenant_id": opportunity.tenant_id,
            "integration_category": opportunity.integration_category.value,
            "provider_id": opportunity.provider_id,
            "resource_scope": opportunity.resource_scope,
            "current_revision": opportunity.current_revision,
            "configuration_type": config_type,
            "configuration_payload": payload_record,
            "configuration_fingerprint": opportunity.configuration_fingerprint,
            "risk_classification": opportunity.risk_classification.value,
        },
    }
    return json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_configuration_opportunity(
    raw: bytes,
    *,
    payload_codecs: IntegrationConfigurationPayloadCodecRegistry,
) -> ExistingCapabilityConfigurationOpportunity:
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.CORRUPT_RECORD,
            detail="invalid opportunity encoding",
        ) from exc
    if not isinstance(envelope, dict):
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.CORRUPT_RECORD,
            detail="invalid opportunity envelope",
        )
    if envelope.get("schema_version") != _OPPORTUNITY_SCHEMA_VERSION:
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.UNSUPPORTED_SCHEMA_VERSION,
        )
    record = envelope.get("record")
    if not isinstance(record, dict):
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.CORRUPT_RECORD,
            detail="missing opportunity record",
        )
    try:
        configuration_ref = validate_configuration_opportunity_ref(record["configuration_ref"])
        tenant_id = record["tenant_id"]
        category = IntegrationCategory(record["integration_category"])
        provider_id = record["provider_id"]
        resource_scope = record["resource_scope"]
        current_revision = record["current_revision"]
        configuration_type = record["configuration_type"]
        configuration_payload = record["configuration_payload"]
        configuration_fingerprint = record["configuration_fingerprint"]
        risk_value = record["risk_classification"]
        configuration = payload_codecs.decode(
            configuration_type=configuration_type,
            payload=configuration_payload,
        )
        risk = ControlPlaneMutationRisk(risk_value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.CORRUPT_RECORD,
            detail=str(exc),
        ) from exc
    opportunity = ExistingCapabilityConfigurationOpportunity(
        configuration_ref=configuration_ref,
        tenant_id=tenant_id,
        integration_category=category,
        provider_id=provider_id,
        resource_scope=resource_scope,
        current_revision=current_revision,
        configuration=configuration,
        configuration_fingerprint=configuration_fingerprint,
        risk_classification=risk,
    )
    validate_existing_capability_configuration_opportunity(opportunity)
    return opportunity


def _encode_configured_slice(slice_: ConfiguredIntegrationProvenanceSlice) -> dict[str, Any]:
    return {
        "tenant_id": slice_.tenant_id,
        "integration_category": slice_.integration_category.value,
        "provider_id": slice_.provider_id,
        "resource_scope": slice_.resource_scope,
        "configuration_type": slice_.configuration_type,
        "configuration_version": slice_.configuration_version,
        "configuration_fingerprint": slice_.configuration_fingerprint,
        "realization_evidence_refs": list(slice_.realization_evidence_refs),
    }


def _decode_configured_slice(raw: object) -> ConfiguredIntegrationProvenanceSlice | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ExecutionIntegrationConfigurationPinningError(
            ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
            detail="invalid configured slice",
        )
    return ConfiguredIntegrationProvenanceSlice(
        tenant_id=raw["tenant_id"],
        integration_category=IntegrationCategory(raw["integration_category"]),
        provider_id=raw["provider_id"],
        resource_scope=raw["resource_scope"],
        configuration_type=raw["configuration_type"],
        configuration_version=raw["configuration_version"],
        configuration_fingerprint=raw["configuration_fingerprint"],
        realization_evidence_refs=tuple(raw.get("realization_evidence_refs", ())),
    )


def encode_integration_configuration_provenance(
    provenance: ExecutionIntegrationConfigurationProvenance,
    *,
    subject: IntegrationConfigurationSubject,
) -> bytes:
    validate_pin_subject_against_provenance(subject=subject, provenance=provenance)
    configured_raw: dict[str, Any] | None
    if provenance.configured is None:
        configured_raw = None
    else:
        configured_raw = _encode_configured_slice(provenance.configured)
    envelope: dict[str, Any] = {
        "schema_version": _PROVENANCE_SCHEMA_VERSION,
        "record": {
            "tenant_id": provenance.tenant_id,
            "execution_id": str(provenance.execution_id),
            "mode": provenance.mode.value,
            "effective": {
                "integration_category": provenance.effective.integration_category.value,
                "provider_id": provenance.effective.provider_id,
                "materialization_kind": provenance.effective.materialization_kind.value,
            },
            "configured": configured_raw,
            "pin_subject": {
                "integration_category": subject.integration_category.value,
                "provider_id": subject.provider_id,
                "resource_scope": subject.resource_scope,
                "configuration_type": subject.configuration_type,
            },
        },
    }
    return json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_integration_configuration_provenance(
    raw: bytes,
) -> tuple[IntegrationConfigurationSubject, ExecutionIntegrationConfigurationProvenance]:
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionIntegrationConfigurationPinningError(
            ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
            detail="invalid provenance encoding",
        ) from exc
    if not isinstance(envelope, dict):
        raise ExecutionIntegrationConfigurationPinningError(
            ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
            detail="invalid provenance envelope",
        )
    if envelope.get("schema_version") != _PROVENANCE_SCHEMA_VERSION:
        raise ExecutionIntegrationConfigurationPinningError(
            ExecutionIntegrationConfigurationPinningFailureReason.UNSUPPORTED_SCHEMA_VERSION,
        )
    record = envelope.get("record")
    if not isinstance(record, dict):
        raise ExecutionIntegrationConfigurationPinningError(
            ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
            detail="missing provenance record",
        )
    try:
        subject_raw = record["pin_subject"]
        effective_raw = record["effective"]
        subject = IntegrationConfigurationSubject(
            integration_category=IntegrationCategory(subject_raw["integration_category"]),
            provider_id=subject_raw["provider_id"],
            resource_scope=subject_raw["resource_scope"],
            configuration_type=subject_raw["configuration_type"],
        )
        provenance = ExecutionIntegrationConfigurationProvenance(
            tenant_id=record["tenant_id"],
            execution_id=validate_execution_id(record["execution_id"]),
            mode=ExecutionIntegrationConfigurationProvenanceMode(record["mode"]),
            effective=EffectiveIntegrationIdentity(
                integration_category=IntegrationCategory(effective_raw["integration_category"]),
                provider_id=effective_raw["provider_id"],
                materialization_kind=IntegrationMaterializationKind(
                    effective_raw["materialization_kind"],
                ),
            ),
            configured=_decode_configured_slice(record.get("configured")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExecutionIntegrationConfigurationPinningError(
            ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
            detail=str(exc),
        ) from exc
    validate_pin_subject_against_provenance(subject=subject, provenance=provenance)
    return subject, provenance


def _require_opportunity_tenant(tenant_id: str) -> str:
    if type(tenant_id) is not str or not tenant_id or tenant_id != tenant_id.strip():
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.INVALID,
            detail="invalid tenant_id",
        )
    return tenant_id


def _opportunities_equal(
    left: ExistingCapabilityConfigurationOpportunity,
    right: ExistingCapabilityConfigurationOpportunity,
) -> bool:
    return left == right


class KvExistingCapabilityConfigurationOpportunityStore:
    """DistributedKVStore-backed immutable opportunity store."""

    def __init__(
        self,
        kv_store: DistributedKVStore,
        *,
        payload_codecs: IntegrationConfigurationPayloadCodecRegistry,
    ) -> None:
        self._kv_store = kv_store
        self._payload_codecs = payload_codecs

    @property
    def is_durable(self) -> bool:
        return True

    def persist(self, opportunity: ExistingCapabilityConfigurationOpportunity) -> None:
        validate_existing_capability_configuration_opportunity(opportunity)
        encoded = encode_configuration_opportunity(
            opportunity,
            payload_codecs=self._payload_codecs,
        )
        key = _opportunity_kv_key(opportunity.configuration_ref)
        if self._kv_store.compare_and_set(
            tenant_id=opportunity.tenant_id,
            key=key,
            expected=None,
            new_value=encoded,
        ):
            return
        existing_raw = self._kv_store.get(tenant_id=opportunity.tenant_id, key=key)
        if existing_raw is None:
            raise ExistingCapabilityConfigurationOpportunityLookupError(
                ExistingCapabilityConfigurationOpportunityLookupFailureReason.CORRUPT_RECORD,
                detail="opportunity compare-and-set failed",
            )
        existing = decode_configuration_opportunity(
            existing_raw,
            payload_codecs=self._payload_codecs,
        )
        if not _opportunities_equal(existing, opportunity):
            raise ExistingCapabilityConfigurationOpportunityLookupError(
                ExistingCapabilityConfigurationOpportunityLookupFailureReason.CONFLICT,
            )

    def read_exact(
        self,
        *,
        tenant_id: str,
        configuration_ref: ConfigurationOpportunityRef,
    ) -> ExistingCapabilityConfigurationOpportunity:
        tenant = _require_opportunity_tenant(tenant_id)
        ref = validate_configuration_opportunity_ref(configuration_ref)
        raw = self._kv_store.get(tenant_id=tenant, key=_opportunity_kv_key(ref))
        if raw is None:
            raise ExistingCapabilityConfigurationOpportunityLookupError(
                ExistingCapabilityConfigurationOpportunityLookupFailureReason.NOT_FOUND,
            )
        opportunity = decode_configuration_opportunity(
            raw,
            payload_codecs=self._payload_codecs,
        )
        if opportunity.tenant_id != tenant:
            raise ExistingCapabilityConfigurationOpportunityLookupError(
                ExistingCapabilityConfigurationOpportunityLookupFailureReason.TENANT_MISMATCH,
            )
        if opportunity.configuration_ref != ref:
            raise ExistingCapabilityConfigurationOpportunityLookupError(
                ExistingCapabilityConfigurationOpportunityLookupFailureReason.CORRUPT_RECORD,
                detail="configuration_ref mismatch",
            )
        return opportunity


class DocumentStoreExistingCapabilityConfigurationOpportunityStore:
    """ConditionalDocumentStore-backed immutable opportunity store."""

    def __init__(
        self,
        document_store: ConditionalDocumentStore,
        *,
        payload_codecs: IntegrationConfigurationPayloadCodecRegistry,
    ) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "configuration opportunity persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store
        self._payload_codecs = payload_codecs

    @property
    def is_durable(self) -> bool:
        return True

    def persist(self, opportunity: ExistingCapabilityConfigurationOpportunity) -> None:
        validate_existing_capability_configuration_opportunity(opportunity)
        partition = _opportunity_document_partition(opportunity.tenant_id)
        document = DocumentRecord(
            partition_key=partition,
            row_key=opportunity.configuration_ref,
            data={
                "opportunity": encode_configuration_opportunity(
                    opportunity,
                    payload_codecs=self._payload_codecs,
                ).decode("utf-8"),
            },
        )
        if self._document_store.put_if_absent(document):
            return
        existing = self._document_store.get(partition, opportunity.configuration_ref)
        if existing is None:
            raise ExistingCapabilityConfigurationOpportunityLookupError(
                ExistingCapabilityConfigurationOpportunityLookupFailureReason.CORRUPT_RECORD,
                detail="opportunity document create failed",
            )
        stored = decode_configuration_opportunity(
            _opportunity_record_to_bytes(existing),
            payload_codecs=self._payload_codecs,
        )
        if not _opportunities_equal(stored, opportunity):
            raise ExistingCapabilityConfigurationOpportunityLookupError(
                ExistingCapabilityConfigurationOpportunityLookupFailureReason.CONFLICT,
            )

    def read_exact(
        self,
        *,
        tenant_id: str,
        configuration_ref: ConfigurationOpportunityRef,
    ) -> ExistingCapabilityConfigurationOpportunity:
        tenant = _require_opportunity_tenant(tenant_id)
        ref = validate_configuration_opportunity_ref(configuration_ref)
        record = self._document_store.get(_opportunity_document_partition(tenant), ref)
        if record is None:
            raise ExistingCapabilityConfigurationOpportunityLookupError(
                ExistingCapabilityConfigurationOpportunityLookupFailureReason.NOT_FOUND,
            )
        opportunity = decode_configuration_opportunity(
            _opportunity_record_to_bytes(record),
            payload_codecs=self._payload_codecs,
        )
        if opportunity.tenant_id != tenant:
            raise ExistingCapabilityConfigurationOpportunityLookupError(
                ExistingCapabilityConfigurationOpportunityLookupFailureReason.TENANT_MISMATCH,
            )
        return opportunity


def _opportunity_record_to_bytes(record: DocumentRecord) -> bytes:
    raw = record.data.get("opportunity")
    if not isinstance(raw, str):
        raise ExistingCapabilityConfigurationOpportunityLookupError(
            ExistingCapabilityConfigurationOpportunityLookupFailureReason.CORRUPT_RECORD,
            detail="invalid opportunity document record",
        )
    return raw.encode("utf-8")


def _append_provenance_index_entry(
    kv_store: DistributedKVStore,
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    subject_key: str,
) -> None:
    index_key = _provenance_index_kv_key(execution_id)
    while True:
        raw = kv_store.get(tenant_id=tenant_id, key=index_key)
        if raw is None:
            payload = json.dumps([subject_key], separators=(",", ":")).encode("utf-8")
            if kv_store.compare_and_set(
                tenant_id=tenant_id,
                key=index_key,
                expected=None,
                new_value=payload,
            ):
                return
            continue
        try:
            existing = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
                detail="invalid provenance index",
            ) from exc
        if not isinstance(existing, list) or not all(type(x) is str for x in existing):
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
                detail="invalid provenance index shape",
            )
        if subject_key in existing:
            return
        updated = sorted(set(existing + [subject_key]))
        new_raw = json.dumps(updated, separators=(",", ":")).encode("utf-8")
        if kv_store.compare_and_set(
            tenant_id=tenant_id,
            key=index_key,
            expected=raw,
            new_value=new_raw,
        ):
            return


def _sorted_provenance_records(
    records: list[ExecutionIntegrationConfigurationProvenance],
) -> tuple[ExecutionIntegrationConfigurationProvenance, ...]:
    return tuple(records)


def _parse_provenance_index_subject_keys(
    subject_keys: list[object],
) -> tuple[str, ...]:
    if not all(type(x) is str for x in subject_keys):
        raise ExecutionIntegrationConfigurationPinningError(
            ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
            detail="invalid provenance index entry",
        )
    typed = [x for x in subject_keys if type(x) is str]
    if len(typed) != len(set(typed)):
        raise ExecutionIntegrationConfigurationPinningError(
            ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
            detail="duplicate provenance index subject key",
        )
    return tuple(sorted(typed))


def _subject_sort_key(subject: IntegrationConfigurationSubject) -> tuple[str, str, str, str]:
    return (
        subject.integration_category.value,
        subject.provider_id,
        subject.resource_scope,
        subject.configuration_type,
    )


class KvExecutionIntegrationConfigurationPinningStore:
    """DistributedKVStore-backed integration configuration provenance pinning."""

    def __init__(self, kv_store: DistributedKVStore) -> None:
        self._kv_store = kv_store

    @property
    def is_durable(self) -> bool:
        return True

    def pin(
        self,
        *,
        subject: IntegrationConfigurationSubject,
        provenance: ExecutionIntegrationConfigurationProvenance,
    ) -> None:
        validate_pin_subject_against_provenance(subject=subject, provenance=provenance)
        tenant = require_tenant_id_for_integration_configuration_provenance(provenance.tenant_id)
        execution_id = validate_execution_id(provenance.execution_id)
        encoded = encode_integration_configuration_provenance(provenance, subject=subject)
        key = _provenance_kv_key(execution_id, subject)
        subject_key = _subject_row_key(subject)
        _append_provenance_index_entry(
            self._kv_store,
            tenant_id=tenant,
            execution_id=execution_id,
            subject_key=subject_key,
        )
        if self._kv_store.compare_and_set(
            tenant_id=tenant,
            key=key,
            expected=None,
            new_value=encoded,
        ):
            return
        existing_raw = self._kv_store.get(tenant_id=tenant, key=key)
        if existing_raw is None:
            if self._kv_store.compare_and_set(
                tenant_id=tenant,
                key=key,
                expected=None,
                new_value=encoded,
            ):
                return
            existing_raw = self._kv_store.get(tenant_id=tenant, key=key)
        if existing_raw is None:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
                detail="provenance record missing after index marker",
            )
        _, existing = decode_integration_configuration_provenance(existing_raw)
        if existing != provenance:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.CONFLICT,
            )

    def read_all(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> tuple[ExecutionIntegrationConfigurationProvenance, ...]:
        tenant = require_tenant_id_for_integration_configuration_provenance(tenant_id)
        validated_execution_id = validate_execution_id(execution_id)
        index_raw = self._kv_store.get(
            tenant_id=tenant,
            key=_provenance_index_kv_key(validated_execution_id),
        )
        if index_raw is None:
            return ()
        try:
            subject_keys = json.loads(index_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
                detail="invalid provenance index",
            ) from exc
        if not isinstance(subject_keys, list):
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
                detail="invalid provenance index shape",
            )
        pairs: list[tuple[IntegrationConfigurationSubject, ExecutionIntegrationConfigurationProvenance]] = []
        for subject_key in _parse_provenance_index_subject_keys(subject_keys):
            pin_key = f"{_PROVENANCE_KV_PREFIX}:{validated_execution_id}:{subject_key}"
            raw = self._kv_store.get(tenant_id=tenant, key=pin_key)
            if raw is None:
                raise ExecutionIntegrationConfigurationPinningError(
                    ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
                    detail="missing provenance pin for index entry",
                )
            subject, record = decode_integration_configuration_provenance(raw)
            validate_execution_integration_configuration_provenance_record(
                record,
                expected_tenant_id=tenant,
                expected_execution_id=validated_execution_id,
            )
            pairs.append((subject, record))
        pairs.sort(key=lambda item: _subject_sort_key(item[0]))
        return _sorted_provenance_records([record for _, record in pairs])


class DocumentStoreExecutionIntegrationConfigurationPinningStore:
    """ConditionalDocumentStore-backed integration configuration provenance pinning."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "integration configuration provenance pinning requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    @property
    def is_durable(self) -> bool:
        return True

    def pin(
        self,
        *,
        subject: IntegrationConfigurationSubject,
        provenance: ExecutionIntegrationConfigurationProvenance,
    ) -> None:
        validate_pin_subject_against_provenance(subject=subject, provenance=provenance)
        tenant = require_tenant_id_for_integration_configuration_provenance(provenance.tenant_id)
        execution_id = validate_execution_id(provenance.execution_id)
        partition = _provenance_document_partition(tenant, execution_id)
        row_key = _subject_row_key(subject)
        document = DocumentRecord(
            partition_key=partition,
            row_key=row_key,
            data={
                "provenance": encode_integration_configuration_provenance(
                    provenance,
                    subject=subject,
                ).decode("utf-8"),
            },
        )
        if self._document_store.put_if_absent(document):
            return
        existing = self._document_store.get(partition, row_key)
        if existing is None:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
                detail="provenance document create failed",
            )
        _, stored = decode_integration_configuration_provenance(_provenance_record_to_bytes(existing))
        if stored != provenance:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.CONFLICT,
            )

    def read_all(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> tuple[ExecutionIntegrationConfigurationProvenance, ...]:
        tenant = require_tenant_id_for_integration_configuration_provenance(tenant_id)
        validated_execution_id = validate_execution_id(execution_id)
        partition = _provenance_document_partition(tenant, validated_execution_id)
        pairs: list[tuple[IntegrationConfigurationSubject, ExecutionIntegrationConfigurationProvenance]] = []
        seen_row_keys: set[str] = set()
        seen_subjects: set[tuple[str, str, str, str]] = set()
        seen_cursors: set[str] = set()
        cursor: str | None = None
        while True:
            if cursor is not None:
                if cursor in seen_cursors:
                    raise ExecutionIntegrationConfigurationPinningError(
                        ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
                        detail="non-progressing document provenance query cursor",
                    )
                seen_cursors.add(cursor)
            page = self._document_store.query(
                partition,
                limit=_PROVENANCE_DOCUMENT_QUERY_PAGE_SIZE,
                cursor=cursor,
            )
            for document in page.documents:
                if document.row_key in seen_row_keys:
                    raise ExecutionIntegrationConfigurationPinningError(
                        ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
                        detail="duplicate provenance document row key",
                    )
                seen_row_keys.add(document.row_key)
                subject, record = decode_integration_configuration_provenance(
                    _provenance_record_to_bytes(document),
                )
                if document.row_key != _subject_row_key(subject):
                    raise ExecutionIntegrationConfigurationPinningError(
                        ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
                        detail="provenance document row key does not match decoded subject",
                    )
                validate_execution_integration_configuration_provenance_record(
                    record,
                    expected_tenant_id=tenant,
                    expected_execution_id=validated_execution_id,
                )
                subject_identity = _subject_sort_key(subject)
                if subject_identity in seen_subjects:
                    raise ExecutionIntegrationConfigurationPinningError(
                        ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
                        detail="duplicate provenance subject in document query result",
                    )
                seen_subjects.add(subject_identity)
                pairs.append((subject, record))
            next_cursor = page.next_cursor
            if next_cursor is None:
                break
            if next_cursor == cursor:
                raise ExecutionIntegrationConfigurationPinningError(
                    ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
                    detail="non-progressing document provenance query cursor",
                )
            cursor = next_cursor
        pairs.sort(key=lambda item: _subject_sort_key(item[0]))
        return _sorted_provenance_records([record for _, record in pairs])


def _provenance_record_to_bytes(record: DocumentRecord) -> bytes:
    raw = record.data.get("provenance")
    if not isinstance(raw, str):
        raise ExecutionIntegrationConfigurationPinningError(
            ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
            detail="invalid provenance document record",
        )
    return raw.encode("utf-8")


class InMemoryExistingCapabilityConfigurationOpportunityStore:
    """Non-production reference opportunity store."""

    def __init__(
        self,
        *,
        payload_codecs: IntegrationConfigurationPayloadCodecRegistry,
    ) -> None:
        self._payload_codecs = payload_codecs
        self._records: dict[tuple[str, str], ExistingCapabilityConfigurationOpportunity] = {}

    @property
    def is_durable(self) -> bool:
        return False

    def persist(self, opportunity: ExistingCapabilityConfigurationOpportunity) -> None:
        validate_existing_capability_configuration_opportunity(opportunity)
        key = (opportunity.tenant_id, opportunity.configuration_ref)
        if key in self._records and self._records[key] != opportunity:
            raise ExistingCapabilityConfigurationOpportunityLookupError(
                ExistingCapabilityConfigurationOpportunityLookupFailureReason.CONFLICT,
            )
        self._records[key] = opportunity

    def read_exact(
        self,
        *,
        tenant_id: str,
        configuration_ref: ConfigurationOpportunityRef,
    ) -> ExistingCapabilityConfigurationOpportunity:
        tenant = _require_opportunity_tenant(tenant_id)
        ref = validate_configuration_opportunity_ref(configuration_ref)
        record = self._records.get((tenant, ref))
        if record is None:
            raise ExistingCapabilityConfigurationOpportunityLookupError(
                ExistingCapabilityConfigurationOpportunityLookupFailureReason.NOT_FOUND,
            )
        return record


class InMemoryExecutionIntegrationConfigurationPinningStore:
    """Non-production reference provenance pinning store."""

    def __init__(self) -> None:
        self._records: dict[
            tuple[str, str, str, str, str, str],
            ExecutionIntegrationConfigurationProvenance,
        ] = {}

    @property
    def is_durable(self) -> bool:
        return False

    def pin(
        self,
        *,
        subject: IntegrationConfigurationSubject,
        provenance: ExecutionIntegrationConfigurationProvenance,
    ) -> None:
        validate_pin_subject_against_provenance(subject=subject, provenance=provenance)
        tenant = require_tenant_id_for_integration_configuration_provenance(provenance.tenant_id)
        execution_id = validate_execution_id(provenance.execution_id)
        key = (
            tenant,
            str(execution_id),
            subject.integration_category.value,
            subject.provider_id,
            subject.resource_scope,
            subject.configuration_type,
        )
        if key in self._records and self._records[key] != provenance:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.CONFLICT,
            )
        self._records[key] = provenance

    def read_all(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> tuple[ExecutionIntegrationConfigurationProvenance, ...]:
        tenant = require_tenant_id_for_integration_configuration_provenance(tenant_id)
        validated_execution_id = validate_execution_id(execution_id)
        matches = [
            record
            for key, record in self._records.items()
            if key[0] == tenant and key[1] == str(validated_execution_id)
        ]
        return _sorted_provenance_records(matches)


def wire_existing_capability_configuration_opportunity_store(
    *,
    payload_codecs: IntegrationConfigurationPayloadCodecRegistry,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
) -> ExistingCapabilityConfigurationOpportunityStore:
    codecs = payload_codecs
    if kv_store is not None and document_store is not None:
        raise ValueError(
            "wire_existing_capability_configuration_opportunity_store accepts one backing store",
        )
    if kv_store is not None:
        return KvExistingCapabilityConfigurationOpportunityStore(kv_store, payload_codecs=codecs)
    if document_store is not None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "configuration opportunity persistence requires ConditionalDocumentStore",
            )
        return DocumentStoreExistingCapabilityConfigurationOpportunityStore(
            document_store,
            payload_codecs=codecs,
        )
    raise ValueError(
        "wire_existing_capability_configuration_opportunity_store requires a backing store",
    )


def wire_execution_integration_configuration_pinning_store(
    *,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
) -> ExecutionIntegrationConfigurationPinningStore:
    if kv_store is not None and document_store is not None:
        raise ValueError(
            "wire_execution_integration_configuration_pinning_store accepts one backing store",
        )
    if kv_store is not None:
        return KvExecutionIntegrationConfigurationPinningStore(kv_store)
    if document_store is not None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "integration configuration provenance pinning requires ConditionalDocumentStore",
            )
        return DocumentStoreExecutionIntegrationConfigurationPinningStore(document_store)
    raise ValueError(
        "wire_execution_integration_configuration_pinning_store requires a backing store",
    )


__all__ = [
    "DocumentStoreExistingCapabilityConfigurationOpportunityStore",
    "DocumentStoreExecutionIntegrationConfigurationPinningStore",
    "InMemoryExistingCapabilityConfigurationOpportunityStore",
    "InMemoryExecutionIntegrationConfigurationPinningStore",
    "KvExistingCapabilityConfigurationOpportunityStore",
    "KvExecutionIntegrationConfigurationPinningStore",
    "decode_configuration_opportunity",
    "decode_integration_configuration_provenance",
    "encode_configuration_opportunity",
    "encode_integration_configuration_provenance",
    "wire_existing_capability_configuration_opportunity_store",
    "wire_execution_integration_configuration_pinning_store",
]
