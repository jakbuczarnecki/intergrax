# © Artur Czarnecki. All rights reserved.

"""NPSC-5F v1/v2 compatibility and migration matrices."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.npsc5f_compatibility import (
    AmbiguousLegacyExecutionIdentityError,
    BackgroundExecutionIdentityConflictError,
    ForbiddenPlatformCausalEvidenceV1WriteError,
    LegacyBackgroundExecutionIdentityIncompatibleError,
    LegacyCausalEvidenceIncompatibleError,
    UnknownCausalEvidenceExportVersionError,
    UnknownPlatformCausalEvidenceSchemaError,
)
from intergrax.runtime.background_execution.identity_persistence import (
    DocumentStoreBackgroundExecutionIdentityPersistence,
    KvBackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.identity_record_codec import (
    BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V1,
    BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V2,
    decode_background_identity_kv_record,
    encode_background_identity_v2_record,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.runtime.observability.causal_evidence import (
    PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.runtime.observability.causal_evidence_enrichment import (
    CanonicalExecutionIdLookupPort,
    CanonicalExecutionIdLookupResult,
    enrich_decoded_causal_evidence,
)
from intergrax.runtime.observability.causal_evidence_export import (
    CausalEvidenceExportVersion,
    causal_evidence_export_source_from_evidence,
    envelope_from_causal_evidence,
    export_source_for_version,
    legacy_causal_evidence_export_source_from_evidence,
)
from intergrax.runtime.observability.causal_evidence_legacy import (
    PLATFORM_CAUSAL_EVIDENCE_SCHEMA_V1,
    LegacyPlatformCausalEvidence,
)
from intergrax.runtime.observability.causal_evidence_record_codec import (
    decode_causal_evidence_record,
    decode_causal_evidence_record_v2,
    encode_causal_evidence_record,
    forbid_platform_causal_evidence_v1_write,
)
from intergrax.runtime.observability.platform_causal_evidence_codec import (
    decode_platform_causal_evidence_payload,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import DocumentRecord
from tests.unit.runtime.background_execution.test_background_execution_identity import (
    _KV,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"


def _v2_evidence() -> PlatformCausalEvidence:
    return PlatformCausalEvidence(
        relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
        tenant_id=_TENANT,
        source=MessageBusTaskRef(provider="celery", task_id="t-1", tenant_id=_TENANT),
        target=RuntimeExecutionRef(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
            tenant_id=_TENANT,
        ),
    )


def _v1_legacy_payload(*, evidence_id: str | None = None) -> dict[str, object]:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    return {
        "schema_version": PLATFORM_CAUSAL_EVIDENCE_SCHEMA_V1,
        "evidence_id": evidence_id or mint_event_id(),
        "relation_kind": CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION.value,
        "tenant_id": _TENANT,
        "source": {
            "provider": "celery",
            "task_id": "t-1",
            "tenant_id": _TENANT,
        },
        "target": {
            "task_id": str(task_id),
            "run_id": str(run_id),
            "attempt_id": str(attempt_id),
            "tenant_id": _TENANT,
        },
        "recorded_at": "2026-01-01T00:00:00+00:00",
    }


class _OutcomeLookup(CanonicalExecutionIdLookupPort):
    def __init__(self, result: CanonicalExecutionIdLookupResult) -> None:
        self._result = result

    def lookup_execution_id(self, **kwargs: object) -> CanonicalExecutionIdLookupResult:
        _ = kwargs
        return self._result


class _TenantScopedLookup(CanonicalExecutionIdLookupPort):
    def __init__(self, *, by_tenant: dict[str, str]) -> None:
        self._by_tenant = by_tenant

    def lookup_execution_id(
        self,
        *,
        tenant_id: str,
        task_id: object,
        run_id: object,
        attempt_id: object,
    ) -> CanonicalExecutionIdLookupResult:
        _ = (task_id, run_id, attempt_id)
        resolved = self._by_tenant.get(tenant_id)
        if resolved is None:
            return CanonicalExecutionIdLookupResult(outcome="unresolved")
        return CanonicalExecutionIdLookupResult(
            outcome="resolved",
            execution_id=resolved,  # type: ignore[arg-type]
        )


def test_v2_deserialize_and_write_round_trip() -> None:
    evidence = _v2_evidence()
    decoded = decode_causal_evidence_record(encode_causal_evidence_record(evidence))
    assert decoded.kind == "complete_v2"
    restored = decode_causal_evidence_record_v2(encode_causal_evidence_record(evidence))
    assert restored.target.execution_id == evidence.target.execution_id


def test_v1_deserialize_legacy_incomplete() -> None:
    decoded = decode_platform_causal_evidence_payload(_v1_legacy_payload())
    assert decoded.kind == "legacy_incomplete_v1"
    assert decoded.legacy_v1 is not None
    assert decoded.legacy_v1.schema_version == PLATFORM_CAUSAL_EVIDENCE_SCHEMA_V1


def test_v1_write_forbidden() -> None:
    with pytest.raises(ForbiddenPlatformCausalEvidenceV1WriteError):
        forbid_platform_causal_evidence_v1_write(_v1_legacy_payload())


def test_v1_enrichment_resolved_with_canonical_lookup() -> None:
    execution_id = mint_execution_id()
    decoded = decode_platform_causal_evidence_payload(_v1_legacy_payload())
    lookup = _OutcomeLookup(
        CanonicalExecutionIdLookupResult(outcome="resolved", execution_id=execution_id),
    )
    outcome = enrich_decoded_causal_evidence(decoded, lookup=lookup)
    assert outcome.status == "resolved"
    assert outcome.complete_v2 is not None
    assert outcome.complete_v2.target.execution_id == execution_id


def test_v1_enrichment_unresolved_fail_closed() -> None:
    decoded = decode_platform_causal_evidence_payload(_v1_legacy_payload())
    lookup = _OutcomeLookup(CanonicalExecutionIdLookupResult(outcome="unresolved"))
    with pytest.raises(LegacyCausalEvidenceIncompatibleError):
        enrich_decoded_causal_evidence(decoded, lookup=lookup)


def test_v1_enrichment_ambiguous_fail_closed() -> None:
    decoded = decode_platform_causal_evidence_payload(_v1_legacy_payload())
    lookup = _OutcomeLookup(CanonicalExecutionIdLookupResult(outcome="ambiguous"))
    with pytest.raises(AmbiguousLegacyExecutionIdentityError):
        enrich_decoded_causal_evidence(decoded, lookup=lookup)


def test_unknown_platform_schema_typed_failure() -> None:
    with pytest.raises(UnknownPlatformCausalEvidenceSchemaError):
        decode_platform_causal_evidence_payload(
            {"schema_version": "platform_causal_evidence.v9"}
        )


def test_v2_export_default_requires_target_execution_id() -> None:
    evidence = _v2_evidence()
    source = causal_evidence_export_source_from_evidence(evidence)
    assert source.schema_version == "causal_evidence_export_source.v2"
    assert source.target_execution_id == evidence.target.execution_id
    envelope = envelope_from_causal_evidence(evidence)
    assert envelope.source_schema_id == PLATFORM_CAUSAL_EVIDENCE_SCHEMA
    assert envelope.causal_evidence_source is not None
    assert (
        envelope.causal_evidence_source.schema_version
        == "causal_evidence_export_source.v2"
    )


def test_v1_export_explicit_legacy_only() -> None:
    legacy = LegacyPlatformCausalEvidence.model_validate(_v1_legacy_payload())
    source = legacy_causal_evidence_export_source_from_evidence(legacy)
    assert source.schema_version == "causal_evidence_export_source.v1"
    assert not hasattr(source, "target_execution_id")


def test_unknown_export_version_typed_failure() -> None:
    with pytest.raises(UnknownCausalEvidenceExportVersionError):
        export_source_for_version(
            _v2_evidence(),
            requested_version=CausalEvidenceExportVersion.V1_LEGACY,
        )


def test_kv_v2_read_preferred() -> None:
    kv = _KV()
    persistence = KvBackgroundExecutionIdentityPersistence(kv)
    transport = BackgroundTransportExecutionRef(
        tenant_id=_TENANT,
        provider="celery",
        transport_task_id="transport-1",
    )
    execution_id = mint_execution_id()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    kv.set(
        _TENANT,
        "bg_exec_identity:celery:transport-1",
        encode_background_identity_v2_record(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ),
    )
    loaded = persistence.load(transport)
    assert loaded is not None
    assert loaded.execution_id == execution_id


def test_kv_v1_only_fail_closed_without_lookup() -> None:
    kv = _KV()
    persistence = KvBackgroundExecutionIdentityPersistence(kv)
    transport = BackgroundTransportExecutionRef(
        tenant_id=_TENANT,
        provider="celery",
        transport_task_id="legacy-only",
    )
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    kv.set(
        _TENANT,
        "bg_exec_identity:celery:legacy-only",
        f"{task_id}\n{run_id}\n{attempt_id}".encode("utf-8"),
    )
    with pytest.raises(LegacyBackgroundExecutionIdentityIncompatibleError):
        persistence.load(transport)


def test_kv_v1_enriched_when_lookup_unique() -> None:
    kv = _KV()
    execution_id = mint_execution_id()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    lookup = _TenantScopedLookup(by_tenant={_TENANT: execution_id})
    persistence = KvBackgroundExecutionIdentityPersistence(kv, legacy_lookup=lookup)
    transport = BackgroundTransportExecutionRef(
        tenant_id=_TENANT,
        provider="celery",
        transport_task_id="legacy-enrich",
    )
    kv.set(
        _TENANT,
        "bg_exec_identity:celery:legacy-enrich",
        f"{task_id}\n{run_id}\n{attempt_id}".encode("utf-8"),
    )
    loaded = persistence.load(transport)
    assert loaded is not None
    assert loaded.execution_id == execution_id


def test_document_v1_v2_conflict_fail_closed() -> None:
    store = InMemoryDocumentStore()
    persistence = DocumentStoreBackgroundExecutionIdentityPersistence(store)
    transport = BackgroundTransportExecutionRef(
        tenant_id=_TENANT,
        provider="celery",
        transport_task_id="conflict",
    )
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    store.put_if_absent(
        DocumentRecord(
            partition_key=f"{BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V2}:{_TENANT}",
            row_key="celery:conflict",
            data={
                "task_id": str(task_id),
                "run_id": str(run_id),
                "attempt_id": str(attempt_id),
                "execution_id": str(mint_execution_id()),
            },
        )
    )
    other_task = mint_task_id()
    store.put_if_absent(
        DocumentRecord(
            partition_key=f"{BG_EXEC_IDENTITY_DOCUMENT_PARTITION_V1}:{_TENANT}",
            row_key="celery:conflict",
            data={
                "task_id": str(other_task),
                "run_id": str(run_id),
                "attempt_id": str(attempt_id),
            },
        )
    )
    with pytest.raises(BackgroundExecutionIdentityConflictError):
        persistence.load(transport)


def test_cross_tenant_lookup_never_resolves_other_tenant_execution_id() -> None:
    execution_a = mint_execution_id()
    lookup = _TenantScopedLookup(by_tenant={"tenant-a": execution_a})
    kv = _KV()
    persistence = KvBackgroundExecutionIdentityPersistence(kv, legacy_lookup=lookup)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    transport_b = BackgroundTransportExecutionRef(
        tenant_id="tenant-b",
        provider="celery",
        transport_task_id="cross-tenant",
    )
    kv.set(
        "tenant-b",
        "bg_exec_identity:celery:cross-tenant",
        f"{task_id}\n{run_id}\n{attempt_id}".encode("utf-8"),
    )
    with pytest.raises(LegacyBackgroundExecutionIdentityIncompatibleError):
        persistence.load(transport_b)


def test_migration_enrichment_does_not_import_identity_mint() -> None:
    import intergrax.runtime.observability.causal_evidence_enrichment as enrichment_mod
    import intergrax.runtime.background_execution.identity_dual_read as dual_read_mod

    enrichment_source = enrichment_mod.__file__
    dual_read_source = dual_read_mod.__file__
    assert enrichment_source is not None
    assert dual_read_source is not None
    for path in (enrichment_source, dual_read_source):
        text = open(path, encoding="utf-8").read()
        assert "mint_execution_id" not in text
        assert "mint_background_transport_identity" not in text


def test_decode_background_identity_kv_v1_v2_shapes() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    legacy = decode_background_identity_kv_record(
        f"{task_id}\n{run_id}\n{attempt_id}".encode("utf-8")
    )
    assert legacy.kind == "legacy_v1"
    complete = decode_background_identity_kv_record(
        encode_background_identity_v2_record(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
    )
    assert complete.kind == "complete_v2"
    assert complete.execution_id == execution_id
