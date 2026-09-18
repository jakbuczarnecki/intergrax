# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5D-R — SessionTurnIndex trusted production admission."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.memory_vector_wiring import (
    build_session_turn_index_store,
    resolve_session_turn_index_provider_identity,
    vector_store_backing_provider_id,
)
from intergrax.applications._shared.memory_wiring import build_session_manager_from_environment
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.integrations.providers.vector_store.qdrant.integration import (
    QDRANT_VECTOR_STORE_PROVIDER_ID,
)
from intergrax.integrations.registry.catalog_manifests import QDRANT
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.memory.contracts.provider_admission import (
    MemoryProviderAdmissionError,
    MemoryProviderAdmissionReasonCode,
    evaluate_production_session_turn_index_store_admission,
    lookup_trusted_memory_provider_qualification_evidence,
)
from intergrax.memory.contracts.provider_identity import (
    BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
    builtin_session_turn_index_store_identity,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.contracts.provider_qualification_evidence import (
    MemoryProviderQualificationEvidence,
    MemoryProviderQualificationEvidenceResolveStatus,
)
from intergrax.memory.memory_vector_errors import MemoryVectorBackendUnavailableError
from intergrax.memory.provider_qualification.in_memory_evidence_registry import (
    InMemoryMemoryProviderQualificationEvidenceRegistry,
)
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.applications._shared.memory_wiring import MemoryPlatformWiring

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _qdrant_sti_identity() -> object:
    return builtin_session_turn_index_store_identity(
        BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
        backing_provider_id=QDRANT_VECTOR_STORE_PROVIDER_ID,
    )


def _qdrant_qualified_evidence(run_id: str = "run-qdrant") -> MemoryProviderQualificationEvidence:
    return MemoryProviderQualificationEvidence(
        provider_id=BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
        capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
        status=MemoryProviderQualificationStatus.QUALIFIED,
        qualification_run_id=run_id,
        reference_time_iso="2025-01-01T00:00:00+00:00",
        evidence_source="test",
        backing_provider_id=QDRANT_VECTOR_STORE_PROVIDER_ID,
    )


def _registry_with(*evidence: MemoryProviderQualificationEvidence) -> InMemoryMemoryProviderQualificationEvidenceRegistry:
    registry = InMemoryMemoryProviderQualificationEvidenceRegistry()
    for item in evidence:
        registry.register(item)
    return registry


def _rag_stack() -> RagStack:
    return RagStack(
        profile=RagProfile(),
        vectorstore_manager=MagicMock(),
        embedding_manager=MagicMock(),
        retriever_manager=MagicMock(),
        reranker_manager=MagicMock(),
        retrieval_service=MagicMock(),
    )


def test_vector_store_backing_from_integration_profile() -> None:
    profile = IntegrationProfile(vector_store=QDRANT)
    assert vector_store_backing_provider_id(profile) == QDRANT_VECTOR_STORE_PROVIDER_ID


def test_product_qdrant_runtime_with_matching_evidence_passes() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5dr.pass")
    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    integration = IntegrationProfile(vector_store=QDRANT)
    registry = _registry_with(_qdrant_qualified_evidence())
    store = build_session_turn_index_store(
        env,
        tenant_id="tenant-a",
        rag_stack=_rag_stack(),
        integration_profile=integration,
        qualification_evidence_registry=registry,
    )
    assert isinstance(store, VectorSessionTurnIndexStore)


def test_product_without_evidence_fails_closed() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5dr.no.evidence")
    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    integration = IntegrationProfile(vector_store=QDRANT)
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
            integration_profile=integration,
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING


def test_product_pgvector_evidence_with_qdrant_runtime_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5dr.wrong.backing")
    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    integration = IntegrationProfile(vector_store=QDRANT)
    registry = _registry_with(
        MemoryProviderQualificationEvidence(
            provider_id=BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
            capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
            status=MemoryProviderQualificationStatus.QUALIFIED,
            qualification_run_id="run-pg",
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source="test",
            backing_provider_id="pgvector",
        ),
    )
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
            integration_profile=integration,
            qualification_evidence_registry=registry,
        )
    assert exc_info.value.reason_code in {
        MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING,
        MemoryProviderAdmissionReasonCode.PROVIDER_BACKING_IDENTITY_MISMATCH,
    }


def test_product_adapter_only_evidence_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5dr.adapter")
    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    integration = IntegrationProfile(vector_store=QDRANT)
    registry = _registry_with(
        MemoryProviderQualificationEvidence(
            provider_id=BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
            capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
            status=MemoryProviderQualificationStatus.QUALIFIED,
            qualification_run_id="run-generic",
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source="test",
            backing_provider_id=None,
        ),
    )
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
            integration_profile=integration,
            qualification_evidence_registry=registry,
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING


def test_product_not_qualified_evidence_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5dr.not.qualified")
    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    integration = IntegrationProfile(vector_store=QDRANT)
    evidence = _qdrant_qualified_evidence()
    registry = _registry_with(
        replace(evidence, status=MemoryProviderQualificationStatus.NOT_QUALIFIED),
    )
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
            integration_profile=integration,
            qualification_evidence_registry=registry,
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.PROVIDER_NOT_QUALIFIED


def test_product_missing_vector_backend_fails_explicitly() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5dr.no.backend")
    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    with pytest.raises(MemoryVectorBackendUnavailableError):
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=None,
            integration_profile=IntegrationProfile(vector_store=QDRANT),
            qualification_evidence_registry=_registry_with(_qdrant_qualified_evidence()),
        )


def test_lab_sti_without_evidence_remains_legal() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.5dr.lab")
    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    store = build_session_turn_index_store(
        env,
        tenant_id="tenant-lab",
        rag_stack=_rag_stack(),
        integration_profile=IntegrationProfile(vector_store=QDRANT),
    )
    assert isinstance(store, VectorSessionTurnIndexStore)


def test_sti_disabled_skips_admission() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5dr.disabled")
    env.memory_profile = MemoryProfile(enable_session_vector_index=False)
    assert (
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
        )
        is None
    )


def test_direct_injection_without_identity_fails_product() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5dr.inject")
    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    wiring = MemoryPlatformWiring(
        session_storage=InMemorySessionStorage(),
        user_profile_store=InMemoryUserProfileStore(),
        organization_profile_store=None,
    )
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        build_session_manager_from_environment(
            env,
            tenant_id="tenant-a",
            memory_wiring=wiring,
            rag_stack=_rag_stack(),
            session_turn_index_store=MagicMock(),
            session_turn_index_store_identity=None,
            qualification_evidence_registry=_registry_with(_qdrant_qualified_evidence()),
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.PROVIDER_IDENTITY_MISSING


def test_ambiguous_evidence_fails_evaluation() -> None:
    identity = _qdrant_sti_identity()
    lookup = lookup_trusted_memory_provider_qualification_evidence(
        _registry_with(_qdrant_qualified_evidence(), _qdrant_qualified_evidence("run-2")),
        identity,  # type: ignore[arg-type]
    )
    assert lookup.resolve_status is MemoryProviderQualificationEvidenceResolveStatus.AMBIGUOUS
    evaluation = evaluate_production_session_turn_index_store_admission(identity, lookup)  # type: ignore[arg-type]
    assert evaluation.admitted is False
    assert evaluation.reason_code is MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISMATCH


def test_resolve_identity_uses_platform_vector_binding() -> None:
    identity = resolve_session_turn_index_provider_identity(
        IntegrationProfile(vector_store=QDRANT),
    )
    assert identity.backing_provider_id == QDRANT_VECTOR_STORE_PROVIDER_ID
