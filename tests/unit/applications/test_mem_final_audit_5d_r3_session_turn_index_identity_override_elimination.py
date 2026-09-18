# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5D-R3 — SessionTurnIndex provider identity override elimination."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.memory_vector_wiring import build_session_turn_index_store
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.integrations.registry.catalog_manifests import QDRANT
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.providers.vector_store.qdrant.integration import (
    QDRANT_VECTOR_STORE_PROVIDER_ID,
)
from intergrax.memory.contracts.provider_admission import (
    MemoryProviderAdmissionError,
    MemoryProviderAdmissionReasonCode,
)
from intergrax.memory.contracts.provider_identity import (
    builtin_session_turn_index_store_identity,
    plugin_session_turn_index_store_identity,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.contracts.provider_qualification_evidence import (
    MemoryProviderQualificationEvidence,
)
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStoreCreationContext
from intergrax.memory.provider_qualification.in_memory_evidence_registry import (
    InMemoryMemoryProviderQualificationEvidenceRegistry,
)
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack
from intergrax.rag.profiles.rag_profile import RagProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_EXTERNAL_STI_PLUGIN_ID = "external.sti.foo"
_FACTORY_CALLS = 0

_BUILTIN_QDRANT_EVIDENCE = MemoryProviderQualificationEvidence(
    provider_id="vector.session_turn_index",
    capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
    status=MemoryProviderQualificationStatus.QUALIFIED,
    qualification_run_id="run-builtin-qdrant",
    reference_time_iso="2025-01-01T00:00:00+00:00",
    evidence_source="test",
    backing_provider_id="qdrant",
)


def _registry_with(
    *evidence: MemoryProviderQualificationEvidence,
) -> InMemoryMemoryProviderQualificationEvidenceRegistry:
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


def _product_env() -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5dr3.product")
    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    return env


def _qualified_external_evidence(
    plugin_id: str = _EXTERNAL_STI_PLUGIN_ID,
) -> MemoryProviderQualificationEvidence:
    return MemoryProviderQualificationEvidence(
        provider_id=plugin_id,
        capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
        status=MemoryProviderQualificationStatus.QUALIFIED,
        qualification_run_id="run-external-sti",
        reference_time_iso="2025-01-01T00:00:00+00:00",
        evidence_source="test",
        backing_provider_id=None,
    )


class _ExternalStiFooPlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return _EXTERNAL_STI_PLUGIN_ID

    @classmethod
    def create_session_turn_index(
        cls,
        context: SessionTurnIndexStoreCreationContext,
    ) -> VectorSessionTurnIndexStore:
        global _FACTORY_CALLS
        _FACTORY_CALLS += 1
        if context.embedding_manager is None or context.vectorstore_manager is None:
            raise ValueError("ports required")
        return VectorSessionTurnIndexStore(
            embedding_port=context.embedding_manager,
            vectorstore_port=context.vectorstore_manager,
            index_roles=context.index_roles,
            tenant_id=context.tenant_id,
            vector_index_namespace=context.vector_index_namespace,
            workspace_id=context.workspace_id,
        )


def test_build_session_turn_index_store_has_no_provider_identity_parameter() -> None:
    params = inspect.signature(build_session_turn_index_store).parameters
    assert "provider_identity" not in params


def test_caller_provider_identity_kwarg_rejected() -> None:
    env = _product_env()
    spoofed = builtin_session_turn_index_store_identity(QDRANT_VECTOR_STORE_PROVIDER_ID)
    with pytest.raises(TypeError):
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
            integration_profile=IntegrationProfile(vector_store=QDRANT),
            session_turn_index_plugins=(_ExternalStiFooPlugin,),
            provider_identity=spoofed,
            qualification_evidence_registry=_registry_with(_BUILTIN_QDRANT_EVIDENCE),
        )


def test_attack_external_plugin_builtin_qdrant_evidence_factory_not_called() -> None:
    global _FACTORY_CALLS
    _FACTORY_CALLS = 0
    env = _product_env()
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
            integration_profile=IntegrationProfile(vector_store=QDRANT),
            session_turn_index_plugins=(_ExternalStiFooPlugin,),
            qualification_evidence_registry=_registry_with(_BUILTIN_QDRANT_EVIDENCE),
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING
    assert _FACTORY_CALLS == 0


def test_attack_wrong_plugin_evidence_factory_not_called() -> None:
    global _FACTORY_CALLS
    _FACTORY_CALLS = 0
    env = _product_env()
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
            integration_profile=IntegrationProfile(vector_store=QDRANT),
            session_turn_index_plugins=(_ExternalStiFooPlugin,),
            qualification_evidence_registry=_registry_with(
                _qualified_external_evidence("external.sti.bar"),
            ),
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING
    assert _FACTORY_CALLS == 0


def test_matching_plugin_evidence_materializes_and_calls_factory_once() -> None:
    global _FACTORY_CALLS
    _FACTORY_CALLS = 0
    env = _product_env()
    store = build_session_turn_index_store(
        env,
        tenant_id="tenant-a",
        rag_stack=_rag_stack(),
        integration_profile=IntegrationProfile(vector_store=QDRANT),
        session_turn_index_plugins=(_ExternalStiFooPlugin,),
        qualification_evidence_registry=_registry_with(_qualified_external_evidence()),
    )
    assert isinstance(store, VectorSessionTurnIndexStore)
    assert _FACTORY_CALLS == 1


def test_builtin_qdrant_matching_evidence_passes() -> None:
    env = _product_env()
    store = build_session_turn_index_store(
        env,
        tenant_id="tenant-a",
        rag_stack=_rag_stack(),
        integration_profile=IntegrationProfile(vector_store=QDRANT),
        qualification_evidence_registry=_registry_with(_BUILTIN_QDRANT_EVIDENCE),
    )
    assert isinstance(store, VectorSessionTurnIndexStore)


def test_builtin_qdrant_wrong_backing_evidence_fails() -> None:
    env = _product_env()
    wrong = MemoryProviderQualificationEvidence(
        provider_id="vector.session_turn_index",
        capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
        status=MemoryProviderQualificationStatus.QUALIFIED,
        qualification_run_id="run-pg",
        reference_time_iso="2025-01-01T00:00:00+00:00",
        evidence_source="test",
        backing_provider_id="pgvector",
    )
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
            integration_profile=IntegrationProfile(vector_store=QDRANT),
            qualification_evidence_registry=_registry_with(wrong),
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING


def test_plugin_missing_evidence_fails_product() -> None:
    env = _product_env()
    with pytest.raises(MemoryProviderAdmissionError) as exc_info:
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
            integration_profile=IntegrationProfile(vector_store=QDRANT),
            session_turn_index_plugins=(_ExternalStiFooPlugin,),
        )
    assert exc_info.value.reason_code is MemoryProviderAdmissionReasonCode.QUALIFICATION_EVIDENCE_MISSING


def test_canonical_builder_has_no_identity_override_or_fallback() -> None:
    source = Path("intergrax/applications/_shared/memory_vector_wiring.py").read_text(
        encoding="utf-8",
    )
    builder_block = source.split("def build_session_turn_index_store", 1)[1]
    builder_block = builder_block.split("\ndef ", 1)[0]
    assert "provider_identity:" not in builder_block
    assert "provider_identity or resolve_plugin_session_turn_index_provider_identity" not in builder_block
    assert "provider_identity or resolve_builtin_session_turn_index_provider_identity" not in builder_block


def test_derived_plugin_identity_matches_classifier() -> None:
    identity = plugin_session_turn_index_store_identity(_EXTERNAL_STI_PLUGIN_ID)
    assert identity.provider_id == _EXTERNAL_STI_PLUGIN_ID
    assert identity.backing_provider_id is None
