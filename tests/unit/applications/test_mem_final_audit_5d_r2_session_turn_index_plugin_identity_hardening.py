# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5D-R2 — SessionTurnIndex plugin identity hardening."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.memory_vector_wiring import (
    build_session_turn_index_store,
    resolve_plugin_session_turn_index_provider_identity,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.integrations.registry.catalog_manifests import QDRANT
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.memory.contracts.provider_admission import (
    MemoryProviderAdmissionError,
    MemoryProviderAdmissionReasonCode,
)
from intergrax.memory.contracts.provider_identity import (
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
from intergrax.memory.resolver.classifier import ClassifiedMemoryStorePlugin, MemoryStorePluginKind
from intergrax.memory.resolver.errors import MemoryStorePluginResolutionError
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack
from intergrax.rag.profiles.rag_profile import RagProfile
from tests.fixtures.plugin_packages.session_turn_index_plugin.session_turn_index_plugin.plugin import (
    ExternalSessionTurnIndexStorePlugin,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_EXTERNAL_STI_PLUGIN_ID = "external.sti.foo"
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
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="mem.5dr2.product")
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


class _ExternalStiMissingPluginId:
    @classmethod
    def create_session_turn_index(
        cls,
        context: SessionTurnIndexStoreCreationContext,
    ) -> VectorSessionTurnIndexStore:
        raise AssertionError("factory must not run when plugin_id contract is invalid")


class _ExternalStiEmptyPluginId:
    @classmethod
    def plugin_id(cls) -> str:
        return "   "

    @classmethod
    def create_session_turn_index(
        cls,
        context: SessionTurnIndexStoreCreationContext,
    ) -> VectorSessionTurnIndexStore:
        raise AssertionError("factory must not run when plugin_id is empty")


def test_plugin_identity_has_no_builtin_backing() -> None:
    record = ClassifiedMemoryStorePlugin(
        plugin_id=_EXTERNAL_STI_PLUGIN_ID,
        kind=MemoryStorePluginKind.SESSION_TURN_INDEX,
        plugin_type=_ExternalStiFooPlugin,
    )
    identity = resolve_plugin_session_turn_index_provider_identity(record)
    assert identity.provider_id == _EXTERNAL_STI_PLUGIN_ID
    assert identity.backing_provider_id is None


def test_missing_plugin_id_fails_before_factory() -> None:
    env = _product_env()
    with pytest.raises(MemoryStorePluginResolutionError):
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
            integration_profile=IntegrationProfile(vector_store=QDRANT),
            session_turn_index_plugins=(_ExternalStiMissingPluginId,),
            qualification_evidence_registry=_registry_with(_qualified_external_evidence()),
        )


def test_empty_plugin_id_fails_before_factory() -> None:
    env = _product_env()
    with pytest.raises(MemoryStorePluginResolutionError):
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
            integration_profile=IntegrationProfile(vector_store=QDRANT),
            session_turn_index_plugins=(_ExternalStiEmptyPluginId,),
            qualification_evidence_registry=_registry_with(_qualified_external_evidence()),
        )


def test_builtin_qdrant_evidence_cannot_admit_external_plugin() -> None:
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


def test_wrong_plugin_evidence_fails() -> None:
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


def test_matching_plugin_evidence_materializes_exact_plugin() -> None:
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


def test_ambiguous_explicit_plugin_list_fails_closed() -> None:
    env = _product_env()

    class _OtherPlugin:
        @classmethod
        def plugin_id(cls) -> str:
            return "external.sti.other"

        @classmethod
        def create_session_turn_index(
            cls,
            context: SessionTurnIndexStoreCreationContext,
        ) -> VectorSessionTurnIndexStore:
            raise NotImplementedError

    with pytest.raises(MemoryStorePluginResolutionError, match="Ambiguous"):
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
            integration_profile=IntegrationProfile(vector_store=QDRANT),
            session_turn_index_plugins=(_ExternalStiFooPlugin, _OtherPlugin),
            qualification_evidence_registry=_registry_with(_qualified_external_evidence()),
        )


def test_duplicate_plugin_ids_fail_closed() -> None:
    env = _product_env()

    class _DuplicateIdPlugin:
        @classmethod
        def plugin_id(cls) -> str:
            return _EXTERNAL_STI_PLUGIN_ID

        @classmethod
        def create_session_turn_index(
            cls,
            context: SessionTurnIndexStoreCreationContext,
        ) -> VectorSessionTurnIndexStore:
            raise NotImplementedError

    with pytest.raises(MemoryStorePluginResolutionError, match="Duplicate"):
        build_session_turn_index_store(
            env,
            tenant_id="tenant-a",
            rag_stack=_rag_stack(),
            integration_profile=IntegrationProfile(vector_store=QDRANT),
            session_turn_index_plugins=(_ExternalStiFooPlugin, _DuplicateIdPlugin),
            qualification_evidence_registry=_registry_with(_qualified_external_evidence()),
        )


def test_lab_invalid_plugin_still_fails_contract() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.5dr2.lab")
    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    with pytest.raises(MemoryStorePluginResolutionError):
        build_session_turn_index_store(
            env,
            tenant_id="tenant-lab",
            rag_stack=_rag_stack(),
            integration_profile=IntegrationProfile(vector_store=QDRANT),
            session_turn_index_plugins=(_ExternalStiMissingPluginId,),
        )


def test_fixture_external_plugin_identity_is_plugin_scoped() -> None:
    identity = plugin_session_turn_index_store_identity(
        ExternalSessionTurnIndexStorePlugin.plugin_id(),
    )
    assert identity.provider_id == "external.session_turn_index"
    assert identity.backing_provider_id is None


def test_resolve_identity_path_has_no_getattr() -> None:
    source = Path(
        "intergrax/applications/_shared/memory_vector_wiring.py",
    ).read_text(encoding="utf-8")
    resolve_block = source.split("def resolve_builtin_session_turn_index_provider_identity", 1)[1]
    resolve_block = resolve_block.split("def _require_runtime_tenant", 1)[0]
    assert "getattr(" not in resolve_block
    assert "hasattr(" not in resolve_block
