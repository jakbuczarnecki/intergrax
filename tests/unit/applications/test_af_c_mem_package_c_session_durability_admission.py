# © Artur Czarnecki. All rights reserved.

"""ARCH-FREEZE package C: session/org durability evidence and episodic composition admission."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.applications._shared.memory_provider_admission import (
    MemoryOrgSessionAdmissionError,
    MemoryOrgSessionAdmissionReason,
    validate_episodic_recall_composition_admission,
    validate_organization_and_session_admission,
)
from intergrax.applications._shared.memory_wiring import (
    build_session_manager_from_environment,
    resolve_memory_platform_wiring,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderDurabilityEvidence,
    MemoryProviderDurabilityProofKind,
    MemoryProviderTrustedDurabilityStatus,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.contracts.provider_qualification_evidence import (
    MemoryProviderQualificationEvidence,
)
from intergrax.memory.provider_qualification import InMemoryMemoryProviderDurabilityEvidenceRegistry
from intergrax.memory.provider_qualification.in_memory_evidence_registry import (
    InMemoryMemoryProviderQualificationEvidenceRegistry,
)
from intergrax.runtime.nexus.session.document_store_session_storage import DocumentStoreSessionStorage
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _strict_product(memory: MemoryProfile) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="af.c.mem")
    env.integration_profile = IntegrationProfile()
    env.execution_mode = ExecutionMode.STRICT
    env.memory_profile = memory
    return env


def _session_restart_registry(
    provider_id: str,
    *,
    backing_provider_id: str | None = None,
) -> InMemoryMemoryProviderDurabilityEvidenceRegistry:
    registry = InMemoryMemoryProviderDurabilityEvidenceRegistry()
    registry.register(
        MemoryProviderDurabilityEvidence(
            provider_id=provider_id,
            capability=MemoryProviderCapabilityKind.SESSION_STORAGE,
            durability_status=MemoryProviderTrustedDurabilityStatus.DURABLE,
            qualification_run_id="run-af-c",
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source="test_durability_registry",
            proof_kind=MemoryProviderDurabilityProofKind.RESTART_REOPEN,
            backing_provider_id=backing_provider_id,
        ),
    )
    return registry


def test_session_with_restart_evidence_passes_strict_posture(tmp_path: Path) -> None:
    env = _strict_product(
        MemoryProfile(
            enable_user_memory=True,
            enable_long_term_memory=True,
        ),
    )
    env.integration_profile = IntegrationProfile.lab_harness_preset()
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    qual = InMemoryMemoryProviderQualificationEvidenceRegistry()
    qual.register(
        MemoryProviderQualificationEvidence(
            provider_id="sqlite.user_profile",
            capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
            status=MemoryProviderQualificationStatus.QUALIFIED,
            qualification_run_id="run-af-c",
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source="test_registry",
        ),
    )
    durability = InMemoryMemoryProviderDurabilityEvidenceRegistry()
    for provider_id, capability in (
        ("sqlite.user_profile", MemoryProviderCapabilityKind.USER_PROFILE_STORE),
        ("sqlite.session_storage", MemoryProviderCapabilityKind.SESSION_STORAGE),
    ):
        durability.register(
            MemoryProviderDurabilityEvidence(
                provider_id=provider_id,
                capability=capability,
                durability_status=MemoryProviderTrustedDurabilityStatus.DURABLE,
                qualification_run_id="run-af-c",
                reference_time_iso="2025-01-01T00:00:00+00:00",
                evidence_source="test_durability_registry",
                proof_kind=MemoryProviderDurabilityProofKind.RESTART_REOPEN,
            ),
        )
    wiring = resolve_memory_platform_wiring(
        env,
        tenant_id="tenant-af-c",
        qualification_evidence_registry=qual,
        durability_evidence_registry=durability,
    )
    validate_organization_and_session_admission(
        env,
        session_storage=wiring.session_storage,
        organization_profile_store=wiring.organization_profile_store,
        durability_evidence_registry=durability,
    )


def test_conditional_document_store_without_restart_evidence_fails_strict() -> None:
    env = _strict_product(
        MemoryProfile(enable_user_memory=True, enable_long_term_memory=True),
    )
    storage = DocumentStoreSessionStorage(InMemoryDocumentStore())
    with pytest.raises(MemoryOrgSessionAdmissionError) as refusal:
        validate_organization_and_session_admission(
            env,
            session_storage=storage,
            organization_profile_store=None,
            durability_evidence_registry=InMemoryMemoryProviderDurabilityEvidenceRegistry(),
        )
    assert refusal.value.reason is MemoryOrgSessionAdmissionReason.SESSION_STORAGE_NOT_DURABLE


def test_in_memory_session_passes_when_persistent_memory_not_required() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="af.c.mem.nopersist")
    wiring = resolve_memory_platform_wiring(env, tenant_id="tenant-af-c")
    assert isinstance(wiring.session_storage, InMemorySessionStorage)
    validate_organization_and_session_admission(
        env,
        session_storage=wiring.session_storage,
        organization_profile_store=None,
    )


def test_enabled_episodic_recall_without_implementation_fails_closed() -> None:
    env = _strict_product(MemoryProfile(enable_session_vector_index=True))
    env.integration_profile = IntegrationProfile()
    with pytest.raises(MemoryOrgSessionAdmissionError) as refusal:
        build_session_manager_from_environment(env, tenant_id="tenant-af-c", rag_stack=None)
    assert refusal.value.reason is MemoryOrgSessionAdmissionReason.EPISODIC_RECALL_NOT_BOUND

    with pytest.raises(MemoryOrgSessionAdmissionError) as direct:
        validate_episodic_recall_composition_admission(
            env,
            session_turn_index_store=None,
        )
    assert direct.value.reason is MemoryOrgSessionAdmissionReason.EPISODIC_RECALL_NOT_BOUND


def test_document_store_session_with_restart_evidence_passes() -> None:
    env = _strict_product(
        MemoryProfile(enable_user_memory=True, enable_long_term_memory=True),
    )
    backing = InMemoryDocumentStore()
    storage = DocumentStoreSessionStorage(backing)
    durability = _session_restart_registry(
        "document_store.session_storage",
        backing_provider_id=None,
    )
    validate_organization_and_session_admission(
        env,
        session_storage=storage,
        organization_profile_store=None,
        durability_evidence_registry=durability,
        document_store_backing_provider_id=None,
    )
