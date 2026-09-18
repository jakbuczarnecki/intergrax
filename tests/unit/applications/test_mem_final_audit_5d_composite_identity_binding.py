# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5D — composite STI adapter/backend qualification identity binding."""

from __future__ import annotations

import pytest

from intergrax.integrations.providers.vector_store.qdrant.integration import (
    QDRANT_VECTOR_STORE_PROVIDER_ID,
)
from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderDurabilityEvidence,
    MemoryProviderDurabilityProofKind,
    MemoryProviderTrustedDurabilityStatus,
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
from intergrax.memory.provider_qualification.in_memory_durability_evidence_registry import (
    InMemoryMemoryProviderDurabilityEvidenceRegistry,
)
from intergrax.memory.provider_qualification.in_memory_evidence_registry import (
    InMemoryMemoryProviderQualificationEvidenceRegistry,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_registry_distinguishes_qdrant_from_pgvector_backing() -> None:
    registry = InMemoryMemoryProviderQualificationEvidenceRegistry()
    registry.register(
        MemoryProviderQualificationEvidence(
            provider_id=BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
            capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
            status=MemoryProviderQualificationStatus.QUALIFIED,
            qualification_run_id="run-qdrant",
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source="test",
            backing_provider_id=QDRANT_VECTOR_STORE_PROVIDER_ID,
        ),
    )
    qdrant_lookup = registry.resolve(
        BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
        MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
        None,
        QDRANT_VECTOR_STORE_PROVIDER_ID,
    )
    assert qdrant_lookup.evidence is not None
    pg_lookup = registry.resolve(
        BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
        MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
        None,
        "pgvector",
    )
    assert pg_lookup.resolve_status is MemoryProviderQualificationEvidenceResolveStatus.MISSING


def test_adapter_only_evidence_cannot_satisfy_qdrant_backing() -> None:
    registry = InMemoryMemoryProviderQualificationEvidenceRegistry()
    registry.register(
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
    lookup = registry.resolve(
        BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
        MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
        None,
        QDRANT_VECTOR_STORE_PROVIDER_ID,
    )
    assert lookup.resolve_status is MemoryProviderQualificationEvidenceResolveStatus.MISSING


def test_durability_registry_requires_matching_backing() -> None:
    registry = InMemoryMemoryProviderDurabilityEvidenceRegistry()
    registry.register(
        MemoryProviderDurabilityEvidence(
            provider_id=BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
            capability=MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,
            durability_status=MemoryProviderTrustedDurabilityStatus.DURABLE,
            qualification_run_id="run-qdrant",
            reference_time_iso="2025-01-01T00:00:00+00:00",
            evidence_source="test",
            proof_kind=MemoryProviderDurabilityProofKind.REAL_VENDOR_RECONNECT,
            backing_provider_id=QDRANT_VECTOR_STORE_PROVIDER_ID,
        ),
    )
    identity = builtin_session_turn_index_store_identity(
        BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
        backing_provider_id="pgvector",
    )
    lookup = registry.resolve(
        identity.provider_id,
        identity.capability,
        None,
        identity.backing_provider_id,
    )
    assert lookup.evidence is None
