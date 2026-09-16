# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-5: canonical enterprise memory record."""

from __future__ import annotations

import pytest

from intergrax.contracts.data_classification import DataClassification
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordGovernance,
    MemoryRecordLineage,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
)
from intergrax.memory.memory_entry_materialization import materialize_user_profile_memory_entry
from intergrax.memory.strategies.models import MemoryCandidate
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry
from intergrax.memory.user_profile_serialization import memory_entry_from_dict, memory_entry_to_dict

pytestmark = pytest.mark.gate


def test_new_record_defaults() -> None:
    entry = UserProfileMemoryEntry(content="hello")
    assert entry.memory_id == entry.entry_id
    assert entry.revision == 1
    assert entry.provenance.source_type is MemoryRecordSourceType.UNKNOWN
    assert entry.trust.trust_class is MemoryTrustClass.UNKNOWN
    assert entry.governance.data_classification is DataClassification.INTERNAL


def test_session_extraction_materialization_provenance() -> None:
    candidate = MemoryCandidate(
        content="Prefers concise answers",
        kind=MemoryKind.PREFERENCE,
        session_id="sess-42",
        source="session_consolidation",
    )
    entry = materialize_user_profile_memory_entry(
        candidate,
        strategy_id="builtin.memory.extraction.llm_session",
        run_id="run-99",
    )
    assert entry.provenance.source_type is MemoryRecordSourceType.SESSION_EXTRACTION
    assert entry.provenance.session_id == "sess-42"
    assert entry.provenance.strategy_id == "builtin.memory.extraction.llm_session"
    assert entry.provenance.run_id == "run-99"
    assert entry.trust.trust_class is MemoryTrustClass.MODEL_INFERENCE


def test_temporal_ordering_rejected() -> None:
    with pytest.raises(ValueError, match="valid_from"):
        UserProfileMemoryEntry(
            content="x",
            valid_from="2026-01-02T00:00:00",
            valid_until="2026-01-01T00:00:00",
        )


def test_temporal_naive_valid_ordering_accepted() -> None:
    UserProfileMemoryEntry(
        content="x",
        valid_from="2026-01-01T10:00:00",
        valid_until="2026-01-01T11:00:00",
    )


def test_temporal_naive_invalid_ordering_rejected() -> None:
    with pytest.raises(ValueError, match="valid_from must not be after valid_until"):
        UserProfileMemoryEntry(
            content="x",
            valid_from="2026-01-01T12:00:00",
            valid_until="2026-01-01T11:00:00",
        )


def test_temporal_offset_aware_valid_ordering_accepted() -> None:
    """10:00+02:00 is 08:00 UTC, before 09:30+00:00 — must not use lexical compare."""
    UserProfileMemoryEntry(
        content="x",
        valid_from="2026-01-01T10:00:00+02:00",
        valid_until="2026-01-01T09:30:00+00:00",
    )


def test_temporal_offset_aware_invalid_ordering_rejected() -> None:
    with pytest.raises(ValueError, match="valid_from must not be after valid_until"):
        UserProfileMemoryEntry(
            content="x",
            valid_from="2026-01-01T12:00:00+02:00",
            valid_until="2026-01-01T09:30:00+00:00",
        )


def test_temporal_mixed_naive_and_aware_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware or both naive"):
        UserProfileMemoryEntry(
            content="x",
            valid_from="2026-01-01T10:00:00",
            valid_until="2026-01-01T11:00:00+00:00",
        )


def test_temporal_malformed_valid_from_rejected() -> None:
    with pytest.raises(ValueError, match="invalid valid_from"):
        UserProfileMemoryEntry(
            content="x",
            valid_from="abc",
            valid_until="2026-01-01T11:00:00",
        )


def test_temporal_malformed_valid_until_rejected() -> None:
    with pytest.raises(ValueError, match="invalid valid_until"):
        UserProfileMemoryEntry(
            content="x",
            valid_from="2026-01-01T10:00:00",
            valid_until="2026-13-99",
        )


def test_temporal_z_suffix_accepted() -> None:
    UserProfileMemoryEntry(
        content="x",
        valid_from="2026-01-01T10:00:00Z",
        valid_until="2026-01-01T11:00:00+00:00",
    )


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        UserProfileMemoryEntry(
            content="x",
            trust=MemoryRecordTrust(confidence=-0.1),
        )
    with pytest.raises(ValueError, match="confidence"):
        UserProfileMemoryEntry(
            content="x",
            trust=MemoryRecordTrust(confidence=1.1),
        )


def test_confidence_nan_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        UserProfileMemoryEntry(
            content="x",
            trust=MemoryRecordTrust(confidence=float("nan")),
        )


def test_confidence_positive_infinity_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        UserProfileMemoryEntry(
            content="x",
            trust=MemoryRecordTrust(confidence=float("inf")),
        )


def test_confidence_negative_infinity_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        UserProfileMemoryEntry(
            content="x",
            trust=MemoryRecordTrust(confidence=float("-inf")),
        )


def test_confidence_boundary_values_accepted() -> None:
    UserProfileMemoryEntry(content="lo", trust=MemoryRecordTrust(confidence=0.0))
    UserProfileMemoryEntry(content="hi", trust=MemoryRecordTrust(confidence=1.0))


def test_self_supersession_rejected() -> None:
    entry_id = "abc123"
    with pytest.raises(ValueError, match="supersede"):
        UserProfileMemoryEntry(
            entry_id=entry_id,
            content="x",
            lineage=MemoryRecordLineage(supersedes_memory_id=entry_id),
        )


def test_backward_compatible_constructor() -> None:
    entry = UserProfileMemoryEntry(content="legacy", kind=MemoryKind.USER_FACT)
    assert entry.content == "legacy"
    assert entry.revision == 1


def test_serialization_roundtrip() -> None:
    original = UserProfileMemoryEntry(
        content="stored",
        provenance=MemoryProvenance(
            source_type=MemoryRecordSourceType.USER_EXPLICIT,
            actor_user_id="user-1",
        ),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT, confidence=0.9),
        governance=MemoryRecordGovernance(data_classification=DataClassification.CONFIDENTIAL),
        evidence_refs=("evidence.run-1",),
        lineage=MemoryRecordLineage(supersedes_memory_id="prior-mem"),
        valid_from="2026-01-01T10:00:00",
        valid_until="2026-01-01T12:00:00",
        updated_at="2026-06-15T08:00:00Z",
        revision=2,
    )
    restored = memory_entry_from_dict(memory_entry_to_dict(original))
    assert restored.entry_id == original.entry_id
    assert restored.revision == 2
    assert restored.provenance.source_type is MemoryRecordSourceType.USER_EXPLICIT
    assert restored.trust.confidence == 0.9
    assert restored.governance.data_classification is DataClassification.CONFIDENTIAL
    assert restored.evidence_refs == ("evidence.run-1",)
    assert restored.lineage.supersedes_memory_id == "prior-mem"
    assert restored.valid_from == "2026-01-01T10:00:00"
    assert restored.valid_until == "2026-01-01T12:00:00"
    assert restored.updated_at == "2026-06-15T08:00:00Z"


@pytest.mark.asyncio
async def test_ltm_vector_projection_metadata_includes_memory_id_and_revision() -> None:
    from unittest.mock import MagicMock

    from intergrax.memory.user_profile_ltm_vector_projection import UserProfileLtmVectorProjection
    from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreRecord

    entry = UserProfileMemoryEntry(content="proj", revision=3)
    embedding_manager = MagicMock()
    embedding_manager.embed_texts.return_value = [[0.1, 0.2]]
    captured: list[VectorStoreRecord] = []
    vectorstore_manager = MagicMock()

    def _capture_add(records: list[VectorStoreRecord], *, scope: object) -> None:
        captured.extend(records)

    vectorstore_manager.add_records = _capture_add

    projection = UserProfileLtmVectorProjection(
        embedding_manager=embedding_manager,
        vectorstore_manager=vectorstore_manager,
        tenant_id="tenant-1",
        vector_index_namespace=None,
        workspace_id=None,
    )
    from intergrax.memory.contracts.memory_lifecycle import user_profile_memory_projection_context
    from tests.unit.memory._projection_identity import memory_test_identity

    await projection.upsert_memory_entry(
        user_profile_memory_projection_context(
            memory_test_identity(tenant_id="tenant-1", user_id="user-1")
        ),
        entry,
    )
    meta = captured[0].document.metadata
    assert meta["memory_id"] == entry.memory_id
    assert meta["revision"] == 3


@pytest.mark.asyncio
async def test_update_increments_revision_preserves_memory_id() -> None:
    from unittest.mock import AsyncMock, MagicMock

    from intergrax.memory.user_profile_manager import UserProfileManager
    from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile

    entry = UserProfileMemoryEntry(content="v1")
    memory_id = entry.entry_id
    profile = UserProfile(
        identity=UserIdentity(user_id="u1"),
        preferences=UserPreferences(),
        memory_entries=[entry],
    )
    store = MagicMock()
    store.get_profile = AsyncMock(return_value=profile)
    store.save_profile = AsyncMock()
    mgr = UserProfileManager(store)

    from tests.unit.memory._projection_identity import memory_test_identity

    await mgr.update_memory_entry(
        memory_test_identity(user_id="u1"),
        "u1",
        memory_id,
        content="v2",
    )

    assert profile.memory_entries[0].entry_id == memory_id
    assert profile.memory_entries[0].revision == 2
    assert profile.memory_entries[0].content == "v2"
