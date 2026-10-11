# © Artur Czarnecki. All rights reserved.

"""UserProfile legacy wire migration semantic equivalence (COMPAT-X-R2-R2-R1-R1-R1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.contracts.data_classification import DataClassification
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryRecordSourceType,
    MemoryTrustClass,
)
from intergrax.memory.user_profile_legacy_wire import (
    LegacyUserProfileMemoryEntryWire,
    LegacyUserProfileWire,
)
from intergrax.memory.user_profile_serialization import (
    deserialize_user_profile_payload,
    memory_entry_from_dict,
)

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "compat_x"
    / "historical_user_profile_wire_e5ae6a3.json"
)


def _historical_dict_from_legacy_memory_entry(
    entry: LegacyUserProfileMemoryEntryWire,
) -> dict[str, object]:
    """Pre-R1-R1-R1 intermediate dict shape (historical bypass path)."""
    provenance_payload = entry.provenance.model_dump() if entry.provenance is not None else None
    return {
        "entry_id": entry.entry_id,
        "revision": entry.revision,
        "content": entry.content,
        "session_id": entry.session_id,
        "kind": entry.kind,
        "title": entry.title,
        "importance": entry.importance,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
        "valid_from": entry.valid_from,
        "valid_until": entry.valid_until,
        "provenance": provenance_payload,
        "trust": entry.trust.model_dump() if entry.trust is not None else None,
        "governance": entry.governance.model_dump() if entry.governance is not None else None,
        "lineage": entry.lineage.model_dump() if entry.lineage is not None else None,
        "evidence_refs": list(entry.evidence_refs),
        "metadata": dict(entry.metadata),
        "deleted": entry.deleted,
        "modified": entry.modified,
    }


def test_user_profile_legacy_migration_matches_historical_memory_entry_dict_path() -> None:
    evidence = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert evidence["provenance_sha"]

    payload = {
        "identity": {"user_id": "u-legacy", "display_name": "Legacy User"},
        "preferences": {"preferred_language": "pl", "extra": {"tier": "gold"}},
        "system_instructions": "be helpful",
        "version": 2,
        "entry_id": "profile-entry-1",
        "memory_entries": [
            {
                "entry_id": "mem-1",
                "revision": 4,
                "content": "remember this",
                "session_id": "sess-9",
                "kind": "not_a_real_kind",
                "title": "t",
                "importance": "not_real",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "valid_from": "2024-01-01",
                "valid_until": "2025-01-01",
                "provenance": {
                    "source_type": "tool_result",
                    "source_id": "src-1",
                    "session_id": "sess-9",
                },
                "trust": {"trust_class": "model_inference", "confidence": 0.87},
                "governance": {"data_classification": "internal"},
                "lineage": {
                    "supersedes_memory_id": "old-mem",
                    "superseded_by_memory_id": None,
                },
                "evidence_refs": ["ev-1", "ev-2"],
                "metadata": {"source": "session_extraction", "note": {"nested": True}},
                "deleted": False,
                "modified": True,
            },
            {
                "entry_id": "mem-2",
                "content": "provenance from metadata only",
                "metadata": {"source": "user"},
                "provenance": {},
            },
        ],
    }
    legacy_wire = LegacyUserProfileWire.model_validate(payload)
    profile = deserialize_user_profile_payload(payload)

    assert profile.version == 2
    assert profile.entry_id == "profile-entry-1"
    assert profile.identity.user_id == "u-legacy"
    assert profile.preferences.preferred_language == "pl"
    assert profile.preferences.extra == {"tier": "gold"}

    for wire_entry, domain_entry in zip(legacy_wire.memory_entries, profile.memory_entries, strict=True):
        reference = memory_entry_from_dict(_historical_dict_from_legacy_memory_entry(wire_entry))
        assert domain_entry == reference

    first = profile.memory_entries[0]
    assert first.provenance.source_type == MemoryRecordSourceType.TOOL_RESULT
    assert first.trust.trust_class == MemoryTrustClass.MODEL_INFERENCE
    assert first.trust.confidence == 0.87
    assert first.governance.data_classification == DataClassification.INTERNAL
    assert first.lineage.supersedes_memory_id == "old-mem"
    assert first.evidence_refs == ("ev-1", "ev-2")
    assert first.metadata["note"] == {"nested": True}

    second = profile.memory_entries[1]
    assert second.entry_id
    assert second.provenance.source_type == MemoryRecordSourceType.USER_EXPLICIT
