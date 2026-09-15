# © Artur Czarnecki. All rights reserved.

"""Materialize canonical memory records from strategy candidates (MEM-ENT-5)."""

from __future__ import annotations

from typing import Any, Dict

from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
    memory_record_source_from_legacy_string,
)
from intergrax.memory.strategies.models import MemoryCandidate
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry


def materialize_user_profile_memory_entry(
    candidate: MemoryCandidate,
    *,
    strategy_id: str | None = None,
    run_id: str | None = None,
    provenance: MemoryProvenance | None = None,
) -> UserProfileMemoryEntry:
    metadata: Dict[str, Any] = {
        "tags": list(candidate.tags),
        "source": candidate.source,
    }
    if candidate.structured_summary is not None:
        metadata["structured_summary"] = candidate.structured_summary.model_dump(mode="json")
    if candidate.kind.value == "episodic_event":
        metadata["structured"] = True

    resolved_provenance = provenance or MemoryProvenance(
        source_type=MemoryRecordSourceType.SESSION_EXTRACTION,
        session_id=candidate.session_id,
        strategy_id=strategy_id or "builtin.memory.extraction.llm_session",
        run_id=run_id,
        source_id=candidate.session_id,
    )
    trust = MemoryRecordTrust(trust_class=MemoryTrustClass.MODEL_INFERENCE)

    return UserProfileMemoryEntry(
        content=candidate.content,
        session_id=candidate.session_id,
        kind=candidate.kind,
        title=candidate.title,
        importance=candidate.importance,
        metadata=metadata,
        provenance=resolved_provenance,
        trust=trust,
        deleted=False,
        modified=False,
        revision=1,
    )


def apply_legacy_metadata_provenance(entry: UserProfileMemoryEntry) -> UserProfileMemoryEntry:
    """Upgrade legacy metadata-only source hints into typed provenance when loading."""
    if entry.provenance.source_type is not MemoryRecordSourceType.UNKNOWN:
        return entry
    legacy_source = entry.metadata.get("source") if entry.metadata else None
    if legacy_source is None and entry.session_id:
        legacy_source = "session_consolidation"
    if legacy_source is None:
        return entry
    entry.provenance = MemoryProvenance(
        source_type=memory_record_source_from_legacy_string(str(legacy_source)),
        session_id=entry.session_id or entry.provenance.session_id,
        source_id=entry.provenance.source_id,
        run_id=entry.provenance.run_id,
        strategy_id=entry.provenance.strategy_id,
        actor_user_id=entry.provenance.actor_user_id,
    )
    return entry
