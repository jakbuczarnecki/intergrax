# © Artur Czarnecki. All rights reserved.

"""Typed row serializers for CE provider handle metadata (no getattr/setattr)."""

from __future__ import annotations

from typing import Any

from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from intergrax.memory.user_profile_serialization import memory_entry_to_dict
from intergrax.runtime.nexus.context.context_builder import RetrievedChunk


def ltm_entry_row(entry: UserProfileMemoryEntry | dict[str, Any]) -> dict[str, Any]:
    if isinstance(entry, dict):
        return dict(entry)
    return memory_entry_to_dict(entry)


def retrieved_chunk_row(chunk: RetrievedChunk | dict[str, Any]) -> dict[str, Any]:
    if isinstance(chunk, dict):
        metadata = dict(chunk.get("metadata") or {})
        if chunk.get("id") is not None:
            metadata.setdefault("id", chunk.get("id"))
        if chunk.get("score") is not None:
            metadata.setdefault("score", chunk.get("score"))
        text = str(chunk.get("text") or chunk.get("content") or "").strip()
        return {"text": text, "metadata": metadata}
    metadata = dict(chunk.metadata)
    metadata.setdefault("id", chunk.id)
    metadata.setdefault("score", chunk.score)
    return {"text": str(chunk.text or "").strip(), "metadata": metadata}
