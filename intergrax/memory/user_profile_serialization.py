# © Artur Czarnecki. All rights reserved.

"""JSON serialization for UserProfile aggregates (Phase MEM-2.1)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from intergrax.memory.user_profile_memory import (
    MemoryImportance,
    MemoryKind,
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
)


def _memory_kind(value: str) -> MemoryKind:
    try:
        return MemoryKind(value)
    except ValueError:
        return MemoryKind.OTHER


def _memory_importance(value: str) -> MemoryImportance:
    try:
        return MemoryImportance(value)
    except ValueError:
        return MemoryImportance.MEDIUM


def memory_entry_to_dict(entry: UserProfileMemoryEntry) -> Dict[str, Any]:
    return {
        "entry_id": entry.entry_id,
        "content": entry.content,
        "session_id": entry.session_id,
        "kind": entry.kind.value,
        "title": entry.title,
        "importance": entry.importance.value,
        "created_at": entry.created_at,
        "valid_from": entry.valid_from,
        "valid_until": entry.valid_until,
        "metadata": dict(entry.metadata),
        "deleted": entry.deleted,
        "modified": entry.modified,
    }


def memory_entry_from_dict(payload: Dict[str, Any]) -> UserProfileMemoryEntry:
    return UserProfileMemoryEntry(
        entry_id=str(payload.get("entry_id", "")),
        content=str(payload.get("content", "")),
        session_id=payload.get("session_id"),
        kind=_memory_kind(str(payload.get("kind", MemoryKind.OTHER.value))),
        title=payload.get("title"),
        importance=_memory_importance(str(payload.get("importance", MemoryImportance.MEDIUM.value))),
        created_at=str(payload.get("created_at", "")),
        valid_from=payload.get("valid_from"),
        valid_until=payload.get("valid_until"),
        metadata=dict(payload.get("metadata") or {}),
        deleted=bool(payload.get("deleted", False)),
        modified=bool(payload.get("modified", False)),
    )


def user_profile_to_dict(profile: UserProfile) -> Dict[str, Any]:
    identity = profile.identity
    preferences = profile.preferences
    return {
        "identity": {
            "user_id": identity.user_id,
            "display_name": identity.display_name,
            "role": identity.role,
            "domain_expertise": identity.domain_expertise,
            "language": identity.language,
            "locale": identity.locale,
            "timezone": identity.timezone,
        },
        "preferences": {
            "preferred_language": preferences.preferred_language,
            "answer_length": preferences.answer_length,
            "tone": preferences.tone,
            "no_emojis_in_code": preferences.no_emojis_in_code,
            "no_emojis_in_docs": preferences.no_emojis_in_docs,
            "prefer_markdown": preferences.prefer_markdown,
            "prefer_code_blocks": preferences.prefer_code_blocks,
            "default_project_context": preferences.default_project_context,
            "extra": dict(preferences.extra),
        },
        "system_instructions": profile.system_instructions,
        "memory_entries": [memory_entry_to_dict(entry) for entry in profile.memory_entries],
        "version": profile.version,
        "entry_id": profile.entry_id,
        "deleted": profile.deleted,
        "modified": profile.modified,
    }


def user_profile_from_dict(payload: Dict[str, Any]) -> UserProfile:
    identity_raw = payload.get("identity") or {}
    preferences_raw = payload.get("preferences") or {}
    identity = UserIdentity(
        user_id=str(identity_raw.get("user_id", "")),
        display_name=identity_raw.get("display_name"),
        role=identity_raw.get("role"),
        domain_expertise=identity_raw.get("domain_expertise"),
        language=identity_raw.get("language"),
        locale=identity_raw.get("locale"),
        timezone=identity_raw.get("timezone"),
    )
    preferences = UserPreferences(
        preferred_language=preferences_raw.get("preferred_language"),
        answer_length=preferences_raw.get("answer_length"),
        tone=preferences_raw.get("tone"),
        no_emojis_in_code=bool(preferences_raw.get("no_emojis_in_code", False)),
        no_emojis_in_docs=bool(preferences_raw.get("no_emojis_in_docs", False)),
        prefer_markdown=bool(preferences_raw.get("prefer_markdown", True)),
        prefer_code_blocks=bool(preferences_raw.get("prefer_code_blocks", True)),
        default_project_context=preferences_raw.get("default_project_context"),
        extra=dict(preferences_raw.get("extra") or {}),
    )
    entries_raw: List[Dict[str, Any]] = list(payload.get("memory_entries") or [])
    return UserProfile(
        identity=identity,
        preferences=preferences,
        system_instructions=payload.get("system_instructions"),
        memory_entries=[memory_entry_from_dict(item) for item in entries_raw],
        version=int(payload.get("version", 1)),
        entry_id=str(payload.get("entry_id", identity.user_id)),
        deleted=bool(payload.get("deleted", False)),
        modified=bool(payload.get("modified", False)),
    )


def user_profile_to_json(profile: UserProfile) -> str:
    return json.dumps(user_profile_to_dict(profile), ensure_ascii=False)


def user_profile_from_json(raw: str) -> UserProfile:
    return user_profile_from_dict(json.loads(raw))
