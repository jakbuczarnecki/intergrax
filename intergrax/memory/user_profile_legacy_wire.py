# © Artur Czarnecki. All rights reserved.

"""Typed historical durable wire for user profiles (evidence @ e5ae6a3)."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, JsonValue

_PRE_R2_R1_EVIDENCE_SHA: Final = "e5ae6a3e358470607c2acd2eeb5a18334f36db63"


class LegacyUserIdentityWire(BaseModel):
    """Fields emitted by ``user_profile_to_dict`` @ ``e5ae6a3`` (identity block)."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = ""
    display_name: str | None = None
    role: str | None = None
    domain_expertise: str | None = None
    language: str | None = None
    locale: str | None = None
    timezone: str | None = None


class LegacyUserPreferencesWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred_language: str | None = None
    answer_length: str | None = None
    tone: str | None = None
    no_emojis_in_code: bool = False
    no_emojis_in_docs: bool = False
    prefer_markdown: bool = True
    prefer_code_blocks: bool = True
    default_project_context: str | None = None
    extra: dict[str, JsonValue] = Field(default_factory=dict)


class LegacyMemoryProvenanceWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: str | None = None
    source_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    strategy_id: str | None = None
    actor_user_id: str | None = None


class LegacyMemoryRecordTrustWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trust_class: str | None = None
    confidence: float | None = None


class LegacyMemoryRecordGovernanceWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_classification: str | None = None


class LegacyMemoryRecordLineageWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supersedes_memory_id: str | None = None
    superseded_by_memory_id: str | None = None


class LegacyUserProfileMemoryEntryWire(BaseModel):
    """Fields emitted by ``memory_entry_to_dict`` @ ``e5ae6a3``."""

    model_config = ConfigDict(extra="forbid")

    entry_id: str | None = None
    revision: int = 1
    content: str = ""
    session_id: str | None = None
    kind: str | None = None
    title: str | None = None
    importance: str | None = None
    created_at: str = ""
    updated_at: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    provenance: LegacyMemoryProvenanceWire | None = None
    trust: LegacyMemoryRecordTrustWire | None = None
    governance: LegacyMemoryRecordGovernanceWire | None = None
    lineage: LegacyMemoryRecordLineageWire | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    deleted: bool = False
    modified: bool = False


class LegacyUserProfileWire(BaseModel):
    """Historical durable JSON before ``schema_version`` (evidence @ e5ae6a3)."""

    model_config = ConfigDict(extra="forbid")

    identity: LegacyUserIdentityWire
    preferences: LegacyUserPreferencesWire = Field(default_factory=LegacyUserPreferencesWire)
    system_instructions: str | None = None
    memory_entries: list[LegacyUserProfileMemoryEntryWire] = Field(default_factory=list)
    version: int = 1
    entry_id: str | None = None
    deleted: bool = False
    modified: bool = False


HISTORICAL_USER_PROFILE_WIRE_EVIDENCE_SHA: Final = _PRE_R2_R1_EVIDENCE_SHA

__all__ = [
    "HISTORICAL_USER_PROFILE_WIRE_EVIDENCE_SHA",
    "LegacyMemoryProvenanceWire",
    "LegacyMemoryRecordGovernanceWire",
    "LegacyMemoryRecordLineageWire",
    "LegacyMemoryRecordTrustWire",
    "LegacyUserIdentityWire",
    "LegacyUserPreferencesWire",
    "LegacyUserProfileMemoryEntryWire",
    "LegacyUserProfileWire",
]
