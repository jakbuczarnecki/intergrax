# © Artur Czarnecki. All rights reserved.

"""JSON serialization for UserProfile aggregates (Phase MEM-2.1)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Any, Dict, Final, List

from intergrax.contracts.data_classification import DataClassification
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordGovernance,
    MemoryRecordLineage,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
    memory_record_source_from_legacy_string,
)
from intergrax.memory.memory_entry_materialization import apply_legacy_metadata_provenance
from intergrax.memory.user_profile_legacy_wire import (
    LegacyMemoryProvenanceWire,
    LegacyMemoryRecordGovernanceWire,
    LegacyMemoryRecordLineageWire,
    LegacyMemoryRecordTrustWire,
    LegacyUserProfileMemoryEntryWire,
    LegacyUserProfileWire,
)
from intergrax.memory.user_profile_memory import (
    MemoryImportance,
    MemoryKind,
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
)

SCHEMA_USER_PROFILE_V1: Final = "user_profile.v1"
LEGACY_UNVERSIONED_PRE_V1: Final = "LEGACY_UNVERSIONED_PRE_V1"


def _is_legacy_user_profile_mapping(payload: Mapping[str, object]) -> bool:
    return "schema_version" not in payload and isinstance(payload.get("identity"), dict)


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


def _provenance_to_dict(provenance: MemoryProvenance) -> Dict[str, Any]:
    return {
        "source_type": provenance.source_type.value,
        "source_id": provenance.source_id,
        "session_id": provenance.session_id,
        "run_id": provenance.run_id,
        "strategy_id": provenance.strategy_id,
        "actor_user_id": provenance.actor_user_id,
    }


def _provenance_from_dict(payload: Dict[str, Any] | None) -> MemoryProvenance:
    if not payload:
        return MemoryProvenance()
    try:
        source_type = MemoryRecordSourceType(str(payload.get("source_type", MemoryRecordSourceType.UNKNOWN.value)))
    except ValueError:
        source_type = MemoryRecordSourceType.UNKNOWN
    return MemoryProvenance(
        source_type=source_type,
        source_id=payload.get("source_id"),
        session_id=payload.get("session_id"),
        run_id=payload.get("run_id"),
        strategy_id=payload.get("strategy_id"),
        actor_user_id=payload.get("actor_user_id"),
    )


def _trust_to_dict(trust: MemoryRecordTrust) -> Dict[str, Any]:
    return {
        "trust_class": trust.trust_class.value,
        "confidence": trust.confidence,
    }


def _trust_from_dict(payload: Dict[str, Any] | None) -> MemoryRecordTrust:
    if not payload:
        return MemoryRecordTrust()
    try:
        trust_class = MemoryTrustClass(str(payload.get("trust_class", MemoryTrustClass.UNKNOWN.value)))
    except ValueError:
        trust_class = MemoryTrustClass.UNKNOWN
    confidence = payload.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
    return MemoryRecordTrust(trust_class=trust_class, confidence=confidence)


def _governance_to_dict(governance: MemoryRecordGovernance) -> Dict[str, Any]:
    return {"data_classification": governance.data_classification.value}


def _governance_from_dict(payload: Dict[str, Any] | None) -> MemoryRecordGovernance:
    if not payload:
        return MemoryRecordGovernance()
    raw = payload.get("data_classification", DataClassification.INTERNAL.value)
    try:
        classification = DataClassification(str(raw).lower())
    except ValueError:
        classification = DataClassification.INTERNAL
    return MemoryRecordGovernance(data_classification=classification)


def _lineage_to_dict(lineage: MemoryRecordLineage) -> Dict[str, Any]:
    return {
        "supersedes_memory_id": lineage.supersedes_memory_id,
        "superseded_by_memory_id": lineage.superseded_by_memory_id,
    }


def _lineage_from_dict(payload: Dict[str, Any] | None) -> MemoryRecordLineage:
    if not payload:
        return MemoryRecordLineage()
    return MemoryRecordLineage(
        supersedes_memory_id=payload.get("supersedes_memory_id"),
        superseded_by_memory_id=payload.get("superseded_by_memory_id"),
    )


def _provenance_from_legacy_wire(wire: LegacyMemoryProvenanceWire | None) -> MemoryProvenance:
    if wire is None:
        return MemoryProvenance()
    try:
        source_type = MemoryRecordSourceType(
            str(wire.source_type or MemoryRecordSourceType.UNKNOWN.value)
        )
    except ValueError:
        source_type = MemoryRecordSourceType.UNKNOWN
    return MemoryProvenance(
        source_type=source_type,
        source_id=wire.source_id,
        session_id=wire.session_id,
        run_id=wire.run_id,
        strategy_id=wire.strategy_id,
        actor_user_id=wire.actor_user_id,
    )


def _trust_from_legacy_wire(wire: LegacyMemoryRecordTrustWire | None) -> MemoryRecordTrust:
    if wire is None:
        return MemoryRecordTrust()
    try:
        trust_class = MemoryTrustClass(str(wire.trust_class or MemoryTrustClass.UNKNOWN.value))
    except ValueError:
        trust_class = MemoryTrustClass.UNKNOWN
    confidence = wire.confidence
    if confidence is not None:
        confidence = float(confidence)
    return MemoryRecordTrust(trust_class=trust_class, confidence=confidence)


def _governance_from_legacy_wire(wire: LegacyMemoryRecordGovernanceWire | None) -> MemoryRecordGovernance:
    if wire is None:
        return MemoryRecordGovernance()
    raw = wire.data_classification or DataClassification.INTERNAL.value
    try:
        classification = DataClassification(str(raw).lower())
    except ValueError:
        classification = DataClassification.INTERNAL
    return MemoryRecordGovernance(data_classification=classification)


def _lineage_from_legacy_wire(wire: LegacyMemoryRecordLineageWire | None) -> MemoryRecordLineage:
    if wire is None:
        return MemoryRecordLineage()
    return MemoryRecordLineage(
        supersedes_memory_id=wire.supersedes_memory_id,
        superseded_by_memory_id=wire.superseded_by_memory_id,
    )


def memory_entry_to_dict(entry: UserProfileMemoryEntry) -> Dict[str, Any]:
    return {
        "entry_id": entry.entry_id,
        "revision": entry.revision,
        "content": entry.content,
        "session_id": entry.session_id,
        "kind": entry.kind.value,
        "title": entry.title,
        "importance": entry.importance.value,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
        "valid_from": entry.valid_from,
        "valid_until": entry.valid_until,
        "provenance": _provenance_to_dict(entry.provenance),
        "trust": _trust_to_dict(entry.trust),
        "governance": _governance_to_dict(entry.governance),
        "lineage": _lineage_to_dict(entry.lineage),
        "evidence_refs": list(entry.evidence_refs),
        "metadata": dict(entry.metadata),
        "deleted": entry.deleted,
        "modified": entry.modified,
    }


def memory_entry_from_dict(payload: Dict[str, Any]) -> UserProfileMemoryEntry:
    provenance = _provenance_from_dict(payload.get("provenance") if isinstance(payload.get("provenance"), dict) else None)
    entry_id = str(payload.get("entry_id", "")).strip()
    if not entry_id:
        entry_id = uuid.uuid4().hex
    entry = UserProfileMemoryEntry(
        entry_id=entry_id,
        revision=int(payload.get("revision", 1)),
        content=str(payload.get("content", "")),
        session_id=payload.get("session_id"),
        kind=_memory_kind(str(payload.get("kind", MemoryKind.OTHER.value))),
        title=payload.get("title"),
        importance=_memory_importance(str(payload.get("importance", MemoryImportance.MEDIUM.value))),
        created_at=str(payload.get("created_at", "")),
        updated_at=payload.get("updated_at"),
        valid_from=payload.get("valid_from"),
        valid_until=payload.get("valid_until"),
        provenance=provenance,
        trust=_trust_from_dict(payload.get("trust") if isinstance(payload.get("trust"), dict) else None),
        governance=_governance_from_dict(
            payload.get("governance") if isinstance(payload.get("governance"), dict) else None
        ),
        lineage=_lineage_from_dict(payload.get("lineage") if isinstance(payload.get("lineage"), dict) else None),
        evidence_refs=tuple(str(item) for item in (payload.get("evidence_refs") or [])),
        metadata=dict(payload.get("metadata") or {}),
        deleted=bool(payload.get("deleted", False)),
        modified=bool(payload.get("modified", False)),
    )
    if provenance.source_type is MemoryRecordSourceType.UNKNOWN:
        legacy = entry.metadata.get("source")
        if legacy is not None:
            entry.provenance = MemoryProvenance(
                source_type=memory_record_source_from_legacy_string(str(legacy)),
                session_id=entry.session_id,
            )
    return apply_legacy_metadata_provenance(entry)


def user_profile_to_dict(profile: UserProfile) -> Dict[str, Any]:
    identity = profile.identity
    preferences = profile.preferences
    return {
        "schema_version": SCHEMA_USER_PROFILE_V1,
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


def _memory_entry_from_legacy_wire(entry: LegacyUserProfileMemoryEntryWire) -> UserProfileMemoryEntry:
    provenance = _provenance_from_legacy_wire(entry.provenance)
    entry_id = str(entry.entry_id or "").strip()
    if not entry_id:
        entry_id = uuid.uuid4().hex
    kind_value = entry.kind if entry.kind is not None else MemoryKind.OTHER.value
    importance_value = (
        entry.importance if entry.importance is not None else MemoryImportance.MEDIUM.value
    )
    memory_entry = UserProfileMemoryEntry(
        entry_id=entry_id,
        revision=entry.revision,
        content=entry.content,
        session_id=entry.session_id,
        kind=_memory_kind(str(kind_value)),
        title=entry.title,
        importance=_memory_importance(str(importance_value)),
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        valid_from=entry.valid_from,
        valid_until=entry.valid_until,
        provenance=provenance,
        trust=_trust_from_legacy_wire(entry.trust),
        governance=_governance_from_legacy_wire(entry.governance),
        lineage=_lineage_from_legacy_wire(entry.lineage),
        evidence_refs=tuple(str(item) for item in entry.evidence_refs),
        metadata=dict(entry.metadata),
        deleted=entry.deleted,
        modified=entry.modified,
    )
    if provenance.source_type is MemoryRecordSourceType.UNKNOWN:
        legacy = memory_entry.metadata.get("source")
        if legacy is not None:
            memory_entry.provenance = MemoryProvenance(
                source_type=memory_record_source_from_legacy_string(str(legacy)),
                session_id=memory_entry.session_id,
            )
    return apply_legacy_metadata_provenance(memory_entry)


def _user_profile_from_legacy_wire(legacy: LegacyUserProfileWire) -> UserProfile:
    identity_wire = legacy.identity
    preferences_wire = legacy.preferences
    identity = UserIdentity(
        user_id=identity_wire.user_id,
        display_name=identity_wire.display_name,
        role=identity_wire.role,
        domain_expertise=identity_wire.domain_expertise,
        language=identity_wire.language,
        locale=identity_wire.locale,
        timezone=identity_wire.timezone,
    )
    preferences = UserPreferences(
        preferred_language=preferences_wire.preferred_language,
        answer_length=preferences_wire.answer_length,
        tone=preferences_wire.tone,
        no_emojis_in_code=preferences_wire.no_emojis_in_code,
        no_emojis_in_docs=preferences_wire.no_emojis_in_docs,
        prefer_markdown=preferences_wire.prefer_markdown,
        prefer_code_blocks=preferences_wire.prefer_code_blocks,
        default_project_context=preferences_wire.default_project_context,
        extra=dict(preferences_wire.extra),
    )
    return UserProfile(
        identity=identity,
        preferences=preferences,
        system_instructions=legacy.system_instructions,
        memory_entries=[_memory_entry_from_legacy_wire(item) for item in legacy.memory_entries],
        version=legacy.version,
        entry_id=str(legacy.entry_id or identity.user_id),
        deleted=legacy.deleted,
        modified=legacy.modified,
    )


def deserialize_user_profile_payload(payload: Mapping[str, object]) -> UserProfile:
    schema_version = payload.get("schema_version")
    if schema_version == SCHEMA_USER_PROFILE_V1:
        return _user_profile_from_versioned_dict(dict(payload))
    if _is_legacy_user_profile_mapping(payload):
        legacy = LegacyUserProfileWire.model_validate(payload)
        return _user_profile_from_legacy_wire(legacy)
    raise ValueError(
        f"unsupported user profile schema version {schema_version!r}; "
        f"expected {SCHEMA_USER_PROFILE_V1!r} or {LEGACY_UNVERSIONED_PRE_V1}"
    )


def user_profile_from_dict(payload: Mapping[str, object]) -> UserProfile:
    return deserialize_user_profile_payload(payload)


def _user_profile_from_versioned_dict(payload: Dict[str, Any]) -> UserProfile:
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_USER_PROFILE_V1:
        raise ValueError(
            f"unsupported user profile schema version {schema_version!r}; "
            f"expected {SCHEMA_USER_PROFILE_V1!r}"
        )
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
