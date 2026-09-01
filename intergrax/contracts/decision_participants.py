# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Configurable participant roles and inference profile bindings (DS-DELIB-04).

User-defined opaque role identifiers with semantic instructions and logical
inference profile references. Decision System does not construct adapters or
interpret role names — Execution resolves InferenceProfileId to adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from intergrax.runtime.execution.inference_profile import (
    InferenceProfileId,
    validate_inference_profile_id,
)

ParticipantRoleId = NewType("ParticipantRoleId", str)
ParticipantId = NewType("ParticipantId", str)


def validate_participant_role_id(value: object) -> ParticipantRoleId:
    """Validate a user-defined opaque role identifier."""
    if type(value) is not str:
        raise TypeError(
            f"ParticipantRoleId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "ParticipantRoleId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "ParticipantRoleId must not contain leading or trailing whitespace",
        )
    return ParticipantRoleId(value)


def validate_participant_id(value: object) -> ParticipantId:
    """Validate a participant identity independent of role identity."""
    if type(value) is not str:
        raise TypeError(
            f"ParticipantId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "ParticipantId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "ParticipantId must not contain leading or trailing whitespace",
        )
    return ParticipantId(value)


def _validate_canonical_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{label} must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    return value


@dataclass(frozen=True, slots=True)
class ParticipantRoleDefinition:
    """Semantic role instruction supplied by the host application."""

    role_id: ParticipantRoleId
    instruction: str

    def __post_init__(self) -> None:
        validate_participant_role_id(self.role_id)
        _validate_canonical_string(
            self.instruction,
            "ParticipantRoleDefinition.instruction",
        )


@dataclass(frozen=True, slots=True)
class ParticipantBinding:
    """Maps one participant identity to a role and logical inference profile."""

    participant_id: ParticipantId
    role_id: ParticipantRoleId
    inference_profile_id: InferenceProfileId

    def __post_init__(self) -> None:
        validate_participant_id(self.participant_id)
        validate_participant_role_id(self.role_id)
        validate_inference_profile_id(self.inference_profile_id)


def _canonicalize_roles(
    roles: tuple[ParticipantRoleDefinition, ...],
) -> tuple[ParticipantRoleDefinition, ...]:
    if not roles:
        raise ValueError("roles must be non-empty")
    normalized: list[ParticipantRoleDefinition] = []
    seen: set[ParticipantRoleId] = set()
    for role in roles:
        if type(role) is not ParticipantRoleDefinition:
            raise TypeError("roles must contain ParticipantRoleDefinition")
        validated_role = ParticipantRoleDefinition(
            role_id=role.role_id,
            instruction=role.instruction,
        )
        if validated_role.role_id in seen:
            raise ValueError(
                f"roles must not contain duplicate role_id: {validated_role.role_id!r}",
            )
        seen.add(validated_role.role_id)
        normalized.append(validated_role)
    return tuple(sorted(normalized, key=lambda item: item.role_id))


def _canonicalize_participants(
    participants: tuple[ParticipantBinding, ...],
    *,
    known_role_ids: set[ParticipantRoleId],
) -> tuple[ParticipantBinding, ...]:
    if not participants:
        raise ValueError("participants must be non-empty")
    normalized: list[ParticipantBinding] = []
    seen: set[ParticipantId] = set()
    for participant in participants:
        if type(participant) is not ParticipantBinding:
            raise TypeError("participants must contain ParticipantBinding")
        validated_participant = ParticipantBinding(
            participant_id=participant.participant_id,
            role_id=participant.role_id,
            inference_profile_id=participant.inference_profile_id,
        )
        if validated_participant.participant_id in seen:
            raise ValueError(
                "participants must not contain duplicate participant_id: "
                f"{validated_participant.participant_id!r}",
            )
        if validated_participant.role_id not in known_role_ids:
            raise ValueError(
                "ParticipantBinding.role_id must reference known RoleDefinition: "
                f"{validated_participant.role_id!r}",
            )
        seen.add(validated_participant.participant_id)
        normalized.append(validated_participant)
    return tuple(sorted(normalized, key=lambda item: item.participant_id))


def _require_canonical_roles(
    roles: tuple[ParticipantRoleDefinition, ...],
) -> None:
    canonical = _canonicalize_roles(roles)
    if roles != canonical:
        raise ValueError("roles must be in canonical order without duplicates")


def _require_canonical_participants(
    participants: tuple[ParticipantBinding, ...],
    *,
    known_role_ids: set[ParticipantRoleId],
) -> None:
    canonical = _canonicalize_participants(
        participants,
        known_role_ids=known_role_ids,
    )
    if participants != canonical:
        raise ValueError(
            "participants must be in canonical order without duplicates",
        )


@dataclass(frozen=True, slots=True)
class ParticipantConfiguration:
    """Immutable aggregate of role definitions and participant bindings."""

    roles: tuple[ParticipantRoleDefinition, ...]
    participants: tuple[ParticipantBinding, ...]

    def __post_init__(self) -> None:
        canonical_roles = _canonicalize_roles(self.roles)
        known_role_ids = {role.role_id for role in canonical_roles}
        _require_canonical_roles(self.roles)
        _require_canonical_participants(
            self.participants,
            known_role_ids=known_role_ids,
        )


def participant_role_definition(
    *,
    role_id: object,
    instruction: str,
) -> ParticipantRoleDefinition:
    """Build one role definition with validated identifiers and instruction."""
    return ParticipantRoleDefinition(
        role_id=validate_participant_role_id(role_id),
        instruction=instruction,
    )


def participant_binding(
    *,
    participant_id: object,
    role_id: object,
    inference_profile_id: object,
) -> ParticipantBinding:
    """Build one participant binding with validated identifiers."""
    return ParticipantBinding(
        participant_id=validate_participant_id(participant_id),
        role_id=validate_participant_role_id(role_id),
        inference_profile_id=validate_inference_profile_id(inference_profile_id),
    )


def participant_configuration(
    *,
    roles: tuple[ParticipantRoleDefinition, ...],
    participants: tuple[ParticipantBinding, ...],
) -> ParticipantConfiguration:
    """Build participant configuration with canonical ordering throughout."""
    canonical_roles = _canonicalize_roles(roles)
    known_role_ids = {role.role_id for role in canonical_roles}
    return ParticipantConfiguration(
        roles=canonical_roles,
        participants=_canonicalize_participants(
            participants,
            known_role_ids=known_role_ids,
        ),
    )
