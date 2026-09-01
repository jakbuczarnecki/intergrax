# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Per-role context visibility policy contracts (DS-DELIB-05).

Declarative allowlist of context channels each participant role may receive.
Visibility is not authorization — policies govern what context may be
materialized, not what tools or side effects may be executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from intergrax.contracts.decision_participants import (
    ParticipantConfiguration,
    ParticipantRoleId,
    validate_participant_role_id,
)

DeliberationContextId = NewType("DeliberationContextId", str)


def validate_deliberation_context_id(value: object) -> DeliberationContextId:
    """Validate an opaque logical context surface identifier."""
    if type(value) is not str:
        raise TypeError(
            f"DeliberationContextId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "DeliberationContextId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "DeliberationContextId must not contain leading or trailing whitespace",
        )
    return DeliberationContextId(value)


def _canonicalize_context_ids(
    context_ids: tuple[DeliberationContextId, ...],
) -> tuple[DeliberationContextId, ...]:
    normalized: list[DeliberationContextId] = []
    seen: set[DeliberationContextId] = set()
    for item in context_ids:
        context_id = validate_deliberation_context_id(item)
        if context_id in seen:
            raise ValueError(
                "visible_contexts must not contain duplicate DeliberationContextId",
            )
        seen.add(context_id)
        normalized.append(context_id)
    return tuple(sorted(normalized, key=str))


def _require_canonical_context_ids(
    context_ids: tuple[DeliberationContextId, ...],
) -> None:
    canonical = _canonicalize_context_ids(context_ids)
    if context_ids != canonical:
        raise ValueError(
            "visible_contexts must be in canonical order without duplicates",
        )


def _unique_canonical_role_ids(
    role_ids: tuple[ParticipantRoleId, ...],
    *,
    field_name: str,
) -> tuple[ParticipantRoleId, ...]:
    validated: set[ParticipantRoleId] = set()
    for item in role_ids:
        role_id = validate_participant_role_id(item)
        validated.add(role_id)
    return tuple(sorted(validated, key=str))


def _canonicalize_role_ids(
    role_ids: tuple[ParticipantRoleId, ...],
    *,
    field_name: str,
) -> tuple[ParticipantRoleId, ...]:
    normalized: list[ParticipantRoleId] = []
    seen: set[ParticipantRoleId] = set()
    for item in role_ids:
        role_id = validate_participant_role_id(item)
        if role_id in seen:
            raise ValueError(f"{field_name} must not contain duplicate role_id")
        seen.add(role_id)
        normalized.append(role_id)
    return tuple(sorted(normalized, key=str))


def _require_canonical_role_ids(
    role_ids: tuple[ParticipantRoleId, ...],
    *,
    field_name: str,
) -> None:
    canonical = _canonicalize_role_ids(role_ids, field_name=field_name)
    if role_ids != canonical:
        raise ValueError(f"{field_name} must be in canonical order without duplicates")


@dataclass(frozen=True, slots=True)
class ParticipantContextVisibilityPolicy:
    """Allowlist of context channels one role may receive during deliberation."""

    role_id: ParticipantRoleId
    visible_contexts: tuple[DeliberationContextId, ...]

    def __post_init__(self) -> None:
        validate_participant_role_id(self.role_id)
        _require_canonical_context_ids(self.visible_contexts)


def _canonicalize_policies(
    policies: tuple[ParticipantContextVisibilityPolicy, ...],
    *,
    known_role_ids: set[ParticipantRoleId],
    active_role_ids: set[ParticipantRoleId],
) -> tuple[ParticipantContextVisibilityPolicy, ...]:
    if not policies and active_role_ids:
        raise ValueError(
            "policies must cover every active participant role with explicit visibility",
        )
    normalized: list[ParticipantContextVisibilityPolicy] = []
    seen_roles: set[ParticipantRoleId] = set()
    for policy in policies:
        if type(policy) is not ParticipantContextVisibilityPolicy:
            raise TypeError("policies must contain ParticipantContextVisibilityPolicy")
        validated_policy = participant_context_visibility_policy(
            role_id=policy.role_id,
            visible_contexts=policy.visible_contexts,
        )
        if validated_policy.role_id in seen_roles:
            raise ValueError(
                "policies must not contain duplicate role_id: "
                f"{validated_policy.role_id!r}",
            )
        if validated_policy.role_id not in known_role_ids:
            raise ValueError(
                "ParticipantContextVisibilityPolicy.role_id must reference known role: "
                f"{validated_policy.role_id!r}",
            )
        seen_roles.add(validated_policy.role_id)
        normalized.append(validated_policy)
    missing = active_role_ids - seen_roles
    if missing:
        missing_sorted = sorted(missing, key=str)
        raise ValueError(
            "every active participant role must have explicit visibility policy; "
            f"missing: {missing_sorted!r}",
        )
    return tuple(sorted(normalized, key=lambda item: item.role_id))


def _require_canonical_policies(
    policies: tuple[ParticipantContextVisibilityPolicy, ...],
    *,
    known_role_ids: set[ParticipantRoleId],
    active_role_ids: set[ParticipantRoleId],
) -> None:
    canonical = _canonicalize_policies(
        policies,
        known_role_ids=known_role_ids,
        active_role_ids=active_role_ids,
    )
    if policies != canonical:
        raise ValueError("policies must be in canonical order without duplicates")


@dataclass(frozen=True, slots=True)
class ParticipantContextVisibilityConfiguration:
    """Immutable visibility configuration bound to a role universe.

    ``active_role_ids`` lists roles referenced by participant bindings.
    ``known_role_ids`` lists all configured role definitions. Policies may
  cover unused roles but must cover every active role explicitly.
    """

    active_role_ids: tuple[ParticipantRoleId, ...]
    known_role_ids: tuple[ParticipantRoleId, ...]
    policies: tuple[ParticipantContextVisibilityPolicy, ...]

    def __post_init__(self) -> None:
        canonical_known = _canonicalize_role_ids(
            self.known_role_ids,
            field_name="known_role_ids",
        )
        canonical_active = _canonicalize_role_ids(
            self.active_role_ids,
            field_name="active_role_ids",
        )
        _require_canonical_role_ids(self.known_role_ids, field_name="known_role_ids")
        _require_canonical_role_ids(self.active_role_ids, field_name="active_role_ids")
        known_set = set(canonical_known)
        active_set = set(canonical_active)
        if not active_set.issubset(known_set):
            missing = sorted(active_set - known_set, key=str)
            raise ValueError(
                "active_role_ids must be subset of known_role_ids; "
                f"unknown active roles: {missing!r}",
            )
        _require_canonical_policies(
            self.policies,
            known_role_ids=known_set,
            active_role_ids=active_set,
        )


def participant_context_visibility_policy(
    *,
    role_id: object,
    visible_contexts: tuple[DeliberationContextId, ...],
) -> ParticipantContextVisibilityPolicy:
    """Build one per-role visibility policy with canonical context ordering."""
    return ParticipantContextVisibilityPolicy(
        role_id=validate_participant_role_id(role_id),
        visible_contexts=_canonicalize_context_ids(visible_contexts),
    )


def is_context_visible(
    policy: ParticipantContextVisibilityPolicy,
    context_id: object,
) -> bool:
    """Return whether ``context_id`` is explicitly allowed by ``policy``."""
    if type(policy) is not ParticipantContextVisibilityPolicy:
        raise TypeError("policy must be ParticipantContextVisibilityPolicy")
    validated = validate_deliberation_context_id(context_id)
    return validated in policy.visible_contexts


def participant_context_visibility_configuration(
    *,
    participant_configuration: ParticipantConfiguration,
    policies: tuple[ParticipantContextVisibilityPolicy, ...],
) -> ParticipantContextVisibilityConfiguration:
    """Build visibility configuration validated against participant roles."""
    if type(participant_configuration) is not ParticipantConfiguration:
        raise TypeError("participant_configuration must be ParticipantConfiguration")
    known_role_ids = _canonicalize_role_ids(
        tuple(role.role_id for role in participant_configuration.roles),
        field_name="known_role_ids",
    )
    active_role_ids = _unique_canonical_role_ids(
        tuple(binding.role_id for binding in participant_configuration.participants),
        field_name="active_role_ids",
    )
    known_set = set(known_role_ids)
    active_set = set(active_role_ids)
    canonical_policies = _canonicalize_policies(
        policies,
        known_role_ids=known_set,
        active_role_ids=active_set,
    )
    return ParticipantContextVisibilityConfiguration(
        active_role_ids=active_role_ids,
        known_role_ids=known_role_ids,
        policies=canonical_policies,
    )
