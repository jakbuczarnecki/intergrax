# © Artur Czarnecki. All rights reserved.

"""Canonical JSON serialization for durable Collaborative Work records."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    CollaborativeOperationPolicyProfile,
    CollaborativePolicyRule,
    PrincipalAuthorityGrant,
    WorkspaceMembership,
)


def _encode_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _decode_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def workspace_membership_to_json(record: WorkspaceMembership) -> str:
    return record.model_dump_json()


def workspace_membership_from_json(payload: str) -> WorkspaceMembership:
    return WorkspaceMembership.model_validate_json(payload)


def authority_delegation_to_json(record: AuthorityDelegation) -> str:
    return record.model_dump_json()


def authority_delegation_from_json(payload: str) -> AuthorityDelegation:
    return AuthorityDelegation.model_validate_json(payload)


def principal_authority_grant_to_json(record: PrincipalAuthorityGrant) -> str:
    return record.model_dump_json()


def principal_authority_grant_from_json(payload: str) -> PrincipalAuthorityGrant:
    return PrincipalAuthorityGrant.model_validate_json(payload)


def collaborative_policy_rule_to_json(record: CollaborativePolicyRule) -> str:
    return record.model_dump_json()


def collaborative_policy_rule_from_json(payload: str) -> CollaborativePolicyRule:
    return CollaborativePolicyRule.model_validate_json(payload)


def operation_policy_profile_to_json(record: CollaborativeOperationPolicyProfile) -> str:
    return record.model_dump_json()


def operation_policy_profile_from_json(payload: str) -> CollaborativeOperationPolicyProfile:
    return CollaborativeOperationPolicyProfile.model_validate_json(payload)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
