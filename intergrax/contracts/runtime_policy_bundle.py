# © Artur Czarnecki. All rights reserved.

"""Immutable, digestable runtime policy pack (Execution Evidence).

Distinct from the live wiring object
``intergrax.runtime.policy.policy_bundle.RuntimePolicyBundle`` (tool access /
budgets / plan-loop engines). This contract is provider-neutral, immutable, and
safe for canonical digests and host attestation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.runtime.attestation.canonical_json import stable_payload_hash

SCHEMA_RUNTIME_POLICY_BUNDLE_V1: Final = "runtime_policy_bundle.v1"
_NON_EMPTY = Field(min_length=1)


class PolicyBundleRule(BaseModel):
    """Single ordered rule entry inside an immutable pack."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = _NON_EMPTY
    description: str = ""
    effect: str = ""

    @field_validator("rule_id")
    @classmethod
    def _strip_rule_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("rule_id must be non-empty")
        return normalized


class ImmutableRuntimePolicyBundle(BaseModel):
    """Attested-ready policy pack identity.

    Schema: ``runtime_policy_bundle.v1``. Rule ordering is significant for the
    digest (JSON array order). Must not embed provider transport payloads or
    mutable runtime objects.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runtime_policy_bundle.v1"] = SCHEMA_RUNTIME_POLICY_BUNDLE_V1
    bundle_id: str = _NON_EMPTY
    version: str = _NON_EMPTY
    rules: tuple[PolicyBundleRule, ...] = ()
    issued_at: datetime
    canonical_digest: str = ""

    @field_validator("bundle_id", "version")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    def digest_payload(self) -> dict[str, Any]:
        """Canonical payload used for digest (excludes ``canonical_digest``)."""
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "version": self.version,
            "rules": [rule.model_dump(mode="json") for rule in self.rules],
            "issued_at": _normalize_timestamp(self.issued_at),
        }

    def compute_digest(self) -> str:
        return stable_payload_hash(self.digest_payload())

    def with_canonical_digest(self) -> ImmutableRuntimePolicyBundle:
        digest = self.compute_digest()
        if self.canonical_digest and self.canonical_digest != digest:
            raise ValueError("canonical_digest does not match recomputed digest")
        return self.model_copy(update={"canonical_digest": digest})


def build_immutable_runtime_policy_bundle(
    *,
    bundle_id: str,
    version: str,
    rules: Sequence[PolicyBundleRule | Mapping[str, Any]],
    issued_at: datetime,
) -> ImmutableRuntimePolicyBundle:
    """Build an immutable pack and stamp ``canonical_digest``."""
    normalized_rules: list[PolicyBundleRule] = []
    for rule in rules:
        if isinstance(rule, PolicyBundleRule):
            normalized_rules.append(rule)
        else:
            normalized_rules.append(PolicyBundleRule.model_validate(rule))
    return ImmutableRuntimePolicyBundle(
        bundle_id=bundle_id,
        version=version,
        rules=tuple(normalized_rules),
        issued_at=issued_at,
        canonical_digest="",
    ).with_canonical_digest()


def _normalize_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("issued_at must be timezone-aware")
    return value.isoformat()
