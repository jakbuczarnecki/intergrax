# © Artur Czarnecki. All rights reserved.

"""Bundle-backed evaluated policy decision (PC-1).

Binds a ``PolicyDecision`` to the concrete immutable pack and request that
produced it. Pack identity must be set at evaluation time — never stamped
after the fact.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle
from intergrax.runtime.attestation.canonical_json import stable_payload_hash

SCHEMA_EVALUATED_POLICY_DECISION_V1: Final = "evaluated_policy_decision.v1"
_NON_EMPTY = Field(min_length=1)


class EvaluatedPolicyDecision(BaseModel):
    """Immutable snapshot of a decision derived from a concrete policy pack."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["evaluated_policy_decision.v1"] = (
        SCHEMA_EVALUATED_POLICY_DECISION_V1
    )
    decision: PolicyDecision
    bundle_id: str = _NON_EMPTY
    bundle_version: str = _NON_EMPTY
    bundle_digest: str = _NON_EMPTY
    matched_rule_id: str = _NON_EMPTY
    evaluated_at: datetime
    request_digest: str = _NON_EMPTY

    @field_validator(
        "bundle_id",
        "bundle_version",
        "bundle_digest",
        "matched_rule_id",
        "request_digest",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _bind_decision_to_bundle(self) -> EvaluatedPolicyDecision:
        d = self.decision
        if d.policy_bundle_id != self.bundle_id:
            raise ValueError("decision_bundle_id_mismatch")
        if d.policy_bundle_version != self.bundle_version:
            raise ValueError("decision_bundle_version_mismatch")
        if d.policy_bundle_digest != self.bundle_digest:
            raise ValueError("decision_bundle_digest_mismatch")
        if d.policy_rule_id != self.matched_rule_id:
            raise ValueError("decision_rule_id_mismatch")
        if not self.bundle_digest.startswith("sha256:"):
            raise ValueError("bundle_digest_must_be_sha256")
        if not self.request_digest.startswith("sha256:"):
            raise ValueError("request_digest_must_be_sha256")
        return self

    def assert_consistent_with_bundle(
        self,
        bundle: ImmutableRuntimePolicyBundle,
    ) -> None:
        """Fail closed when decision does not match the evaluated pack body."""
        recomputed = bundle.compute_digest()
        if bundle.canonical_digest and bundle.canonical_digest != recomputed:
            raise ValueError("bundle_canonical_digest_mismatch")
        if self.bundle_digest != recomputed:
            raise ValueError("evaluated_bundle_digest_mismatch")
        if self.bundle_id != bundle.bundle_id:
            raise ValueError("evaluated_bundle_id_mismatch")
        if self.bundle_version != bundle.version:
            raise ValueError("evaluated_bundle_version_mismatch")
        matched = next(
            (r for r in bundle.rules if r.rule_id == self.matched_rule_id),
            None,
        )
        if matched is None:
            raise ValueError("matched_rule_absent_from_bundle")
        if matched.effect.strip():
            try:
                expected = PolicyAction(matched.effect.strip().lower())
            except ValueError as exc:
                raise ValueError("matched_rule_effect_invalid") from exc
            if self.decision.action is not expected:
                raise ValueError("decision_action_mismatch_with_rule")


def request_digest_for_payload(payload: dict[str, Any]) -> str:
    """Canonical digest helper for policy request payloads."""
    return stable_payload_hash(payload)
