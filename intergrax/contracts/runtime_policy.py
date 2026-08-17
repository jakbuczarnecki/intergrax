# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Runtime governance policy decisions (architecture §42.11).

Distinct from replay ``ExecutionPolicyEngine`` — use ``intergrax.runtime.policy.PolicyEngine`` facade.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.agent_decision import AgentDecision


class PolicyAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"
    ESCALATE = "escalate"
    REQUIRE_HUMAN = "require_human"


class EnforcementLevel(str, Enum):
    ADVISORY = "advisory"
    MANDATORY = "mandatory"


class PolicyDecision(BaseModel):
    """Canonical runtime governance outcome.

    Bundle provenance is absent (all empty) or complete (id + version + sha256 digest).
    ``audit_payload`` is non-authoritative diagnostic/domain-specific data.
    Canonical provenance belongs to explicit fields and typed evaluation contracts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: PolicyAction
    reason: str = ""
    modified_decision: Optional[AgentDecision] = None
    enforcement_level: EnforcementLevel = EnforcementLevel.MANDATORY
    policy_rule_id: str = ""
    # Optional immutable pack identity (Execution Evidence / ADR-RUNTIME-POLICY-BUNDLE-001).
    # Empty when the evaluator does not stamp a digestable bundle (fail closed when attestation required).
    policy_bundle_id: str = ""
    policy_bundle_version: str = ""
    policy_bundle_digest: str = ""
    decision_id: str = ""
    audit_payload: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "policy_decision.v1"

    @field_validator(
        "policy_rule_id",
        "policy_bundle_id",
        "policy_bundle_version",
        "policy_bundle_digest",
        "decision_id",
    )
    @classmethod
    def _strip_provenance_identifiers(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _bundle_provenance_complete_or_absent(self) -> PolicyDecision:
        bundle_id = self.policy_bundle_id
        version = self.policy_bundle_version
        digest = self.policy_bundle_digest
        present = (bool(bundle_id), bool(version), bool(digest))
        if any(present) and not all(present):
            raise ValueError("policy_bundle_provenance_incomplete")
        if digest and not digest.startswith("sha256:"):
            raise ValueError("policy_bundle_digest_must_be_sha256")
        return self

    def has_attested_policy_bundle_refs(self) -> bool:
        """True when canonical bundle id, version, and digest are all non-empty."""
        return bool(
            self.policy_bundle_id
            and self.policy_bundle_version
            and self.policy_bundle_digest
        )
