# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Runtime governance policy decisions (architecture §42.11).

Distinct from replay ``ExecutionPolicyEngine`` — use ``intergrax.runtime.policy.PolicyEngine`` facade.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

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

    def has_attested_policy_bundle_refs(self) -> bool:
        """True when bundle id, version, and digest are all non-empty."""
        return bool(
            self.policy_bundle_id.strip()
            and self.policy_bundle_version.strip()
            and self.policy_bundle_digest.strip()
        )
