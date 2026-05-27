# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Runtime governance policy decisions (architecture §42.11).

Distinct from ``intergrax.runtime.replay.policy.ExecutionPolicyEngine`` (eval/replay).
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
    audit_payload: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "policy_decision.v1"
