# © Artur Czarnecki. All rights reserved.

"""Typed evaluation contracts for declarative policy rules (BLOCK B)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.runtime.policy.rules.schema import PolicyRuleAction


class PolicyEnforcementMode(StrEnum):
    """Declarative policy enforcement posture for a host bundle."""

    AUDIT_ONLY = "audit_only"
    ENFORCE = "enforce"


@dataclass(frozen=True, slots=True)
class PolicyEvaluationContext:
    """Request-scoped evaluation context for declarative policy handlers."""

    tool_id: str
    tenant_id: str | None = None
    agent_id: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyRuleEvaluationOutcome:
    """Single rule evaluation result with explicit audit evidence."""

    rule_id: str
    action: PolicyRuleAction
    reasons: tuple[str, ...] = ()
    unknown_handler: bool = False
    handler_exception: bool = False
    unsupported_conditions: bool = False


@dataclass(frozen=True, slots=True)
class PolicyEnforcementDecision:
    """
    Aggregated declarative policy decision for one tool invocation.

    Precedence: DENY > REQUIRE_HITL > ALLOW (deterministic, order-independent).
  """

    action: PolicyRuleAction
    matched_rule_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    enforcement_mode: PolicyEnforcementMode
    enforced: bool
    would_deny: bool
    requires_hitl: bool
    unknown_handler_ids: tuple[str, ...] = ()
    provenance_digest: str | None = None

    @property
    def denied(self) -> bool:
        return self.action is PolicyRuleAction.DENY

    @property
    def should_block_execution(self) -> bool:
        if not self.enforced:
            return False
        return self.action in (PolicyRuleAction.DENY, PolicyRuleAction.REQUIRE_HITL)
