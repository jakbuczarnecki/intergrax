# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution interrupt handling and policy resolution (architecture §42.8, §42.11)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.contracts.agent_decision import (
    AgentDecision,
    AgentDecisionType,
    HumanRequest,
    HumanRequestUrgency,
    human_request_fields_from_payload,
)
from intergrax.contracts.execution_interrupt import ExecutionInterrupt, InterruptType
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

if TYPE_CHECKING:
    from intergrax.runtime.policy.policy_engine import PolicyEngine


class GovernanceResolution(BaseModel):
    """Outcome of evaluating an agent decision or interrupt against runtime policy."""

    policy_decision: PolicyDecision
    agent_decision: AgentDecision
    interrupt: Optional[ExecutionInterrupt] = None
    human_request: Optional[HumanRequest] = None
    schema_version: str = "governance_resolution.v1"

    @property
    def should_pause(self) -> bool:
        if self.agent_decision.type == AgentDecisionType.REQUEST_HUMAN:
            return True
        if self.policy_decision.action == PolicyAction.REQUIRE_HUMAN:
            return True
        if self.interrupt is not None and self.interrupt.blocking:
            return self.policy_decision.action != PolicyAction.ALLOW
        return False

    @property
    def should_fail(self) -> bool:
        return self.agent_decision.type in {
            AgentDecisionType.FAIL,
            AgentDecisionType.CANCEL,
        }


class ExecutionInterruptHandler:
    """Maps ``AgentDecision`` / ``ExecutionInterrupt`` to policy-backed governance outcomes."""

    def __init__(
        self,
        policy_engine: PolicyEngine | RuntimePolicyEngine | None = None,
    ) -> None:
        from intergrax.runtime.policy.policy_engine import coerce_policy_engine

        self._policy = coerce_policy_engine(policy_engine)

    @property
    def policy_engine(self) -> PolicyEngine:
        return self._policy

    def resolve_decision(
        self,
        decision: AgentDecision,
        *,
        task_id: str,
        run_id: str,
        agent_id: str,
        step_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GovernanceResolution:
        interrupt: Optional[ExecutionInterrupt] = None
        policy_ctx = dict(context or {})

        if decision.type == AgentDecisionType.INTERRUPT:
            interrupt = self._interrupt_from_decision(
                decision,
                task_id=task_id,
                run_id=run_id,
                agent_id=agent_id,
                step_id=step_id,
            )
            policy = self._policy.evaluate_interrupt(interrupt)
            if interrupt.blocking:
                policy_ctx.setdefault("has_unresolved_critical_interrupt", True)
        else:
            policy = self._policy.evaluate_decision(decision, context=policy_ctx)

        human_request = decision.human_request
        if human_request is None and (
            decision.type == AgentDecisionType.REQUEST_HUMAN
            or policy.action == PolicyAction.REQUIRE_HUMAN
        ):
            human_request = HumanRequest(
                request_id=f"hr_{uuid4().hex[:12]}",
                prompt=decision.reason or "Human approval required",
                options=["approve", "reject"],
                context_artifacts=list(decision.payload.get("context_artifacts", [])),
                **human_request_fields_from_payload(decision.payload),
            )

        return GovernanceResolution(
            policy_decision=policy,
            agent_decision=decision,
            interrupt=interrupt,
            human_request=human_request,
        )

    def resolve_interrupt(self, interrupt: ExecutionInterrupt) -> GovernanceResolution:
        policy = self._policy.evaluate_interrupt(interrupt)
        decision_type = interrupt.recommended_action
        human_request: Optional[HumanRequest] = None
        if policy.action == PolicyAction.REQUIRE_HUMAN:
            human_request = self._human_request_for_interrupt(interrupt)
        return GovernanceResolution(
            policy_decision=policy,
            agent_decision=AgentDecision(
                type=decision_type,
                reason=f"interrupt:{interrupt.interrupt_type.value}",
                interrupt_id=interrupt.interrupt_id,
            ),
            interrupt=interrupt,
            human_request=human_request,
        )

    @staticmethod
    def _human_request_for_interrupt(interrupt: ExecutionInterrupt) -> HumanRequest:
        options = ["approve", "reject", "escalate"]
        urgency = HumanRequestUrgency.NORMAL
        timeout_seconds: Optional[int] = None
        default_on_timeout: Optional[AgentDecisionType] = None
        if interrupt.interrupt_type == InterruptType.SAFETY_VIOLATION:
            urgency = HumanRequestUrgency.CRITICAL
            timeout_seconds = 1800
            default_on_timeout = AgentDecisionType.ESCALATE
        return HumanRequest(
            request_id=f"hr_{uuid4().hex[:12]}",
            prompt=f"Interrupt requires human review: {interrupt.interrupt_type.value}",
            options=options,
            urgency=urgency,
            timeout_seconds=timeout_seconds,
            default_on_timeout=default_on_timeout,
        )

    @staticmethod
    def _interrupt_from_decision(
        decision: AgentDecision,
        *,
        task_id: str,
        run_id: str,
        agent_id: str,
        step_id: Optional[str],
    ) -> ExecutionInterrupt:
        raw_type = decision.payload.get("interrupt_type")
        interrupt_type = InterruptType.HUMAN_JUDGMENT_REQUIRED
        if raw_type:
            try:
                interrupt_type = InterruptType(str(raw_type))
            except ValueError:
                interrupt_type = InterruptType.HUMAN_JUDGMENT_REQUIRED

        blocking = bool(decision.payload.get("blocking", True))
        return ExecutionInterrupt(
            interrupt_id=decision.interrupt_id or f"int_{uuid4().hex[:12]}",
            interrupt_type=interrupt_type,
            source_agent_id=agent_id,
            source_step_id=step_id,
            task_id=task_id,
            run_id=run_id,
            blocking=blocking,
            recommended_action=decision.payload.get("recommended_action", AgentDecisionType.REQUEST_HUMAN),
            metadata=dict(decision.payload),
        )
