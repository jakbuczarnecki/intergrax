# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution interrupt handling and policy resolution (architecture §42.8, §42.11)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel

from intergrax.contracts.agent_decision import (
    AgentDecision,
    AgentDecisionType,
    HumanRequest,
    HumanRequestUrgency,
    human_request_fields_from_payload,
)
from intergrax.contracts.declarative_hitl import DeclarativeHitlPendingApproval
from intergrax.contracts.execution_interrupt import ExecutionInterrupt, InterruptType
from intergrax.contracts.execution_identity import (
    RunId,
    TaskId,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernedExecutionEvaluationPoint,
)
from intergrax.contracts.runtime_policy import (
    EnforcementLevel,
    PolicyAction,
    PolicyDecision,
)
from intergrax.contracts.runtime_policy_context import AgentDecisionPolicyContext
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

if TYPE_CHECKING:
    from intergrax.runtime.governance.governance_evidence_recorder import (
        GovernanceEvidenceRecorder,
    )
    from intergrax.runtime.policy.policy_engine import PolicyEngine

INTERRUPT_COUNT_KEY = "interrupt_count"


class GovernanceResolution(BaseModel):
    """Outcome of evaluating an agent decision or interrupt against runtime policy."""

    policy_decision: PolicyDecision
    agent_decision: AgentDecision
    interrupt: Optional[ExecutionInterrupt] = None
    human_request: Optional[HumanRequest] = None
    declarative_hitl_pending: Optional[DeclarativeHitlPendingApproval] = None
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
        if self.agent_decision.type is AgentDecisionType.MODIFY_PLAN:
            from intergrax.contracts.agent_handoff import handoff_from_decision

            if handoff_from_decision(self.agent_decision) is None:
                return self.policy_decision.action is not PolicyAction.ALLOW
        return self.agent_decision.type in {
            AgentDecisionType.FAIL,
            AgentDecisionType.CANCEL,
        }

    @property
    def should_block_execution(self) -> bool:
        """Mandatory policy outcomes that must not proceed to decision execution."""
        action = self.policy_decision.action
        if action is PolicyAction.DENY:
            return True
        if action is PolicyAction.ESCALATE:
            return self.policy_decision.enforcement_level is EnforcementLevel.MANDATORY
        if self.interrupt is not None and self.interrupt.blocking:
            return action is not PolicyAction.ALLOW
        return False


class ExecutionInterruptHandler:
    """Maps ``AgentDecision`` / ``ExecutionInterrupt`` to policy-backed governance outcomes."""

    def __init__(
        self,
        policy_engine: PolicyEngine | RuntimePolicyEngine | None = None,
        *,
        allow_dynamic_replan: bool = False,
        governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
    ) -> None:
        from intergrax.runtime.policy.policy_engine import coerce_policy_engine

        self._policy = coerce_policy_engine(policy_engine)
        self._allow_dynamic_replan = allow_dynamic_replan
        self._governance_evidence_recorder = governance_evidence_recorder

    @property
    def policy_engine(self) -> PolicyEngine:
        return self._policy

    def _record_resolution_evidence(
        self,
        *,
        decision: AgentDecision,
        policy: PolicyDecision,
        interrupt: Optional[ExecutionInterrupt],
        task_id: str,
        run_id: str,
        agent_id: str,
        step_id: Optional[str],
    ) -> None:
        recorder = self._governance_evidence_recorder
        if recorder is None or recorder.persistence is None:
            return
        from intergrax.runtime.governance.governance_policy_decision_evidence_recording import (
            record_governance_policy_decision_evidence_for_active_identity,
        )

        evaluation_point = GovernedExecutionEvaluationPoint.AGENT_DECISION
        if interrupt is not None or decision.type == AgentDecisionType.INTERRUPT:
            evaluation_point = GovernedExecutionEvaluationPoint.INTERRUPT
        resolved_task_id: TaskId | None = None
        resolved_run_id: RunId | None = None
        try:
            resolved_task_id = validate_task_id(task_id)
        except (TypeError, ValueError):
            resolved_task_id = None
        try:
            resolved_run_id = validate_run_id(run_id)
        except (TypeError, ValueError):
            resolved_run_id = None
        try:
            record_governance_policy_decision_evidence_for_active_identity(
                recorder,
                evaluation_point=evaluation_point,
                decision=policy,
                action=f"agent_decision:{decision.type.value}",
                resource_type="agent_decision",
                resource_scope=agent_id,
                digest_payload={
                    "task_id": task_id,
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "step_id": step_id or "",
                    "decision_type": decision.type.value,
                    "policy_rule_id": policy.policy_rule_id or "",
                },
                idempotency_prefix=evaluation_point.value,
                task_id=resolved_task_id,
                run_id=resolved_run_id,
            )
        except RuntimeError:
            return

    def resolve_decision(
        self,
        decision: AgentDecision,
        *,
        task_id: str,
        run_id: str,
        agent_id: str,
        step_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        decision_policy_context: AgentDecisionPolicyContext | None = None,
    ) -> GovernanceResolution:
        interrupt: Optional[ExecutionInterrupt] = None
        policy_ctx = dict(context or {})
        decision_ctx = decision_policy_context or AgentDecisionPolicyContext()

        if decision.type == AgentDecisionType.INTERRUPT:
            budget_decision = self._interrupt_budget_decision(policy_ctx)
            if budget_decision is not None:
                policy = budget_decision
            else:
                interrupt = self._interrupt_from_decision(
                    decision,
                    task_id=task_id,
                    run_id=run_id,
                    agent_id=agent_id,
                    step_id=step_id,
                )
                policy = self._policy.evaluate_interrupt(interrupt)
                self._increment_interrupt_count(policy_ctx)
                if interrupt.blocking:
                    policy_ctx.setdefault("has_unresolved_critical_interrupt", True)
        elif decision.type is AgentDecisionType.MODIFY_PLAN:
            from intergrax.contracts.agent_handoff import handoff_from_decision

            handoff = handoff_from_decision(decision)
            if handoff is None:
                if self._allow_dynamic_replan and (
                    policy_ctx.get("nexus_replan_boundary")
                    or policy_ctx.get("engine_replan_boundary")
                ):
                    policy = PolicyDecision(
                        action=PolicyAction.ALLOW,
                        reason="nexus_replan_allowed",
                        policy_rule_id="orchestration.allow_dynamic_replan",
                    )
                else:
                    policy = PolicyDecision(
                        action=PolicyAction.DENY,
                        reason="MODIFY_PLAN_NOT_SUPPORTED",
                        policy_rule_id="modify_plan_not_supported",
                    )
            else:
                policy = self._policy.evaluate_decision(decision, context=decision_ctx)
        else:
            policy = self._policy.evaluate_decision(decision, context=decision_ctx)

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

        self._record_resolution_evidence(
            decision=decision,
            policy=policy,
            interrupt=interrupt,
            task_id=task_id,
            run_id=run_id,
            agent_id=agent_id,
            step_id=step_id,
        )
        return GovernanceResolution(
            policy_decision=policy,
            agent_decision=decision,
            interrupt=interrupt,
            human_request=human_request,
        )

    def resolve_interrupt(
        self,
        interrupt: ExecutionInterrupt,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> GovernanceResolution:
        policy_ctx = dict(context or {})
        budget_decision = self._interrupt_budget_decision(policy_ctx)
        if budget_decision is not None:
            policy = budget_decision
        else:
            policy = self._policy.evaluate_interrupt(interrupt)
            self._increment_interrupt_count(policy_ctx)
        decision_type = interrupt.recommended_action
        human_request: Optional[HumanRequest] = None
        if policy.action == PolicyAction.REQUIRE_HUMAN:
            human_request = self._human_request_for_interrupt(interrupt)
        agent_decision = AgentDecision(
            type=decision_type,
            reason=f"interrupt:{interrupt.interrupt_type.value}",
            interrupt_id=interrupt.interrupt_id,
        )
        self._record_resolution_evidence(
            decision=agent_decision,
            policy=policy,
            interrupt=interrupt,
            task_id=interrupt.task_id,
            run_id=interrupt.run_id,
            agent_id=interrupt.source_agent_id,
            step_id=interrupt.source_step_id,
        )
        return GovernanceResolution(
            policy_decision=policy,
            agent_decision=agent_decision,
            interrupt=interrupt,
            human_request=human_request,
        )

    @staticmethod
    def _interrupt_budget_decision(policy_ctx: Dict[str, Any]) -> PolicyDecision | None:
        count = int(policy_ctx.get(INTERRUPT_COUNT_KEY, 0))
        max_interrupts = int(policy_ctx.get("max_interrupts_per_run", 16))
        if count >= max_interrupts:
            return PolicyDecision(
                action=PolicyAction.ESCALATE,
                reason="interrupt_budget_exceeded",
                enforcement_level=EnforcementLevel.MANDATORY,
                policy_rule_id="default.interrupt_budget",
                audit_payload={
                    "interrupt_count": count,
                    "max_interrupts_per_run": max_interrupts,
                },
            )
        return None

    @staticmethod
    def _increment_interrupt_count(policy_ctx: Dict[str, Any]) -> None:
        policy_ctx[INTERRUPT_COUNT_KEY] = (
            int(policy_ctx.get(INTERRUPT_COUNT_KEY, 0)) + 1
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
            recommended_action=decision.payload.get(
                "recommended_action", AgentDecisionType.REQUEST_HUMAN
            ),
            metadata=dict(decision.payload),
        )
