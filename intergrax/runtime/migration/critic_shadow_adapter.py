# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Legacy Critic shadow adapter — migration-only, non-authoritative (DS-MIG parity).

Runs observational Critic verification without host control flow side effects.
Scheduled for deletion with ``intergrax/runtime/critic/**``.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.critic.contracts import (
    CriticLayer,
    CriticScope,
    CriticVerdict,
    RubricSpec,
    build_critic_request,
)
from intergrax.runtime.critic.critic_orchestrator import CriticOrchestrator
from intergrax.runtime.critic.critic_wiring import CriticHookConfig, enabled_layers_for_scope
from intergrax.runtime.critic.eval_tool_client import CriticEvalToolClient
from intergrax.runtime.critic.l0_gateway import L0Gateway
from intergrax.runtime.critic.l1_gateway import L1Gateway
from intergrax.runtime.decision_flow import DecisionFlowResult, DecisionFlowScope
from intergrax.runtime.migration.decision_critic_parity import (
    DecisionCriticParityObserver,
    DecisionCriticParityResult,
    build_parity_identity,
    compare_decision_critic_parity,
    critic_scope_for_parity_host_scope,
    parity_host_scope_from_flow_scope,
)
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine


@dataclass(frozen=True, slots=True)
class CriticShadowConfig:
    """Shadow-safe critic configuration — no L2, no completion blocking."""

    semantic_judge_enabled: bool = False
    trajectory_eval_enabled: bool = False
    judge_threshold: float = 0.75
    default_rubric_ref: str | None = None


@dataclass(frozen=True, slots=True)
class CriticShadowAdapter:
    """Observational Critic runner that cannot influence host authority."""

    orchestrator: CriticOrchestrator
    config: CriticShadowConfig

    @property
    def l1_client_configured(self) -> bool:
        return self.orchestrator.l1_client_configured

    def verify_execution(
        self,
        *,
        execution: AgentExecutionResult,
        contract: AgentContract,
        scope: CriticScope,
        run_id: str,
        tenant_id: str,
        capability: str | None = None,
        plan_criteria: tuple[str, ...] = (),
        rubric: RubricSpec | None = None,
        partial: bool = False,
        step_id: str | None = None,
        extra_context: dict[str, object] | None = None,
    ) -> CriticVerdict:
        """Run one shadow critic verification without host side effects."""
        if type(execution) is not AgentExecutionResult:
            raise TypeError("execution must be AgentExecutionResult")
        if type(contract) is not AgentContract:
            raise TypeError("contract must be AgentContract")
        resolved_rubric = rubric
        if resolved_rubric is None and self.config.default_rubric_ref:
            resolved_rubric = RubricSpec(
                rubric_id=self.config.default_rubric_ref,
                min_score=self.config.judge_threshold,
            )
        context: dict[str, object] = {
            "tenant_id": tenant_id,
            "contract": contract,
            "trajectory_min_score": self.config.judge_threshold,
        }
        if capability:
            context["capability"] = capability
        if plan_criteria:
            context["plan_criteria"] = list(plan_criteria)
        if step_id is not None:
            context["uaep_step_id"] = step_id
        if extra_context:
            context.update(extra_context)
        hook_config = _shadow_hook_config(self.config)
        request = build_critic_request(
            scope=scope,
            run_id=run_id,
            agent_id=contract.id,
            enabled_layers=enabled_layers_for_scope(hook_config, partial=partial),
            execution=execution,
            rubric=resolved_rubric,
            context=context,
        )
        if scope is CriticScope.GRAPH_FINAL:
            return self.orchestrator.verify_final(request, contract=contract)
        if scope is CriticScope.UAEP_STEP:
            return self.orchestrator.verify(request, contract=contract)
        if scope is CriticScope.NODE_PARTIAL:
            return self.orchestrator.verify_partial(request, contract=contract)
        return self.orchestrator.verify(request, contract=contract)


def _shadow_hook_config(config: CriticShadowConfig) -> CriticHookConfig:
    return CriticHookConfig(
        verify_node_partial=False,
        verify_graph_final=False,
        verify_uaep_step=False,
        semantic_judge_enabled=config.semantic_judge_enabled,
        trajectory_eval_enabled=config.trajectory_eval_enabled,
        judge_threshold=config.judge_threshold,
        default_rubric_ref=config.default_rubric_ref,
        l2_human_required=False,
        require_critic_on_completion=False,
    )


def build_critic_shadow_adapter(
    *,
    config: CriticShadowConfig | None = None,
    l1_client: CriticEvalToolClient | None = None,
    validation_engine: NexusValidationEngine | None = None,
) -> CriticShadowAdapter:
    """Build one observational critic shadow adapter."""
    resolved_config = config if config is not None else CriticShadowConfig()
    orchestrator = CriticOrchestrator(
        l0_gateway=L0Gateway(engine=validation_engine or NexusValidationEngine()),
        l1_gateway=L1Gateway(tool_client=l1_client),
    )
    return CriticShadowAdapter(orchestrator=orchestrator, config=resolved_config)


async def evaluate_critic_shadow_and_compare(
    *,
    shadow: CriticShadowAdapter,
    decision_result: DecisionFlowResult[AgentExecutionResult],
    execution: AgentExecutionResult,
    contract: AgentContract,
    flow_scope: DecisionFlowScope,
    task_id: str,
    run_id: str,
    attempt_id: str,
    tenant_id: str,
    subject: str,
    capability: str | None = None,
    plan_criteria: tuple[str, ...] = (),
    rubric: RubricSpec | None = None,
    step_id: str | None = None,
    extra_context: dict[str, object] | None = None,
    observer: DecisionCriticParityObserver | None = None,
    execution_id: str | None = None,
) -> DecisionCriticParityResult:
    """Evaluate critic shadow and compare against an already-fixed Decision result."""
    identity = build_parity_identity(
        flow_scope=flow_scope,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id=tenant_id,
        agent_id=contract.id,
        subject=subject,
        execution_id=execution_id,
        decision_result=decision_result,
    )
    host_scope = parity_host_scope_from_flow_scope(flow_scope)
    critic_scope = critic_scope_for_parity_host_scope(host_scope)
    partial = flow_scope is DecisionFlowScope.UAEP_STEP
    try:
        if not shadow.l1_client_configured and shadow.config.semantic_judge_enabled:
            parity_result = compare_decision_critic_parity(
                identity=identity,
                decision_result=decision_result,
                shadow_unavailable=True,
            )
        else:
            verdict = shadow.verify_execution(
                execution=execution,
                contract=contract,
                scope=critic_scope,
                run_id=run_id,
                tenant_id=tenant_id,
                capability=capability,
                plan_criteria=plan_criteria,
                rubric=rubric,
                partial=partial,
                step_id=step_id,
                extra_context=extra_context,
            )
            parity_result = compare_decision_critic_parity(
                identity=identity,
                decision_result=decision_result,
                critic_verdict=verdict,
            )
    except Exception as exc:
        parity_result = compare_decision_critic_parity(
            identity=identity,
            decision_result=decision_result,
            shadow_error=str(exc),
        )
    if observer is not None:
        observer.record(parity_result)
    return parity_result


async def observe_graph_final_parity(
    *,
    shadow: CriticShadowAdapter,
    decision_result: DecisionFlowResult[AgentExecutionResult],
    execution: AgentExecutionResult,
    contract: AgentContract,
    task_id: str,
    run_id: str,
    attempt_id: str,
    tenant_id: str,
    graph_id: str,
    capability: str | None = None,
    plan_criteria: tuple[str, ...] = (),
    observer: DecisionCriticParityObserver | None = None,
    execution_id: str | None = None,
) -> DecisionCriticParityResult:
    """Observe graph-final Decision/Critic parity without altering production outcome."""
    return await evaluate_critic_shadow_and_compare(
        shadow=shadow,
        decision_result=decision_result,
        execution=execution,
        contract=contract,
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id=tenant_id,
        subject=graph_id,
        capability=capability,
        plan_criteria=plan_criteria,
        observer=observer,
        execution_id=execution_id,
    )


async def observe_uaep_step_parity(
    *,
    shadow: CriticShadowAdapter,
    decision_result: DecisionFlowResult[AgentExecutionResult],
    execution: AgentExecutionResult,
    contract: AgentContract,
    task_id: str,
    run_id: str,
    attempt_id: str,
    tenant_id: str,
    step_id: str,
    observer: DecisionCriticParityObserver | None = None,
    execution_id: str | None = None,
    extra_context: dict[str, object] | None = None,
) -> DecisionCriticParityResult:
    """Observe UAEP step Decision/Critic parity without altering production outcome."""
    return await evaluate_critic_shadow_and_compare(
        shadow=shadow,
        decision_result=decision_result,
        execution=execution,
        contract=contract,
        flow_scope=DecisionFlowScope.UAEP_STEP,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id=tenant_id,
        subject=step_id,
        step_id=step_id,
        observer=observer,
        execution_id=execution_id,
        extra_context=extra_context,
    )


def shadow_enabled_layers(config: CriticShadowConfig, *, partial: bool) -> tuple[CriticLayer, ...]:
    """Return enabled critic layers for one shadow configuration."""
    return enabled_layers_for_scope(_shadow_hook_config(config), partial=partial)
