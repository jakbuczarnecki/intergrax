# © Artur Czarnecki. All rights reserved.

"""Graph execution critic hooks — Phase CRIT-V-3.4 / CRIT-V-3.5."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.critic.contracts import (
    CriticLayer,
    CriticRequest,
    CriticScope,
    CriticVerdict,
    RubricSpec,
    build_critic_request,
)


def critic_completion_blocked(
    hooks: CriticGraphHooks,
    *,
    verdict: CriticVerdict,
    l1_client_configured: bool,
) -> bool:
    """Fail-closed when completion requires critic layers that are unavailable."""
    if not hooks.config.require_critic_on_completion:
        return False
    if not verdict.passed:
        return True
    if hooks.config.semantic_judge_enabled and not l1_client_configured:
        return True
    return False
from intergrax.runtime.critic.critic_orchestrator import CriticOrchestrator
from intergrax.runtime.critic.eval_tool_client import CriticEvalToolClient
from intergrax.runtime.critic.l0_gateway import L0Gateway
from intergrax.runtime.critic.l1_gateway import L1Gateway
from intergrax.runtime.critic.trace import CriticTraceEmitter
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine


@dataclass(frozen=True, slots=True)
class CriticHookConfig:
    """Tier-1 critic hook flags (mirrors ``CriticProfile`` scopes without Tier-3 import)."""

    verify_node_partial: bool = False
    verify_graph_final: bool = True
    semantic_judge_enabled: bool = False
    trajectory_eval_enabled: bool = False
    judge_threshold: float = 0.75
    default_rubric_ref: str | None = None
    require_critic_on_completion: bool = False


@dataclass(frozen=True, slots=True)
class CriticGraphHooks:
    """Resolved critic wiring for graph partial and final verification."""

    orchestrator: CriticOrchestrator
    config: CriticHookConfig

    @property
    def verify_node_partial(self) -> bool:
        return self.config.verify_node_partial

    @property
    def verify_graph_final(self) -> bool:
        return self.config.verify_graph_final


def build_critic_graph_hooks(
    *,
    config: CriticHookConfig,
    validation_engine: NexusValidationEngine | None = None,
    l1_client: CriticEvalToolClient | None = None,
) -> CriticGraphHooks | None:
    """Build graph hooks when at least one critic scope is enabled."""
    if not config.verify_node_partial and not config.verify_graph_final:
        return None
    orchestrator = CriticOrchestrator(
        l0_gateway=L0Gateway(engine=validation_engine or NexusValidationEngine()),
        l1_gateway=L1Gateway(tool_client=l1_client),
    )
    return CriticGraphHooks(orchestrator=orchestrator, config=config)


def enabled_layers_for_scope(config: CriticHookConfig, *, partial: bool) -> tuple[CriticLayer, ...]:
    layers: list[CriticLayer] = [CriticLayer.L0_DETERMINISTIC]
    if config.semantic_judge_enabled:
        layers.append(CriticLayer.L1_SEMANTIC)
    if config.trajectory_eval_enabled and not partial:
        layers.append(CriticLayer.L1_TRAJECTORY)
    return tuple(layers)


def critic_verdict_to_validation_result(verdict: CriticVerdict) -> ValidationResult:
    scores = [layer.score for layer in verdict.layers if layer.score is not None]
    confidence = sum(scores) / len(scores) if scores else None
    errors = list(verdict.failure_reasons)
    if not errors and not verdict.passed:
        errors = [error for layer in verdict.layers for error in layer.errors]
    warnings = [warning for layer in verdict.layers for warning in layer.warnings]
    return ValidationResult(
        valid=verdict.passed,
        errors=errors,
        warnings=warnings,
        confidence=confidence,
    )


def validate_node_with_critic(
    execution: AgentExecutionResult,
    *,
    contract: AgentContract,
    hooks: CriticGraphHooks,
    run_id: str,
    tenant_id: str,
    capability: str | None = None,
    plan_criteria: list[str] | None = None,
    rubric: RubricSpec | None = None,
    trace_emitter: CriticTraceEmitter | None = None,
    node_id: str | None = None,
) -> ValidationResult:
    validation, _ = validate_node_with_critic_detail(
        execution,
        contract=contract,
        hooks=hooks,
        run_id=run_id,
        tenant_id=tenant_id,
        capability=capability,
        plan_criteria=plan_criteria,
        rubric=rubric,
        trace_emitter=trace_emitter,
        node_id=node_id,
    )
    return validation


def validate_node_with_critic_detail(
    execution: AgentExecutionResult,
    *,
    contract: AgentContract,
    hooks: CriticGraphHooks,
    run_id: str,
    tenant_id: str,
    capability: str | None = None,
    plan_criteria: list[str] | None = None,
    rubric: RubricSpec | None = None,
    trace_emitter: CriticTraceEmitter | None = None,
    node_id: str | None = None,
) -> tuple[ValidationResult, CriticVerdict]:
    request = _build_graph_critic_request(
        execution=execution,
        contract=contract,
        hooks=hooks,
        run_id=run_id,
        tenant_id=tenant_id,
        capability=capability,
        plan_criteria=plan_criteria,
        rubric=rubric,
        partial=True,
    )
    verdict = hooks.orchestrator.verify_partial(request, contract=contract)
    if trace_emitter is not None:
        trace_emitter.emit_verdict(
            request,
            verdict,
            tenant_id=tenant_id,
            task_id=run_id,
            agent_id=contract.id,
            node_id=node_id,
        )
    return critic_verdict_to_validation_result(verdict), verdict


def validate_final_with_critic(
    execution: AgentExecutionResult,
    *,
    contract: AgentContract,
    hooks: CriticGraphHooks,
    run_id: str,
    tenant_id: str,
    capability: str | None = None,
    plan_criteria: list[str] | None = None,
    rubric: RubricSpec | None = None,
    trace_emitter: CriticTraceEmitter | None = None,
) -> ValidationResult:
    validation, _ = validate_final_with_critic_detail(
        execution,
        contract=contract,
        hooks=hooks,
        run_id=run_id,
        tenant_id=tenant_id,
        capability=capability,
        plan_criteria=plan_criteria,
        rubric=rubric,
        trace_emitter=trace_emitter,
    )
    return validation


def validate_final_with_critic_detail(
    execution: AgentExecutionResult,
    *,
    contract: AgentContract,
    hooks: CriticGraphHooks,
    run_id: str,
    tenant_id: str,
    capability: str | None = None,
    plan_criteria: list[str] | None = None,
    rubric: RubricSpec | None = None,
    trace_emitter: CriticTraceEmitter | None = None,
) -> tuple[ValidationResult, CriticVerdict]:
    request = _build_graph_critic_request(
        execution=execution,
        contract=contract,
        hooks=hooks,
        run_id=run_id,
        tenant_id=tenant_id,
        capability=capability,
        plan_criteria=plan_criteria,
        rubric=rubric,
        partial=False,
    )
    verdict = hooks.orchestrator.verify_final(request, contract=contract)
    if trace_emitter is not None:
        trace_emitter.emit_verdict(
            request,
            verdict,
            tenant_id=tenant_id,
            task_id=run_id,
            agent_id=contract.id,
        )
    validation = critic_verdict_to_validation_result(verdict)
    if critic_completion_blocked(
        hooks,
        verdict=verdict,
        l1_client_configured=hooks.orchestrator.l1_client_configured,
    ):
        validation = ValidationResult(
            valid=False,
            errors=["critic_completion_blocked"],
            warnings=list(validation.warnings),
            confidence=validation.confidence,
        )
    return validation, verdict


def _build_graph_critic_request(
    *,
    execution: AgentExecutionResult,
    contract: AgentContract,
    hooks: CriticGraphHooks,
    run_id: str,
    tenant_id: str,
    capability: str | None,
    plan_criteria: list[str] | None,
    rubric: RubricSpec | None,
    partial: bool,
) -> CriticRequest:
    resolved_rubric = rubric
    if resolved_rubric is None and hooks.config.default_rubric_ref:
        resolved_rubric = RubricSpec(
            rubric_id=hooks.config.default_rubric_ref,
            min_score=hooks.config.judge_threshold,
        )
    context: dict[str, object] = {
        "tenant_id": tenant_id,
        "contract": contract,
        "trajectory_min_score": hooks.config.judge_threshold,
    }
    if capability:
        context["capability"] = capability
    if plan_criteria:
        context["plan_criteria"] = plan_criteria
    return build_critic_request(
        scope=CriticScope.NODE_PARTIAL if partial else CriticScope.GRAPH_FINAL,
        run_id=run_id,
        agent_id=contract.id,
        enabled_layers=enabled_layers_for_scope(hooks.config, partial=partial),
        execution=execution,
        rubric=resolved_rubric,
        context=context,
    )
