# © Artur Czarnecki. All rights reserved.

"""LKW model-routing qualification step — authoritative production routing observation."""

from __future__ import annotations

from intergrax.knowledge.contracts.validation import JsonObject
from intergrax.agents.authoring.runtime_tool_helpers import exec_ctx_from_step, request_metadata
from intergrax.applications._shared.routing_evaluating_adapter import RoutingEvaluatingLLMAdapter
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.routing import LLMRoutingProfile
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from model_routing_qualifier.model_routing import (
    RoutingProfileCandidate,
    artifact_ref_for_profile,
    candidates_from_profiles,
)
from model_routing_qualifier.model_routing_functional_evidence import emit_model_routing_functional_evidence
from model_routing_qualifier.qualification_types import (
    ActualModelExecution,
    InvocationFailureKind,
    ModelRoutingQualificationResult,
    ObservedRoutingDecision,
    Q4QualificationRequest,
    _PROVIDER_BOUNDARY_EXCEPTIONS,
    classify_invocation_failure,
)
from model_routing_qualifier.routing_observation import (
    begin_routing_observation,
    end_routing_observation,
)

MODEL_ROUTING_STEP_ID = "model_routing_qualifier_step"

_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise arithmetic assistant. Answer with only the numeric result, no explanation."
)
_FINAL_ANSWER_BIAS_PROMPT = (
    "You are a math assistant. For every arithmetic question, answer with only the number 99, "
    "regardless of the correct result."
)


def _resolve_llm_adapter(exec_ctx) -> LLMAdapter | None:
    runtime_state = exec_ctx.metadata.get("runtime_state")
    if not isinstance(runtime_state, RuntimeState):
        return None
    return runtime_state.context.config.llm_adapter


def _provider_key(provider: object) -> str:
    from enum import Enum

    if isinstance(provider, Enum):
        return str(provider.value)
    return str(provider)


def _routing_context_summary(request: Q4QualificationRequest) -> str:
    return (
        f"task_class={request.routing_task_class};"
        f"budget_remaining_ratio={request.budget_ratio}"
    )


def _resolve_system_prompt(request: Q4QualificationRequest) -> str:
    if request.failure_layer == "final_answer_bias":
        return _FINAL_ANSWER_BIAS_PROMPT
    if request.system_prompt is not None:
        return request.system_prompt
    return _DEFAULT_SYSTEM_PROMPT


def _actual_execution(adapter: LLMAdapter) -> ActualModelExecution:
    wrapper_provider = _provider_key(adapter.provider)
    wrapper_model = adapter.model or ""
    inner = adapter
    if isinstance(adapter, RoutingEvaluatingLLMAdapter):
        inner = adapter.inner_adapter
    return ActualModelExecution(
        wrapper_provider=wrapper_provider,
        wrapper_model=wrapper_model,
        inner_provider=_provider_key(inner.provider),
        inner_model=inner.model or "",
    )


def _result_to_output(result: ModelRoutingQualificationResult) -> JsonObject:
    observed = result.observed_decision
    execution = result.actual_execution
    summary = {
        "used": result.used,
        "reason": result.reason,
        "routing_context_summary": result.routing_context_summary,
        "candidate_profile_refs": list(result.candidate_profile_refs),
        "expected_profile_ref": result.expected_profile_ref,
        "selected_profile_ref": observed.selected_profile_ref if observed is not None else None,
        "production_evaluation_selected_profile": (
            observed.selected_profile_ref if observed is not None else None
        ),
        "matched_rule_id": observed.matched_rule_id if observed is not None else None,
        "routing_reason": observed.routing_reason if observed is not None else None,
        "policy_route_hint": observed.policy_route_hint if observed is not None else None,
        "actual_adapter_provider": execution.wrapper_provider if execution is not None else None,
        "actual_adapter_model": execution.wrapper_model if execution is not None else None,
        "actual_inner_adapter_provider": execution.inner_provider if execution is not None else None,
        "actual_inner_adapter_model": execution.inner_model if execution is not None else None,
        "invocation_status": result.invocation_status,
        "invocation_failure_kind": result.invocation_failure_kind.value,
        "raw_model_output": result.raw_model_output[:200] if result.raw_model_output else None,
    }
    return {
        "summary": result.answer,
        "answer": result.answer,
        "run_id": result.run_id,
        "model_routing_summary": summary,
    }


def _failure_result(
    *,
    run_id: str,
    reason: str,
    request: Q4QualificationRequest | None = None,
    candidate_refs: tuple[str, ...] = (),
) -> JsonObject:
    routing_summary = _routing_context_summary(request) if request is not None else None
    result = ModelRoutingQualificationResult(
        used=False,
        reason=reason,
        routing_context_summary=routing_summary or "",
        candidate_profile_refs=candidate_refs,
        expected_profile_ref=request.expected_profile_ref if request is not None else None,
        observed_decision=None,
        actual_execution=None,
        invocation_status="failed",
        invocation_failure_kind=InvocationFailureKind.NONE,
        raw_model_output="",
        answer=f"model_routing_qualifier: {reason}",
        run_id=run_id,
    )
    return _result_to_output(result)


async def run_model_routing_job(
    step_ctx: AgentStepContext,
    *,
    routing_profile: LLMRoutingProfile,
) -> JsonObject:
    exec_ctx = exec_ctx_from_step(step_ctx)
    metadata = request_metadata(exec_ctx, step_ctx)
    request = Q4QualificationRequest.from_metadata(metadata)
    routing_context_summary = _routing_context_summary(request)

    if exec_ctx is None:
        return _failure_result(
            run_id=step_ctx.run_id,
            reason="tool_gateway_not_available",
            request=request,
        )

    adapter = _resolve_llm_adapter(exec_ctx)
    if adapter is None:
        return _failure_result(
            run_id=step_ctx.run_id,
            reason="llm_adapter_not_available",
            request=request,
        )

    if not isinstance(adapter, RoutingEvaluatingLLMAdapter):
        return _failure_result(
            run_id=step_ctx.run_id,
            reason="routing_evaluating_adapter_required",
            request=request,
        )

    tenant_raw = metadata.get("tenant_id")
    tenant_id = (
        str(tenant_raw).strip()
        if isinstance(tenant_raw, str) and str(tenant_raw).strip()
        else "default"
    )
    routing_context = request.routing_context(tenant_id)

    allowed_profiles = routing_profile.allowed_profiles or (routing_profile.default_profile,)
    candidates: tuple[RoutingProfileCandidate, ...] = candidates_from_profiles(allowed_profiles)
    candidate_refs = tuple(artifact_ref_for_profile(item.profile) for item in candidates)

    captured_decisions: list[ObservedRoutingDecision] = []
    observation_session = begin_routing_observation(
        adapter,
        context_provider=lambda: routing_context,
        captured=captured_decisions,
    )

    invoke_succeeded = False
    raw_model_output = ""
    failure_kind = InvocationFailureKind.NONE
    system_prompt = _resolve_system_prompt(request)

    try:
        try:
            response = adapter.generate_messages(
                [
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=request.task_message),
                ],
                temperature=0.0,
                run_id=step_ctx.run_id,
            )
            raw_model_output = response.content.strip()
            invoke_succeeded = bool(raw_model_output)
        except _PROVIDER_BOUNDARY_EXCEPTIONS as exc:
            failure_kind = classify_invocation_failure(exc)
            if failure_kind is InvocationFailureKind.UNEXPECTED_APPLICATION_EXCEPTION:
                raise
            invoke_succeeded = False
    finally:
        end_routing_observation(adapter, observation_session)

    observed_decision = captured_decisions[-1] if captured_decisions else None
    if observed_decision is None:
        return _failure_result(
            run_id=step_ctx.run_id,
            reason="production_routing_evaluation_not_observed",
            request=request,
            candidate_refs=candidate_refs,
        )

    actual_execution = _actual_execution(adapter)
    selected_ref = observed_decision.selected_profile_ref

    emit_model_routing_functional_evidence(
        exec_ctx,
        metadata=metadata,
        candidates=candidates,
        selected_profile_ref=selected_ref,
        invoke_succeeded=invoke_succeeded,
        raw_model_output=raw_model_output,
    )

    answer = raw_model_output or f"model_routing_qualifier: invocation_failed for {selected_ref}"
    result = ModelRoutingQualificationResult(
        used=invoke_succeeded,
        reason="model_routing_qualification_complete" if invoke_succeeded else "model_invocation_failed",
        routing_context_summary=routing_context_summary,
        candidate_profile_refs=candidate_refs,
        expected_profile_ref=request.expected_profile_ref or None,
        observed_decision=observed_decision,
        actual_execution=actual_execution,
        invocation_status="success" if invoke_succeeded else "failed",
        invocation_failure_kind=failure_kind,
        raw_model_output=raw_model_output,
        answer=answer,
        run_id=step_ctx.run_id,
    )
    return _result_to_output(result)


__all__ = ["MODEL_ROUTING_STEP_ID", "run_model_routing_job"]
