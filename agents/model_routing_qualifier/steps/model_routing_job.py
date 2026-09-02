# © Artur Czarnecki. All rights reserved.

"""LKW model-routing qualification step — production LLMRoutingEvaluator + real model call."""

from __future__ import annotations

from enum import Enum

from intergrax.agents.authoring.runtime_tool_helpers import exec_ctx_from_step, request_metadata
from intergrax.applications._shared.routing_evaluating_adapter import RoutingEvaluatingLLMAdapter
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.routing import LLMRoutingEvaluator, LLMRoutingProfile, RoutingContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from model_routing_qualifier.model_routing import (
    RoutingProfileCandidate,
    artifact_ref_for_profile,
    candidates_from_profiles,
)
from model_routing_qualifier.model_routing_functional_evidence import emit_model_routing_functional_evidence
from model_routing_qualifier.model_routing import Q4_INVOKE_FAIL_TASK_CLASS, Q4_PRIMARY_TASK_CLASS

MODEL_ROUTING_STEP_ID = "model_routing_qualifier_step"
_ROUTING_TASK_CLASS_KEY = "qualification_routing_task_class"
_ROUTING_BUDGET_RATIO_KEY = "qualification_routing_budget_ratio"
_FAILURE_LAYER_KEY = "qualification_failure_injection_layer"
_TASK_MESSAGE_KEY = "qualification_task_message"
_EXPECTED_PROFILE_REF_KEY = "qualification_expected_profile_ref"
_SYSTEM_PROMPT_KEY = "qualification_system_prompt"

_DEFAULT_TASK = "What is 17 + 25? Reply with only the numeric result."
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


def _routing_context_from_metadata(metadata: dict[str, object]) -> RoutingContext:
    task_class_raw = metadata.get(_ROUTING_TASK_CLASS_KEY)
    task_class = (
        str(task_class_raw).strip()
        if isinstance(task_class_raw, str) and str(task_class_raw).strip()
        else Q4_PRIMARY_TASK_CLASS
    )
    budget_raw = metadata.get(_ROUTING_BUDGET_RATIO_KEY)
    budget_ratio = 0.9
    if isinstance(budget_raw, (int, float)):
        budget_ratio = float(budget_raw)
    elif isinstance(budget_raw, str) and budget_raw.strip():
        budget_ratio = float(budget_raw.strip())
    tenant_raw = metadata.get("tenant_id")
    tenant_id = str(tenant_raw).strip() if isinstance(tenant_raw, str) and tenant_raw.strip() else "default"
    return RoutingContext(
        task_class=task_class,
        budget_remaining_ratio=budget_ratio,
        tenant_id=tenant_id,
        agent_id="model_routing_qualifier",
    )


def _provider_key(provider: object) -> str:
    if isinstance(provider, Enum):
        return str(provider.value)
    return str(provider)


def _failure_output(*, run_id: str, reason: str, **extra: object) -> dict[str, object]:
    answer = f"model_routing_qualifier: {reason}"
    summary = {
        "used": False,
        "reason": reason,
        "routing_context_summary": extra.get("routing_context_summary"),
        "candidate_profile_refs": extra.get("candidate_profile_refs"),
        "expected_profile_ref": extra.get("expected_profile_ref"),
        "selected_profile_ref": extra.get("selected_profile_ref"),
        "matched_rule_id": extra.get("matched_rule_id"),
        "routing_reason": extra.get("routing_reason"),
        "actual_adapter_provider": extra.get("actual_adapter_provider"),
        "actual_adapter_model": extra.get("actual_adapter_model"),
        "invocation_status": extra.get("invocation_status"),
        "raw_model_output": extra.get("raw_model_output"),
    }
    return {
        "summary": answer,
        "answer": answer,
        "run_id": run_id,
        "model_routing_summary": summary,
    }


def _evaluate_routing(
    routing_profile: LLMRoutingProfile,
    context: RoutingContext,
) -> tuple[str, str | None, str]:
    evaluation = LLMRoutingEvaluator().evaluate(routing_profile, context)
    selected_ref = artifact_ref_for_profile(evaluation.selected_profile)
    return selected_ref, evaluation.matched_rule_id, evaluation.routing_reason


async def run_model_routing_job(
    step_ctx: AgentStepContext,
    *,
    routing_profile: LLMRoutingProfile,
) -> dict[str, object]:
    exec_ctx = exec_ctx_from_step(step_ctx)
    metadata = request_metadata(exec_ctx, step_ctx)
    failure_layer_raw = metadata.get(_FAILURE_LAYER_KEY)
    failure_layer = str(failure_layer_raw).strip() if failure_layer_raw is not None else None
    task_message = str(metadata.get(_TASK_MESSAGE_KEY) or metadata.get("query") or _DEFAULT_TASK)
    expected_profile_ref = str(metadata.get(_EXPECTED_PROFILE_REF_KEY) or "").strip()
    routing_context = _routing_context_from_metadata(metadata)
    routing_context_summary = (
        f"task_class={routing_context.task_class};"
        f"budget_remaining_ratio={routing_context.budget_remaining_ratio}"
    )

    if exec_ctx is None:
        return _failure_output(
            run_id=step_ctx.run_id,
            reason="tool_gateway_not_available",
            routing_context_summary=routing_context_summary,
        )

    adapter = _resolve_llm_adapter(exec_ctx)
    if adapter is None:
        return _failure_output(
            run_id=step_ctx.run_id,
            reason="llm_adapter_not_available",
            routing_context_summary=routing_context_summary,
        )

    allowed_profiles = routing_profile.allowed_profiles or (routing_profile.default_profile,)
    candidates: tuple[RoutingProfileCandidate, ...] = candidates_from_profiles(allowed_profiles)
    candidate_refs = tuple(artifact_ref_for_profile(item.profile) for item in candidates)

    selected_ref, matched_rule_id, routing_reason = _evaluate_routing(routing_profile, routing_context)

    if isinstance(adapter, RoutingEvaluatingLLMAdapter):
        adapter.set_context_provider(lambda: _routing_context_from_metadata(metadata))

    system_prompt = _DEFAULT_SYSTEM_PROMPT
    if failure_layer == "final_answer_bias":
        system_prompt = _FINAL_ANSWER_BIAS_PROMPT
    override_raw = metadata.get(_SYSTEM_PROMPT_KEY)
    if isinstance(override_raw, str) and override_raw.strip():
        system_prompt = override_raw.strip()

    invoke_succeeded = False
    raw_model_output = ""
    try:
        response = adapter.generate_messages(
            [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=task_message),
            ],
            temperature=0.0,
            run_id=step_ctx.run_id,
        )
        raw_model_output = response.content.strip()
        invoke_succeeded = bool(raw_model_output)
    except Exception:
        invoke_succeeded = False

    actual_provider = _provider_key(adapter.provider)
    actual_model = adapter.model or ""

    emit_model_routing_functional_evidence(
        exec_ctx,
        metadata=metadata,
        candidates=candidates,
        selected_profile_ref=selected_ref,
        invoke_succeeded=invoke_succeeded,
        raw_model_output=raw_model_output,
    )

    answer = raw_model_output or f"model_routing_qualifier: invocation_failed for {selected_ref}"
    summary = {
        "used": invoke_succeeded,
        "reason": "model_routing_qualification_complete" if invoke_succeeded else "model_invocation_failed",
        "routing_context_summary": routing_context_summary,
        "candidate_profile_refs": list(candidate_refs),
        "expected_profile_ref": expected_profile_ref or None,
        "selected_profile_ref": selected_ref,
        "matched_rule_id": matched_rule_id,
        "routing_reason": routing_reason,
        "actual_adapter_provider": actual_provider,
        "actual_adapter_model": actual_model,
        "invocation_status": "success" if invoke_succeeded else "failed",
        "raw_model_output": raw_model_output[:200] if raw_model_output else None,
    }
    return {
        "summary": answer,
        "answer": answer,
        "run_id": step_ctx.run_id,
        "model_routing_summary": summary,
    }


__all__ = ["MODEL_ROUTING_STEP_ID", "run_model_routing_job"]
