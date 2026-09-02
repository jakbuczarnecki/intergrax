# © Artur Czarnecki. All rights reserved.

"""Typed Q4 model-routing qualification inputs and authoritative routing snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from intergrax.llm_adapters.routing import RoutingContext, RoutingEvaluation
from model_routing_qualifier.model_routing import (
    Q4_PRIMARY_TASK_CLASS,
    artifact_ref_for_profile,
)

_ROUTING_TASK_CLASS_KEY = "qualification_routing_task_class"
_ROUTING_BUDGET_RATIO_KEY = "qualification_routing_budget_ratio"
_FAILURE_LAYER_KEY = "qualification_failure_injection_layer"
_TASK_MESSAGE_KEY = "qualification_task_message"
_EXPECTED_PROFILE_REF_KEY = "qualification_expected_profile_ref"
_SYSTEM_PROMPT_KEY = "qualification_system_prompt"

_DEFAULT_TASK = "What is 17 + 25? Reply with only the numeric result."


class InvocationFailureKind(str, Enum):
    NONE = "none"
    PROVIDER_MODEL_NOT_FOUND = "provider_model_not_found"
    PROVIDER_TRANSPORT_FAILURE = "provider_transport_failure"
    UNEXPECTED_APPLICATION_EXCEPTION = "unexpected_application_exception"


def _provider_key(provider: object) -> str:
    if isinstance(provider, Enum):
        return str(provider.value)
    return str(provider)


@dataclass(frozen=True, slots=True)
class ObservedRoutingDecision:
    selected_profile_ref: str
    matched_rule_id: str | None
    routing_reason: str
    policy_route_hint: str | None
    provider: str
    model: str

    @classmethod
    def from_evaluation(cls, evaluation: RoutingEvaluation) -> ObservedRoutingDecision:
        profile = evaluation.selected_profile
        return ObservedRoutingDecision(
            selected_profile_ref=artifact_ref_for_profile(profile),
            matched_rule_id=evaluation.matched_rule_id,
            routing_reason=evaluation.routing_reason,
            policy_route_hint=evaluation.policy_route_hint,
            provider=_provider_key(profile.provider),
            model=profile.model or "",
        )


@dataclass(frozen=True, slots=True)
class Q4QualificationRequest:
    routing_task_class: str
    budget_ratio: float
    failure_layer: str | None
    task_message: str
    expected_profile_ref: str
    system_prompt: str | None

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> Q4QualificationRequest:
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
        failure_layer_raw = metadata.get(_FAILURE_LAYER_KEY)
        failure_layer = (
            str(failure_layer_raw).strip() if failure_layer_raw is not None else None
        )
        task_message = str(metadata.get(_TASK_MESSAGE_KEY) or metadata.get("query") or _DEFAULT_TASK)
        expected_profile_ref = str(metadata.get(_EXPECTED_PROFILE_REF_KEY) or "").strip()
        system_prompt_raw = metadata.get(_SYSTEM_PROMPT_KEY)
        system_prompt = (
            system_prompt_raw.strip()
            if isinstance(system_prompt_raw, str) and system_prompt_raw.strip()
            else None
        )
        return Q4QualificationRequest(
            routing_task_class=task_class,
            budget_ratio=budget_ratio,
            failure_layer=failure_layer,
            task_message=task_message,
            expected_profile_ref=expected_profile_ref,
            system_prompt=system_prompt,
        )

    def routing_context(self, tenant_id: str) -> RoutingContext:
        return RoutingContext(
            task_class=self.routing_task_class,
            budget_remaining_ratio=self.budget_ratio,
            tenant_id=tenant_id,
            agent_id="model_routing_qualifier",
        )


@dataclass(frozen=True, slots=True)
class ActualModelExecution:
    wrapper_provider: str
    wrapper_model: str
    inner_provider: str
    inner_model: str


@dataclass(frozen=True, slots=True)
class ModelRoutingQualificationResult:
    used: bool
    reason: str
    routing_context_summary: str
    candidate_profile_refs: tuple[str, ...]
    expected_profile_ref: str | None
    observed_decision: ObservedRoutingDecision | None
    actual_execution: ActualModelExecution | None
    invocation_status: str
    invocation_failure_kind: InvocationFailureKind
    raw_model_output: str
    answer: str
    run_id: str


_PROVIDER_BOUNDARY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RuntimeError,
    ValueError,
    ConnectionError,
    TimeoutError,
    OSError,
)

_OLLAMA_RESPONSE_ERROR: type[BaseException] | None = None
try:
    from ollama import ResponseError as _ImportedOllamaResponseError

    _OLLAMA_RESPONSE_ERROR = _ImportedOllamaResponseError
    _PROVIDER_BOUNDARY_EXCEPTIONS = _PROVIDER_BOUNDARY_EXCEPTIONS + (_ImportedOllamaResponseError,)
except ImportError:
    pass


def classify_invocation_failure(exc: BaseException) -> InvocationFailureKind:
    if _OLLAMA_RESPONSE_ERROR is not None and isinstance(exc, _OLLAMA_RESPONSE_ERROR):
        return InvocationFailureKind.PROVIDER_MODEL_NOT_FOUND
    message = str(exc).lower()
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return InvocationFailureKind.PROVIDER_TRANSPORT_FAILURE
    if "not found" in message or "404" in message:
        return InvocationFailureKind.PROVIDER_MODEL_NOT_FOUND
    if "connection" in message or "timeout" in message or "refused" in message:
        return InvocationFailureKind.PROVIDER_TRANSPORT_FAILURE
    if isinstance(exc, (RuntimeError, ValueError)):
        return InvocationFailureKind.PROVIDER_TRANSPORT_FAILURE
    return InvocationFailureKind.UNEXPECTED_APPLICATION_EXCEPTION


__all__ = [
    "ActualModelExecution",
    "InvocationFailureKind",
    "ModelRoutingQualificationResult",
    "ObservedRoutingDecision",
    "Q4QualificationRequest",
    "_PROVIDER_BOUNDARY_EXCEPTIONS",
    "classify_invocation_failure",
]
