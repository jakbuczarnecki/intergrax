# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Canonical Decision qualification failure taxonomy (DS-E2E-14.3)."""

from __future__ import annotations

from enum import StrEnum


class DecisionFailureCategory(StrEnum):
    PLATFORM_CONTRACT = "platform_contract"
    MODEL_BEHAVIOR = "model_behavior"
    EVALUATOR_SEMANTICS = "evaluator_semantics"
    PROVIDER_INFRASTRUCTURE = "provider_infrastructure"
    ENVIRONMENT = "environment"
    OBSERVABILITY_GAP = "observability_gap"
    UNCLASSIFIED = "unclassified"


class DecisionFailureOwner(StrEnum):
    DECISION_SYSTEM = "decision_system"
    MODEL = "model"
    EVALUATOR = "evaluator"
    PROVIDER = "provider"
    ENVIRONMENT = "environment"
    EXECUTION_ENGINE = "execution_engine"
    OBSERVABILITY = "observability"


class DecisionFailureBoundary(StrEnum):
    ENVIRONMENT_RESOLUTION = "environment_resolution"
    PROVIDER_BINDING = "provider_binding"
    ROOT_CAPABILITY_ROUTING = "root_capability_routing"
    HOST_EXECUTION = "host_execution"
    PLANNER_PROTOCOL = "planner_protocol"
    STRICT_TOOL_PROJECTION = "strict_tool_projection"
    TOOL_DISPATCH = "tool_dispatch"
    EVIDENCE_LIFECYCLE = "evidence_lifecycle"
    REASONING = "reasoning"
    PHASE_VALIDATION = "phase_validation"
    COMPLETION_RECONCILIATION = "completion_reconciliation"
    TRACE_FINALIZATION = "trace_finalization"
    TERMINAL_ACCEPTANCE = "terminal_acceptance"
    EVALUATOR = "evaluator"


BOUNDARY_ORDER: tuple[DecisionFailureBoundary, ...] = (
    DecisionFailureBoundary.ENVIRONMENT_RESOLUTION,
    DecisionFailureBoundary.PROVIDER_BINDING,
    DecisionFailureBoundary.ROOT_CAPABILITY_ROUTING,
    DecisionFailureBoundary.HOST_EXECUTION,
    DecisionFailureBoundary.PLANNER_PROTOCOL,
    DecisionFailureBoundary.STRICT_TOOL_PROJECTION,
    DecisionFailureBoundary.TOOL_DISPATCH,
    DecisionFailureBoundary.EVIDENCE_LIFECYCLE,
    DecisionFailureBoundary.REASONING,
    DecisionFailureBoundary.PHASE_VALIDATION,
    DecisionFailureBoundary.COMPLETION_RECONCILIATION,
    DecisionFailureBoundary.TRACE_FINALIZATION,
    DecisionFailureBoundary.TERMINAL_ACCEPTANCE,
    DecisionFailureBoundary.EVALUATOR,
)

_BOUNDARY_RANK: dict[DecisionFailureBoundary, int] = {
    boundary: index for index, boundary in enumerate(BOUNDARY_ORDER)
}


def boundary_rank(boundary: DecisionFailureBoundary) -> int:
    return _BOUNDARY_RANK[boundary]


def earliest_boundary(
    *boundaries: DecisionFailureBoundary,
) -> DecisionFailureBoundary:
    return min(boundaries, key=boundary_rank)


class DecisionFailureReason(StrEnum):
    INSUFFICIENT_EVIDENCE_GATHERING = "insufficient_evidence_gathering"
    TOOL_USE_DEFICIENCY = "tool_use_deficiency"
    EPISTEMIC_CONTRADICTION = "epistemic_contradiction"
    UNSUPPORTED_COMPLETION = "unsupported_completion"
    PREMATURE_COMPLETION = "premature_completion"
    WRONG_EXECUTION_ROUTE = "wrong_execution_route"
    TRACE_RUN_NOT_FINALIZED = "trace_run_not_finalized"
    STRICT_TOOL_CONTRACT_VIOLATION = "strict_tool_contract_violation"
    TOOL_DISPATCH_CONTRACT_VIOLATION = "tool_dispatch_contract_violation"
    INVALID_PHASE_TRANSITION = "invalid_phase_transition"
    COMPLETION_RECONCILIATION_CONTRACT_VIOLATION = (
        "completion_reconciliation_contract_violation"
    )
    TERMINAL_ACCEPTANCE_CONTRACT_VIOLATION = "terminal_acceptance_contract_violation"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK_FAILURE = "network_failure"
    PROVIDER_SERVER_ERROR = "provider_server_error"
    PROVIDER_PROTOCOL_ERROR = "provider_protocol_error"
    CREDENTIAL_UNAVAILABLE = "credential_unavailable"
    PROVIDER_CONFIGURATION_INVALID = "provider_configuration_invalid"
    MODEL_CONFIGURATION_INVALID = "model_configuration_invalid"
    QUALIFICATION_DISABLED = "qualification_disabled"
    FALSE_NEGATIVE = "false_negative"
    FALSE_POSITIVE = "false_positive"
    CRITERION_SEMANTICS_INVALID = "criterion_semantics_invalid"
    EVALUATOR_CONTRACT_ERROR = "evaluator_contract_error"
    MISSING_REQUIRED_SIGNAL = "missing_required_signal"
    INCOMPLETE_TRACE = "incomplete_trace"
    AMBIGUOUS_FAILURE_BOUNDARY = "ambiguous_failure_boundary"
    UNCLASSIFIED = "unclassified"


class DecisionFailureDiagnosticCode(StrEnum):
    MODEL_INSUFFICIENT_EVIDENCE = "decision.model.insufficient_evidence_gathering"
    MODEL_TOOL_USE_DEFICIENCY = "decision.model.tool_use_deficiency"
    MODEL_EPISTEMIC_CONTRADICTION = "decision.model.epistemic_contradiction"
    MODEL_UNSUPPORTED_COMPLETION = "decision.model.unsupported_completion"
    MODEL_PREMATURE_COMPLETION = "decision.model.premature_completion"
    PLATFORM_WRONG_EXECUTION_ROUTE = "decision.platform.wrong_execution_route"
    PLATFORM_TRACE_NOT_FINALIZED = "decision.platform.trace_not_finalized"
    PLATFORM_STRICT_TOOL_VIOLATION = "decision.platform.strict_tool_contract_violation"
    PLATFORM_TOOL_DISPATCH_VIOLATION = "decision.platform.tool_dispatch_contract_violation"
    PLATFORM_INVALID_PHASE_TRANSITION = "decision.platform.invalid_phase_transition"
    PLATFORM_COMPLETION_RECONCILIATION = (
        "decision.platform.completion_reconciliation_contract_violation"
    )
    PLATFORM_TERMINAL_ACCEPTANCE = "decision.platform.terminal_acceptance_contract_violation"
    PROVIDER_RATE_LIMIT = "decision.provider.rate_limit"
    PROVIDER_TIMEOUT = "decision.provider.timeout"
    PROVIDER_NETWORK_FAILURE = "decision.provider.network_failure"
    PROVIDER_SERVER_ERROR = "decision.provider.provider_server_error"
    PROVIDER_PROTOCOL_ERROR = "decision.provider.provider_protocol_error"
    ENVIRONMENT_CREDENTIAL_UNAVAILABLE = "decision.environment.credential_unavailable"
    ENVIRONMENT_PROVIDER_CONFIGURATION = (
        "decision.environment.provider_configuration_invalid"
    )
    ENVIRONMENT_MODEL_CONFIGURATION = "decision.environment.model_configuration_invalid"
    ENVIRONMENT_QUALIFICATION_DISABLED = "decision.environment.qualification_disabled"
    EVALUATOR_FALSE_NEGATIVE = "decision.evaluator.false_negative"
    EVALUATOR_FALSE_POSITIVE = "decision.evaluator.false_positive"
    EVALUATOR_CRITERION_SEMANTICS = "decision.evaluator.criterion_semantics_invalid"
    EVALUATOR_CONTRACT_ERROR = "decision.evaluator.evaluator_contract_error"
    OBSERVABILITY_MISSING_SIGNAL = "decision.observability.missing_required_signal"
    OBSERVABILITY_INCOMPLETE_TRACE = "decision.observability.incomplete_trace"
    OBSERVABILITY_AMBIGUOUS_BOUNDARY = "decision.observability.ambiguous_failure_boundary"
    UNCLASSIFIED = "decision.unclassified"


class DecisionRetryability(StrEnum):
    NON_RETRIABLE = "non_retriable"
    RETRIABLE = "retriable"
    UNKNOWN = "unknown"


CATEGORY_PRECEDENCE: tuple[DecisionFailureCategory, ...] = (
    DecisionFailureCategory.ENVIRONMENT,
    DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
    DecisionFailureCategory.PLATFORM_CONTRACT,
    DecisionFailureCategory.MODEL_BEHAVIOR,
    DecisionFailureCategory.EVALUATOR_SEMANTICS,
    DecisionFailureCategory.OBSERVABILITY_GAP,
    DecisionFailureCategory.UNCLASSIFIED,
)
