# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Platform contract failure rule declarations (DS-E2E-14.3A)."""

from __future__ import annotations

from intergrax.decision_system.qualification.classification import (
    DecisionFailureClassificationRule,
)
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.rules.contracts import (
    DecisionFailureRuleSpec,
    resolve_platform_violation_boundary,
    rule_from_spec,
)
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureDiagnosticCode,
    DecisionFailureOwner,
    DecisionFailureReason,
    DecisionRetryability,
)


def _trace_not_finalized(observation: DecisionQualificationObservation) -> bool:
    return not observation.platform_contract.trace_finalized


def _wrong_execution_route(observation: DecisionQualificationObservation) -> bool:
    return observation.platform_contract.wrong_execution_route


def _strict_tool_contract_violation(
    observation: DecisionQualificationObservation,
) -> bool:
    return observation.platform_contract.strict_tool_contract_violation


def _tool_dispatch_contract_violation(
    observation: DecisionQualificationObservation,
) -> bool:
    return observation.platform_contract.tool_dispatch_contract_violation


def _invalid_phase_transition(observation: DecisionQualificationObservation) -> bool:
    return observation.platform_contract.invalid_phase_transition


def _completion_reconciliation_contract_violation(
    observation: DecisionQualificationObservation,
) -> bool:
    return observation.platform_contract.completion_reconciliation_contract_violation


def _terminal_acceptance_contract_violation(
    observation: DecisionQualificationObservation,
) -> bool:
    return observation.platform_contract.terminal_acceptance_contract_violation


_PLATFORM_RULE_SPECS: tuple[DecisionFailureRuleSpec, ...] = (
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.PLATFORM_CONTRACT,
        reason=DecisionFailureReason.TRACE_RUN_NOT_FINALIZED,
        owner=DecisionFailureOwner.EXECUTION_ENGINE,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_TRACE_NOT_FINALIZED,
        default_boundary=DecisionFailureBoundary.TRACE_FINALIZATION,
        predicate=_trace_not_finalized,
        boundary_resolver=resolve_platform_violation_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.PLATFORM_CONTRACT,
        reason=DecisionFailureReason.WRONG_EXECUTION_ROUTE,
        owner=DecisionFailureOwner.DECISION_SYSTEM,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_WRONG_EXECUTION_ROUTE,
        default_boundary=DecisionFailureBoundary.ROOT_CAPABILITY_ROUTING,
        predicate=_wrong_execution_route,
        boundary_resolver=resolve_platform_violation_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.PLATFORM_CONTRACT,
        reason=DecisionFailureReason.STRICT_TOOL_CONTRACT_VIOLATION,
        owner=DecisionFailureOwner.EXECUTION_ENGINE,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_STRICT_TOOL_VIOLATION,
        default_boundary=DecisionFailureBoundary.STRICT_TOOL_PROJECTION,
        predicate=_strict_tool_contract_violation,
        boundary_resolver=resolve_platform_violation_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.PLATFORM_CONTRACT,
        reason=DecisionFailureReason.TOOL_DISPATCH_CONTRACT_VIOLATION,
        owner=DecisionFailureOwner.EXECUTION_ENGINE,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_TOOL_DISPATCH_VIOLATION,
        default_boundary=DecisionFailureBoundary.TOOL_DISPATCH,
        predicate=_tool_dispatch_contract_violation,
        boundary_resolver=resolve_platform_violation_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.PLATFORM_CONTRACT,
        reason=DecisionFailureReason.INVALID_PHASE_TRANSITION,
        owner=DecisionFailureOwner.EXECUTION_ENGINE,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_INVALID_PHASE_TRANSITION,
        default_boundary=DecisionFailureBoundary.PHASE_VALIDATION,
        predicate=_invalid_phase_transition,
        boundary_resolver=resolve_platform_violation_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.PLATFORM_CONTRACT,
        reason=DecisionFailureReason.COMPLETION_RECONCILIATION_CONTRACT_VIOLATION,
        owner=DecisionFailureOwner.EXECUTION_ENGINE,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_COMPLETION_RECONCILIATION,
        default_boundary=DecisionFailureBoundary.COMPLETION_RECONCILIATION,
        predicate=_completion_reconciliation_contract_violation,
        boundary_resolver=resolve_platform_violation_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.PLATFORM_CONTRACT,
        reason=DecisionFailureReason.TERMINAL_ACCEPTANCE_CONTRACT_VIOLATION,
        owner=DecisionFailureOwner.EXECUTION_ENGINE,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_TERMINAL_ACCEPTANCE,
        default_boundary=DecisionFailureBoundary.TERMINAL_ACCEPTANCE,
        predicate=_terminal_acceptance_contract_violation,
        boundary_resolver=resolve_platform_violation_boundary,
    ),
)

PLATFORM_RULES: tuple[DecisionFailureClassificationRule, ...] = tuple(
    rule_from_spec(spec) for spec in _PLATFORM_RULE_SPECS
)
