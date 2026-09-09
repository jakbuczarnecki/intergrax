# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Provider infrastructure failure rule declarations (DS-E2E-14.3A)."""

from __future__ import annotations

from intergrax.decision_system.qualification.classification import (
    DecisionFailureClassificationRule,
)
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.rules.contracts import (
    DecisionFailureRuleSpec,
    resolve_default_boundary,
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


def _rate_limit(observation: DecisionQualificationObservation) -> bool:
    return observation.provider.rate_limit


def _timeout(observation: DecisionQualificationObservation) -> bool:
    return observation.provider.timeout


def _network_failure(observation: DecisionQualificationObservation) -> bool:
    return observation.provider.network_failure


def _server_error(observation: DecisionQualificationObservation) -> bool:
    return observation.provider.server_error


def _protocol_error(observation: DecisionQualificationObservation) -> bool:
    return observation.provider.protocol_error


_PROVIDER_RULE_SPECS: tuple[DecisionFailureRuleSpec, ...] = (
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
        reason=DecisionFailureReason.RATE_LIMIT,
        owner=DecisionFailureOwner.PROVIDER,
        retryability=DecisionRetryability.RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.PROVIDER_RATE_LIMIT,
        default_boundary=DecisionFailureBoundary.PROVIDER_BINDING,
        predicate=_rate_limit,
        boundary_resolver=resolve_default_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
        reason=DecisionFailureReason.TIMEOUT,
        owner=DecisionFailureOwner.PROVIDER,
        retryability=DecisionRetryability.RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.PROVIDER_TIMEOUT,
        default_boundary=DecisionFailureBoundary.HOST_EXECUTION,
        predicate=_timeout,
        boundary_resolver=resolve_default_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
        reason=DecisionFailureReason.NETWORK_FAILURE,
        owner=DecisionFailureOwner.PROVIDER,
        retryability=DecisionRetryability.RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.PROVIDER_NETWORK_FAILURE,
        default_boundary=DecisionFailureBoundary.PROVIDER_BINDING,
        predicate=_network_failure,
        boundary_resolver=resolve_default_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
        reason=DecisionFailureReason.PROVIDER_SERVER_ERROR,
        owner=DecisionFailureOwner.PROVIDER,
        retryability=DecisionRetryability.RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.PROVIDER_SERVER_ERROR,
        default_boundary=DecisionFailureBoundary.HOST_EXECUTION,
        predicate=_server_error,
        boundary_resolver=resolve_default_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
        reason=DecisionFailureReason.PROVIDER_PROTOCOL_ERROR,
        owner=DecisionFailureOwner.PROVIDER,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.PROVIDER_PROTOCOL_ERROR,
        default_boundary=DecisionFailureBoundary.PROVIDER_BINDING,
        predicate=_protocol_error,
        boundary_resolver=resolve_default_boundary,
    ),
)

PROVIDER_RULES: tuple[DecisionFailureClassificationRule, ...] = tuple(
    rule_from_spec(spec) for spec in _PROVIDER_RULE_SPECS
)
