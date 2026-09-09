# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Environment failure rule declarations (DS-E2E-14.3A)."""

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


def _credential_unavailable(
    observation: DecisionQualificationObservation,
) -> bool:
    return observation.environment.credential_unavailable


def _provider_configuration_invalid(
    observation: DecisionQualificationObservation,
) -> bool:
    return observation.environment.provider_configuration_invalid


def _model_configuration_invalid(
    observation: DecisionQualificationObservation,
) -> bool:
    return observation.environment.model_configuration_invalid


def _qualification_disabled(
    observation: DecisionQualificationObservation,
) -> bool:
    return observation.environment.qualification_disabled


_ENVIRONMENT_RULE_SPECS: tuple[DecisionFailureRuleSpec, ...] = (
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.ENVIRONMENT,
        reason=DecisionFailureReason.CREDENTIAL_UNAVAILABLE,
        owner=DecisionFailureOwner.ENVIRONMENT,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.ENVIRONMENT_CREDENTIAL_UNAVAILABLE,
        default_boundary=DecisionFailureBoundary.ENVIRONMENT_RESOLUTION,
        predicate=_credential_unavailable,
        boundary_resolver=resolve_default_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.ENVIRONMENT,
        reason=DecisionFailureReason.PROVIDER_CONFIGURATION_INVALID,
        owner=DecisionFailureOwner.ENVIRONMENT,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.ENVIRONMENT_PROVIDER_CONFIGURATION,
        default_boundary=DecisionFailureBoundary.PROVIDER_BINDING,
        predicate=_provider_configuration_invalid,
        boundary_resolver=resolve_default_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.ENVIRONMENT,
        reason=DecisionFailureReason.MODEL_CONFIGURATION_INVALID,
        owner=DecisionFailureOwner.ENVIRONMENT,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.ENVIRONMENT_MODEL_CONFIGURATION,
        default_boundary=DecisionFailureBoundary.PROVIDER_BINDING,
        predicate=_model_configuration_invalid,
        boundary_resolver=resolve_default_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.ENVIRONMENT,
        reason=DecisionFailureReason.QUALIFICATION_DISABLED,
        owner=DecisionFailureOwner.ENVIRONMENT,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.ENVIRONMENT_QUALIFICATION_DISABLED,
        default_boundary=DecisionFailureBoundary.ENVIRONMENT_RESOLUTION,
        predicate=_qualification_disabled,
        boundary_resolver=resolve_default_boundary,
    ),
)

ENVIRONMENT_RULES: tuple[DecisionFailureClassificationRule, ...] = tuple(
    rule_from_spec(spec) for spec in _ENVIRONMENT_RULE_SPECS
)
