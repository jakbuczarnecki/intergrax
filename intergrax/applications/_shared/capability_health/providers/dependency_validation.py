# © Artur Czarnecki. All rights reserved.

"""P1.3 dependency validation → capability health facts (P1.5)."""

from __future__ import annotations

from intergrax.applications.contracts.capability_dependency import (
    CapabilityDependencyAvailabilityStatus,
    CapabilityDependencyRequirement,
)
from intergrax.applications.contracts.capability_dependency.validation import (
    CapabilityDependencyValidationResult,
)
from intergrax.applications.contracts.capability_health import (
    CapabilityHealthConditionKind,
    CapabilityHealthFact,
    CapabilityHealthFactStatus,
    CapabilityHealthProjectionContext,
    CapabilityHealthProvider,
    CapabilityHealthReason,
)

_PROVIDER_ID = "dependency_validation"
_SOURCE_PROVENANCE = "dependency_validation"


def _map_dependency_status(
    status: CapabilityDependencyAvailabilityStatus,
) -> CapabilityHealthFactStatus:
    if status is CapabilityDependencyAvailabilityStatus.AVAILABLE:
        return CapabilityHealthFactStatus.SATISFIED
    if status is CapabilityDependencyAvailabilityStatus.UNAVAILABLE:
        return CapabilityHealthFactStatus.UNSATISFIED
    return CapabilityHealthFactStatus.UNKNOWN


def dependency_validation_health_facts(
    *,
    capability_key: str,
    validation: CapabilityDependencyValidationResult,
    provider_id: str = _PROVIDER_ID,
    source_provenance: str = _SOURCE_PROVENANCE,
    scope_application_id: str | None = None,
    scope_tenant_id: str | None = None,
) -> tuple[CapabilityHealthFact, ...]:
    """Map P1.3 validation evidence into health facts for one capability."""
    facts: list[CapabilityHealthFact] = []

    for failure in validation.required_failures:
        if failure.owner.canonical_key != capability_key:
            continue
        facts.append(
            CapabilityHealthFact(
                capability=failure.owner,
                source=source_provenance,
                condition_kind=CapabilityHealthConditionKind.DEPENDENCY_REQUIRED,
                condition_ref=failure.dependency.canonical_key,
                scope_application_id=scope_application_id,
                scope_tenant_id=scope_tenant_id,
                status=_map_dependency_status(failure.status),
                blocking=True,
                reason=CapabilityHealthReason(
                    reason_code="dependency.required.unsatisfied",
                    source=failure.source_domain,
                    subject_ref=failure.dependency.canonical_key,
                    detail=failure.reason,
                ),
                provider_id=provider_id,
            ),
        )

    for degradation in validation.optional_degradations:
        if degradation.owner.canonical_key != capability_key:
            continue
        facts.append(
            CapabilityHealthFact(
                capability=degradation.owner,
                source=source_provenance,
                condition_kind=CapabilityHealthConditionKind.DEPENDENCY_OPTIONAL,
                condition_ref=degradation.dependency.canonical_key,
                scope_application_id=scope_application_id,
                scope_tenant_id=scope_tenant_id,
                status=_map_dependency_status(degradation.status),
                blocking=False,
                reason=CapabilityHealthReason(
                    reason_code="dependency.optional.degraded",
                    source=degradation.source_domain,
                    subject_ref=degradation.dependency.canonical_key,
                    detail=degradation.reason,
                ),
                provider_id=provider_id,
            ),
        )

    outcome = next(
        (item for item in validation.outcomes if item.owner.canonical_key == capability_key),
        None,
    )
    if outcome is not None and outcome.available and not outcome.degraded:
        for evaluation in outcome.evaluations:
            facts.append(
                CapabilityHealthFact(
                    capability=outcome.owner,
                    source=source_provenance,
                    condition_kind=(
                        CapabilityHealthConditionKind.DEPENDENCY_REQUIRED
                        if evaluation.dependency.requirement
                        is CapabilityDependencyRequirement.REQUIRED
                        else CapabilityHealthConditionKind.DEPENDENCY_OPTIONAL
                    ),
                    condition_ref=evaluation.dependency.dependency.canonical_key,
                    scope_application_id=scope_application_id,
                    scope_tenant_id=scope_tenant_id,
                    status=CapabilityHealthFactStatus.SATISFIED,
                    blocking=(
                        evaluation.dependency.requirement
                        is CapabilityDependencyRequirement.REQUIRED
                    ),
                    reason=CapabilityHealthReason(
                        reason_code="dependency.satisfied",
                        source=source_provenance,
                        subject_ref=evaluation.dependency.dependency.canonical_key,
                        detail=evaluation.reason,
                    ),
                    provider_id=provider_id,
                ),
            )

    return tuple(
        sorted(facts, key=lambda item: (item.condition_ref, item.condition_kind.value)),
    )


class DependencyValidationHealthProvider:
    """Reuse P1.3 validation evidence — no second dependency validator."""

    @property
    def provider_id(self) -> str:
        return _PROVIDER_ID

    @property
    def source_provenance(self) -> str:
        return _SOURCE_PROVENANCE

    def health_facts_for(
        self,
        context: CapabilityHealthProjectionContext,
    ) -> tuple[CapabilityHealthFact, ...]:
        if context.validation is None:
            return ()
        return dependency_validation_health_facts(
            capability_key=context.capability.canonical_key,
            validation=context.validation,
            scope_application_id=context.scope_application_id,
            scope_tenant_id=context.scope_tenant_id,
        )


def dependency_validation_health_provider() -> CapabilityHealthProvider:
    return DependencyValidationHealthProvider()
