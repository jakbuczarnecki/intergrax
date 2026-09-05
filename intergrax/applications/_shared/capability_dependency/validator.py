# © Artur Czarnecki. All rights reserved.

"""Provider-neutral capability dependency validator core (P1.3)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.applications.contracts.capability_dependency import (
    CapabilityDependency,
    CapabilityDependencyAvailabilityStatus,
    CapabilityDependencyDeclarationConflictError,
    CapabilityDependencyDegradationEvidence,
    CapabilityDependencyEvaluation,
    CapabilityDependencyFailureEvidence,
    CapabilityDependencyOutcome,
    CapabilityDependencyProvider,
    CapabilityDependencyRequirement,
    CapabilityDependencyValidationContext,
    CapabilityDependencyValidationResult,
    CapabilityRef,
)


def _requirement_rank(requirement: CapabilityDependencyRequirement) -> int:
    if requirement is CapabilityDependencyRequirement.REQUIRED:
        return 1
    return 0


def _merge_declarations(
    declarations: Sequence[CapabilityDependency],
) -> tuple[CapabilityDependency, ...]:
    merged: dict[tuple[str, str, str], CapabilityDependency] = {}
    for declaration in declarations:
        key = declaration.dedup_key
        existing = merged.get(key)
        if existing is None:
            merged[key] = declaration
            continue
        if existing == declaration:
            continue
        if existing.requirement is declaration.requirement:
            if existing.source_domain != declaration.source_domain:
                merged[key] = existing.model_copy(
                    update={
                        "source_domain": _merge_source_domains(
                            existing.source_domain,
                            declaration.source_domain,
                        ),
                    },
                )
            continue
        if (
            existing.requirement is CapabilityDependencyRequirement.REQUIRED
            or declaration.requirement is CapabilityDependencyRequirement.REQUIRED
        ):
            winner = (
                existing
                if _requirement_rank(existing.requirement)
                >= _requirement_rank(declaration.requirement)
                else declaration
            )
            merged[key] = winner.model_copy(
                update={
                    "requirement": CapabilityDependencyRequirement.REQUIRED,
                    "source_domain": _merge_source_domains(
                        existing.source_domain,
                        declaration.source_domain,
                    ),
                },
            )
            continue
        raise CapabilityDependencyDeclarationConflictError(
            _empty_validation_result(declarations=declarations),
        )
    return tuple(sorted(merged.values(), key=_declaration_sort_key))


def _merge_source_domains(left: str, right: str) -> str:
    if left == right:
        return left
    parts = sorted({left, right})
    return "+".join(parts)


def _declaration_sort_key(declaration: CapabilityDependency) -> tuple[str, str, str, str]:
    return (
        declaration.owner.canonical_key,
        declaration.dependency.kind.value,
        declaration.dependency.capability_id,
        declaration.requirement.value,
    )


def _evaluation_sort_key(evaluation: CapabilityDependencyEvaluation) -> tuple[str, str, str]:
    dependency = evaluation.dependency
    return (
        dependency.owner.canonical_key,
        dependency.dependency.kind.value,
        dependency.dependency.capability_id,
    )


def _outcome_sort_key(outcome: CapabilityDependencyOutcome) -> str:
    return outcome.owner.canonical_key


def _failure_sort_key(
    failure: CapabilityDependencyFailureEvidence,
) -> tuple[str, str, str]:
    return (
        failure.owner.canonical_key,
        failure.dependency.kind.value,
        failure.dependency.capability_id,
    )


def _degradation_sort_key(
    degradation: CapabilityDependencyDegradationEvidence,
) -> tuple[str, str, str]:
    return (
        degradation.owner.canonical_key,
        degradation.dependency.kind.value,
        degradation.dependency.capability_id,
    )


def _empty_validation_result(
    *,
    declarations: Sequence[CapabilityDependency] = (),
) -> CapabilityDependencyValidationResult:
    return CapabilityDependencyValidationResult(
        declarations=tuple(declarations),
        evaluations=(),
        outcomes=(),
        required_failures=(),
        optional_degradations=(),
    )


class CapabilityDependencyValidator:
    """Collect declarations, merge deterministically, evaluate via domain providers."""

    def __init__(self, providers: Sequence[CapabilityDependencyProvider]) -> None:
        self._providers = tuple(providers)
        self._providers_by_source = {
            provider.source_domain: provider for provider in self._providers
        }

    def validate(
        self,
        context: CapabilityDependencyValidationContext,
    ) -> CapabilityDependencyValidationResult:
        raw_declarations: list[CapabilityDependency] = []
        for provider in self._providers:
            raw_declarations.extend(provider.dependencies_for(context))
        declarations = _merge_declarations(raw_declarations)

        evaluations: list[CapabilityDependencyEvaluation] = []
        required_failures: list[CapabilityDependencyFailureEvidence] = []
        optional_degradations: list[CapabilityDependencyDegradationEvidence] = []

        for declaration in declarations:
            provider = self._providers_by_source.get(declaration.source_domain)
            if provider is None:
                status = CapabilityDependencyAvailabilityStatus.UNKNOWN
                reason = (
                    f"no provider registered for source domain {declaration.source_domain!r}"
                )
            else:
                status, reason = provider.evaluate_availability(declaration, context)

            evaluation = CapabilityDependencyEvaluation(
                dependency=declaration,
                status=status,
                reason=reason,
            )
            evaluations.append(evaluation)

            if declaration.requirement is CapabilityDependencyRequirement.REQUIRED:
                if status is not CapabilityDependencyAvailabilityStatus.AVAILABLE:
                    required_failures.append(
                        CapabilityDependencyFailureEvidence(
                            owner=declaration.owner,
                            dependency=declaration.dependency,
                            dependency_kind=declaration.dependency.kind,
                            requirement=declaration.requirement,
                            status=status,
                            reason=reason,
                            source_domain=declaration.source_domain,
                        ),
                    )
                continue

            if status is CapabilityDependencyAvailabilityStatus.UNAVAILABLE:
                optional_degradations.append(
                    CapabilityDependencyDegradationEvidence(
                        owner=declaration.owner,
                        dependency=declaration.dependency,
                        dependency_kind=declaration.dependency.kind,
                        requirement=declaration.requirement,
                        status=status,
                        reason=reason,
                        source_domain=declaration.source_domain,
                    ),
                )
            elif status is CapabilityDependencyAvailabilityStatus.UNKNOWN:
                optional_degradations.append(
                    CapabilityDependencyDegradationEvidence(
                        owner=declaration.owner,
                        dependency=declaration.dependency,
                        dependency_kind=declaration.dependency.kind,
                        requirement=declaration.requirement,
                        status=status,
                        reason=reason,
                        source_domain=declaration.source_domain,
                    ),
                )

        outcomes = _rollup_outcomes(evaluations)
        return CapabilityDependencyValidationResult(
            declarations=declarations,
            evaluations=tuple(sorted(evaluations, key=_evaluation_sort_key)),
            outcomes=tuple(sorted(outcomes, key=_outcome_sort_key)),
            required_failures=tuple(sorted(required_failures, key=_failure_sort_key)),
            optional_degradations=tuple(
                sorted(optional_degradations, key=_degradation_sort_key),
            ),
        )


def _rollup_outcomes(
    evaluations: Sequence[CapabilityDependencyEvaluation],
) -> tuple[CapabilityDependencyOutcome, ...]:
    grouped: dict[str, list[CapabilityDependencyEvaluation]] = {}
    owners: dict[str, CapabilityRef] = {}
    for evaluation in evaluations:
        owner_key = evaluation.dependency.owner.canonical_key
        grouped.setdefault(owner_key, []).append(evaluation)
        owners[owner_key] = evaluation.dependency.owner

    outcomes: list[CapabilityDependencyOutcome] = []
    for owner_key in sorted(grouped):
        owner_evaluations = tuple(
            sorted(grouped[owner_key], key=_evaluation_sort_key),
        )
        has_required_block = any(
            evaluation.dependency.requirement is CapabilityDependencyRequirement.REQUIRED
            and evaluation.status is not CapabilityDependencyAvailabilityStatus.AVAILABLE
            for evaluation in owner_evaluations
        )
        has_optional_degradation = any(
            evaluation.dependency.requirement is CapabilityDependencyRequirement.OPTIONAL
            and evaluation.status is not CapabilityDependencyAvailabilityStatus.AVAILABLE
            for evaluation in owner_evaluations
        )
        outcomes.append(
            CapabilityDependencyOutcome(
                owner=owners[owner_key],
                available=not has_required_block,
                degraded=not has_required_block and has_optional_degradation,
                evaluations=owner_evaluations,
            ),
        )
    return tuple(outcomes)
