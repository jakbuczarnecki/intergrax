# © Artur Czarnecki. All rights reserved.

"""Provider-neutral capability dependency validator core (P1.3)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from intergrax.applications.contracts.capability_dependency import (
    CapabilityDependency,
    CapabilityDependencyAvailabilityStatus,
    CapabilityDependencyDeclarationConflictError,
    CapabilityDependencyDegradationEvidence,
    CapabilityDependencyEvaluation,
    CapabilityDependencyFailureEvidence,
    CapabilityDependencyOutcome,
    CapabilityDependencyProvider,
    CapabilityDependencyProviderConflictError,
    CapabilityDependencyRequirement,
    CapabilityDependencyValidationContext,
    CapabilityDependencyValidationResult,
    CapabilityRef,
)


def _requirement_rank(requirement: CapabilityDependencyRequirement) -> int:
    if requirement is CapabilityDependencyRequirement.REQUIRED:
        return 1
    return 0


def _status_severity(status: CapabilityDependencyAvailabilityStatus) -> int:
    if status is CapabilityDependencyAvailabilityStatus.UNAVAILABLE:
        return 2
    if status is CapabilityDependencyAvailabilityStatus.UNKNOWN:
        return 1
    return 0


def _merge_source_domains(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(sorted(set(left) | set(right)))


def _canonical_source_domain(source_domains: tuple[str, ...]) -> str:
    if len(source_domains) == 1:
        return source_domains[0]
    return ", ".join(source_domains)


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
            if existing.source_domains != declaration.source_domains:
                merged[key] = existing.model_copy(
                    update={
                        "source_domains": _merge_source_domains(
                            existing.source_domains,
                            declaration.source_domains,
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
                    "source_domains": _merge_source_domains(
                        existing.source_domains,
                        declaration.source_domains,
                    ),
                },
            )
            continue
        raise CapabilityDependencyDeclarationConflictError(
            _empty_validation_result(declarations=declarations),
        )
    return tuple(sorted(merged.values(), key=_declaration_sort_key))


def _merge_evaluations(
    evaluations: Sequence[CapabilityDependencyEvaluation],
    declarations_by_key: dict[tuple[str, str, str], CapabilityDependency],
) -> tuple[CapabilityDependencyEvaluation, ...]:
    grouped: dict[tuple[str, str, str], list[CapabilityDependencyEvaluation]] = {}
    for evaluation in evaluations:
        grouped.setdefault(evaluation.dependency.dedup_key, []).append(evaluation)

    merged: list[CapabilityDependencyEvaluation] = []
    for key in sorted(grouped):
        group = grouped[key]
        declaration = declarations_by_key[key]
        dominant = max(
            group,
            key=lambda item: (
                _status_severity(item.status),
                item.reason,
            ),
        )
        if len(group) == 1:
            reason = dominant.reason
        else:
            reason = "; ".join(
                sorted(
                    {
                        f"{_canonical_source_domain(item.dependency.source_domains)}: {item.reason}"
                        for item in group
                    },
                ),
            )
        merged.append(
            CapabilityDependencyEvaluation(
                dependency=declaration,
                status=dominant.status,
                reason=reason,
            ),
        )
    return tuple(merged)


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


@dataclass(frozen=True, slots=True)
class _TaggedDeclaration:
    provider_id: str
    declaration: CapabilityDependency


class CapabilityDependencyValidator:
    """Collect declarations, merge deterministically, evaluate via domain providers."""

    def __init__(self, providers: Sequence[CapabilityDependencyProvider]) -> None:
        self._providers = tuple(providers)
        providers_by_id: dict[str, CapabilityDependencyProvider] = {}
        for provider in self._providers:
            provider_id = provider.provider_id
            if provider_id in providers_by_id:
                raise CapabilityDependencyProviderConflictError(provider_id)
            providers_by_id[provider_id] = provider
        self._providers_by_id = providers_by_id

    def validate(
        self,
        context: CapabilityDependencyValidationContext,
    ) -> CapabilityDependencyValidationResult:
        tagged_declarations: list[_TaggedDeclaration] = []
        for provider in self._providers:
            for declaration in provider.dependencies_for(context):
                tagged_declarations.append(
                    _TaggedDeclaration(provider.provider_id, declaration),
                )

        raw_evaluations: list[CapabilityDependencyEvaluation] = []
        for tagged in tagged_declarations:
            provider = self._providers_by_id[tagged.provider_id]
            status, reason = provider.evaluate_availability(tagged.declaration, context)
            raw_evaluations.append(
                CapabilityDependencyEvaluation(
                    dependency=tagged.declaration,
                    status=status,
                    reason=reason,
                ),
            )

        declarations = _merge_declarations(
            [tagged.declaration for tagged in tagged_declarations],
        )
        declarations_by_key = {declaration.dedup_key: declaration for declaration in declarations}
        evaluations = _merge_evaluations(raw_evaluations, declarations_by_key)

        required_failures: list[CapabilityDependencyFailureEvidence] = []
        optional_degradations: list[CapabilityDependencyDegradationEvidence] = []

        for evaluation in evaluations:
            declaration = evaluation.dependency
            source_domains = declaration.source_domains
            source_domain = _canonical_source_domain(source_domains)
            if declaration.requirement is CapabilityDependencyRequirement.REQUIRED:
                if evaluation.status is not CapabilityDependencyAvailabilityStatus.AVAILABLE:
                    required_failures.append(
                        CapabilityDependencyFailureEvidence(
                            owner=declaration.owner,
                            dependency=declaration.dependency,
                            dependency_kind=declaration.dependency.kind,
                            requirement=declaration.requirement,
                            status=evaluation.status,
                            reason=evaluation.reason,
                            source_domains=source_domains,
                            source_domain=source_domain,
                        ),
                    )
                continue

            if evaluation.status is CapabilityDependencyAvailabilityStatus.UNAVAILABLE:
                optional_degradations.append(
                    CapabilityDependencyDegradationEvidence(
                        owner=declaration.owner,
                        dependency=declaration.dependency,
                        dependency_kind=declaration.dependency.kind,
                        requirement=declaration.requirement,
                        status=evaluation.status,
                        reason=evaluation.reason,
                        source_domains=source_domains,
                        source_domain=source_domain,
                    ),
                )
            elif evaluation.status is CapabilityDependencyAvailabilityStatus.UNKNOWN:
                optional_degradations.append(
                    CapabilityDependencyDegradationEvidence(
                        owner=declaration.owner,
                        dependency=declaration.dependency,
                        dependency_kind=declaration.dependency.kind,
                        requirement=declaration.requirement,
                        status=evaluation.status,
                        reason=evaluation.reason,
                        source_domains=source_domains,
                        source_domain=source_domain,
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
