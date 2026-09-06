# © Artur Czarnecki. All rights reserved.

"""Deterministic merge helpers for inspection providers (P1.4)."""

from __future__ import annotations

from intergrax.applications.contracts.runtime_inspection.completeness import (
    InspectionCompleteness,
)
from intergrax.applications.contracts.runtime_inspection.explanation import (
    InspectionExplanation,
)
from intergrax.applications.contracts.runtime_inspection.provider import (
    InspectionExtensionEvidence,
    InspectionProviderContribution,
    InspectionProviderFailure,
)


def merge_provider_contributions(
    contributions: tuple[InspectionProviderContribution, ...],
) -> tuple[
    InspectionCompleteness,
    tuple[InspectionExplanation, ...],
    tuple[InspectionExtensionEvidence, ...],
    tuple[InspectionProviderFailure, ...],
]:
    """Merge provider outputs with stable ordering and explicit partial failures."""
    sorted_contributions = sorted(contributions, key=lambda item: item.provider_id)
    explanations: list[InspectionExplanation] = []
    extensions: list[InspectionExtensionEvidence] = []
    failures: list[InspectionProviderFailure] = []
    for contribution in sorted_contributions:
        if contribution.failure is not None:
            failures.append(contribution.failure)
            continue
        explanations.extend(contribution.explanations)
        extensions.extend(contribution.extension_evidence)
    explanations.sort(key=lambda item: (item.subject, item.facts, item.reasons))
    extensions.sort(key=lambda item: (item.provider_id, item.scope.value, item.subject))
    failures.sort(key=lambda item: (item.provider_id, item.reason))
    completeness = InspectionCompleteness.COMPLETE
    if failures:
        completeness = InspectionCompleteness.PARTIAL
    return (
        completeness,
        tuple(explanations),
        tuple(extensions),
        tuple(failures),
    )


def invoke_provider_safely(
    provider: object,
    *,
    method_name: str,
    **kwargs: object,
) -> InspectionProviderContribution:
    """Call one provider method and surface typed partial failure evidence."""
    provider_id = getattr(provider, "provider_id", "unknown")
    try:
        method = getattr(provider, method_name)
        return method(**kwargs)
    except Exception as exc:  # noqa: BLE001 — inspection read path must not fail closed
        return InspectionProviderContribution(
            provider_id=str(provider_id),
            failure=InspectionProviderFailure(
                provider_id=str(provider_id),
                reason=str(exc),
            ),
        )
