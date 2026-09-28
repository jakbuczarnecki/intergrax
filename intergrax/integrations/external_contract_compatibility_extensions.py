# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed compatibility extension bundle and composition helpers (P2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.external_contract_compatibility import (
    ExternalContractCompatibilityAssessment,
    ExternalContractCompatibilityAssessmentRequest,
    ExternalContractCompatibilityEvaluator,
    ExternalContractCompatibilityEvidence,
    ExternalContractCompatibilityEvidencePolicy,
    ExternalContractCompatibilityExpectation,
    ExternalContractCompatibilityExpectationResolver,
    ExternalContractEvidenceCollectionRequest,
    ExternalContractEvidenceProvider,
    ExternalContractExpectationKey,
)
from intergrax.integrations.contracts.plugin import IntegrationPlugin
from intergrax.integrations.external_contract_compatibility_service import (
    ExternalContractCompatibilityService,
)


class ExternalContractCompatibilityExtensionsError(ValueError):
    """Invalid or ambiguous compatibility extension composition."""


def _require_extension_id(value: object, label: str) -> str:
    if type(value) is not str:
        raise ExternalContractCompatibilityExtensionsError(
            f"{label} must be str, got {type(value).__name__}"
        )
    if not value or value != value.strip():
        raise ExternalContractCompatibilityExtensionsError(
            f"{label} must be non-empty trimmed text"
        )
    return value


def _validate_unique_resolver_ids(
    resolvers: tuple[ExternalContractCompatibilityExpectationResolver, ...],
) -> None:
    seen: set[str] = set()
    for resolver in resolvers:
        extension_id = _require_extension_id(resolver.resolver_id, "resolver_id")
        if extension_id in seen:
            raise ExternalContractCompatibilityExtensionsError(
                f"duplicate expectation_resolver resolver_id: {extension_id!r}"
            )
        seen.add(extension_id)


def _validate_unique_evidence_provider_ids(
    providers: tuple[ExternalContractEvidenceProvider, ...],
) -> None:
    seen: set[str] = set()
    for provider in providers:
        extension_id = _require_extension_id(
            provider.evidence_provider_id, "evidence_provider_id"
        )
        if extension_id in seen:
            raise ExternalContractCompatibilityExtensionsError(
                f"duplicate evidence_provider evidence_provider_id: {extension_id!r}"
            )
        seen.add(extension_id)


def _validate_unique_evaluator_ids(
    evaluators: tuple[ExternalContractCompatibilityEvaluator, ...],
) -> None:
    seen: set[str] = set()
    for evaluator in evaluators:
        extension_id = _require_extension_id(evaluator.evaluator_id, "evaluator_id")
        if extension_id in seen:
            raise ExternalContractCompatibilityExtensionsError(
                f"duplicate evaluator evaluator_id: {extension_id!r}"
            )
        seen.add(extension_id)


@dataclass(frozen=True, slots=True)
class ExternalContractCompatibilityExtensions:
    """Immutable bundle of compatibility extensions from one composition source."""

    expectation_resolvers: tuple[
        ExternalContractCompatibilityExpectationResolver, ...
    ] = ()
    evidence_providers: tuple[ExternalContractEvidenceProvider, ...] = ()
    evaluators: tuple[ExternalContractCompatibilityEvaluator, ...] = ()

    def __post_init__(self) -> None:
        _validate_unique_resolver_ids(self.expectation_resolvers)
        _validate_unique_evidence_provider_ids(self.evidence_providers)
        _validate_unique_evaluator_ids(self.evaluators)


def external_contract_compatibility_extensions(
    *,
    expectation_resolvers: tuple[
        ExternalContractCompatibilityExpectationResolver, ...
    ] = (),
    evidence_providers: tuple[ExternalContractEvidenceProvider, ...] = (),
    evaluators: tuple[ExternalContractCompatibilityEvaluator, ...] = (),
) -> ExternalContractCompatibilityExtensions:
    """Construct and validate a compatibility extension bundle (fail closed on duplicates)."""
    return ExternalContractCompatibilityExtensions(
        expectation_resolvers=expectation_resolvers,
        evidence_providers=evidence_providers,
        evaluators=evaluators,
    )


def collect_external_contract_compatibility_evidence(
    extensions: ExternalContractCompatibilityExtensions,
    request: ExternalContractEvidenceCollectionRequest,
) -> tuple[ExternalContractCompatibilityEvidence, ...]:
    """Invoke evidence providers in declaration order; no assessment side effects."""
    collected: list[ExternalContractCompatibilityEvidence] = []
    for provider in extensions.evidence_providers:
        collected.extend(provider.collect(request))
    return tuple(collected)


def resolve_external_contract_compatibility_expectation(
    extensions: ExternalContractCompatibilityExtensions,
    key: ExternalContractExpectationKey,
) -> ExternalContractCompatibilityExpectation | None:
    """First non-None resolver result in declaration order."""
    for resolver in extensions.expectation_resolvers:
        resolved = resolver.resolve(key)
        if resolved is not None:
            return resolved
    return None


def build_external_contract_compatibility_service(
    extensions: ExternalContractCompatibilityExtensions,
    *,
    evidence_policy: ExternalContractCompatibilityEvidencePolicy,
) -> ExternalContractCompatibilityService:
    """Wire extension evaluators into the generic assessment service."""
    return ExternalContractCompatibilityService(
        evaluators=extensions.evaluators,
        evidence_policy=evidence_policy,
    )


def assess_external_contract_compatibility(
    extensions: ExternalContractCompatibilityExtensions,
    *,
    evidence_policy: ExternalContractCompatibilityEvidencePolicy,
    collection_request: ExternalContractEvidenceCollectionRequest,
    assessment_request: ExternalContractCompatibilityAssessmentRequest,
) -> ExternalContractCompatibilityAssessment:
    """
    Composition-root flow: collect typed evidence, then pure assess (no service I/O).
    """
    collected = collect_external_contract_compatibility_evidence(
        extensions, collection_request
    )
    if assessment_request.evidence:
        merged = assessment_request.evidence + collected
    else:
        merged = collected
    merged_request = ExternalContractCompatibilityAssessmentRequest(
        assessment_id=assessment_request.assessment_id,
        expectation=assessment_request.expectation,
        evidence=merged,
        assessed_at=assessment_request.assessed_at,
        assessment_window=assessment_request.assessment_window,
        explicit_evaluator_ids=assessment_request.explicit_evaluator_ids,
    )
    service = build_external_contract_compatibility_service(
        extensions, evidence_policy=evidence_policy
    )
    return service.assess(merged_request)


@runtime_checkable
class IntegrationPluginCompatibilityContributor(Protocol):
    """Optional Integration plugin hook for typed compatibility extensions."""

    @classmethod
    def external_contract_compatibility_extensions(
        cls,
    ) -> ExternalContractCompatibilityExtensions: ...


def external_contract_compatibility_extensions_for_plugin(
    plugin: type[IntegrationPlugin],
) -> ExternalContractCompatibilityExtensions:
    """Resolve extensions from plugin when contributor protocol is satisfied."""
    if issubclass(plugin, IntegrationPluginCompatibilityContributor):
        bundle = plugin.external_contract_compatibility_extensions()
        if type(bundle) is not ExternalContractCompatibilityExtensions:
            raise TypeError(
                f"{plugin.__qualname__}.external_contract_compatibility_extensions() "
                "must return ExternalContractCompatibilityExtensions"
            )
        return bundle
    return ExternalContractCompatibilityExtensions()
