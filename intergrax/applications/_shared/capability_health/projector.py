# © Artur Czarnecki. All rights reserved.

"""Provider-neutral effective capability health projector (P1.5)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.applications._shared.capability_health.redaction import (
    sanitize_health_provider_failure_reason,
)
from intergrax.applications.contracts.capability_health import (
    CapabilityHealthFact,
    CapabilityHealthFactStatus,
    CapabilityHealthProjectionContext,
    CapabilityHealthProvider,
    CapabilityHealthProviderConflictError,
    CapabilityHealthProviderFailure,
    CapabilityHealthReason,
    CapabilityHealthStatus,
    EffectiveCapabilityHealth,
)


def _fact_status_severity(status: CapabilityHealthFactStatus) -> int:
    if status is CapabilityHealthFactStatus.UNSATISFIED:
        return 3
    if status is CapabilityHealthFactStatus.UNKNOWN:
        return 2
    if status is CapabilityHealthFactStatus.DEGRADED:
        return 1
    return 0


def _fact_sort_key(fact: CapabilityHealthFact) -> tuple[str, str, str, str, str, str]:
    return (
        fact.capability.canonical_key,
        fact.condition_kind.value,
        fact.condition_ref,
        fact.scope_application_id or "",
        fact.scope_tenant_id or "",
        fact.status.value,
    )


def _reason_sort_key(reason: CapabilityHealthReason) -> tuple[str, str, str, str]:
    return (
        reason.reason_code,
        reason.source,
        reason.subject_ref,
        reason.detail or "",
    )


def _merge_conflicting_facts(
    facts: Sequence[CapabilityHealthFact],
) -> tuple[CapabilityHealthFact, ...]:
    grouped: dict[tuple[str, str, str, str | None, str | None], list[CapabilityHealthFact]] = {}
    for fact in facts:
        grouped.setdefault(fact.semantic_key, []).append(fact)

    merged: list[CapabilityHealthFact] = []
    for key in sorted(grouped):
        group = grouped[key]
        if len(group) == 1:
            merged.append(group[0])
            continue
        dominant = max(
            group,
            key=lambda item: (
                _fact_status_severity(item.status),
                item.reason.reason_code,
                item.reason.detail or "",
            ),
        )
        merged.append(dominant)
    return tuple(merged)


def project_status_from_facts(
    facts: Sequence[CapabilityHealthFact],
) -> CapabilityHealthStatus:
    """Pure projection: facts → effective status with deterministic dominance."""
    if any(
        fact.blocking
        and fact.status
        in {
            CapabilityHealthFactStatus.UNSATISFIED,
            CapabilityHealthFactStatus.UNKNOWN,
        }
        for fact in facts
    ):
        return CapabilityHealthStatus.UNAVAILABLE

    if any(
        fact.blocking
        and fact.status is CapabilityHealthFactStatus.DEGRADED
        for fact in facts
    ):
        return CapabilityHealthStatus.UNAVAILABLE

    if any(
        fact.status
        in {
            CapabilityHealthFactStatus.UNSATISFIED,
            CapabilityHealthFactStatus.UNKNOWN,
            CapabilityHealthFactStatus.DEGRADED,
        }
        for fact in facts
    ):
        return CapabilityHealthStatus.DEGRADED

    return CapabilityHealthStatus.READY


def project_effective_capability_health(
    *,
    capability: object,
    facts: Sequence[CapabilityHealthFact],
    provider_failures: Sequence[CapabilityHealthProviderFailure] = (),
    provenance: Sequence[str] = (),
    effective_profile_revision_id: object | None = None,
    effective_profile_fingerprint: str | None = None,
) -> EffectiveCapabilityHealth:
    """Pure function for deterministic health projection."""
    merged_facts = _merge_conflicting_facts(facts)
    sorted_facts = tuple(sorted(merged_facts, key=_fact_sort_key))
    status = project_status_from_facts(sorted_facts)
    reasons = tuple(
        sorted(
            {fact.reason for fact in sorted_facts if fact.status is not CapabilityHealthFactStatus.SATISFIED},
            key=_reason_sort_key,
        ),
    )
    return EffectiveCapabilityHealth(
        capability=capability,  # type: ignore[arg-type]
        status=status,
        reasons=reasons,
        facts=sorted_facts,
        provenance=tuple(sorted(set(provenance))),
        provider_failures=tuple(
            sorted(provider_failures, key=lambda item: (item.provider_id, item.reason)),
        ),
        effective_profile_revision_id=effective_profile_revision_id,  # type: ignore[arg-type]
        effective_profile_fingerprint=effective_profile_fingerprint,
    )


def invoke_health_provider_safely(
    provider: CapabilityHealthProvider,
    context: CapabilityHealthProjectionContext,
) -> tuple[tuple[CapabilityHealthFact, ...], CapabilityHealthProviderFailure | None]:
    """Call one provider and surface conservative failure evidence."""
    try:
        return provider.health_facts_for(context), None
    except Exception as exc:  # noqa: BLE001 — projection must not silently omit
        return (
            (),
            CapabilityHealthProviderFailure(
                provider_id=provider.provider_id,
                reason=sanitize_health_provider_failure_reason(exc),
            ),
        )


class EffectiveCapabilityHealthProjector:
    """Collect domain facts and project one effective capability health outcome."""

    def __init__(self, providers: Sequence[CapabilityHealthProvider]) -> None:
        providers_by_id: dict[str, CapabilityHealthProvider] = {}
        for provider in providers:
            provider_id = provider.provider_id
            if provider_id in providers_by_id:
                raise CapabilityHealthProviderConflictError(provider_id)
            providers_by_id[provider_id] = provider
        self._providers = tuple(sorted(providers, key=lambda item: item.provider_id))

    @property
    def providers(self) -> tuple[CapabilityHealthProvider, ...]:
        return self._providers

    def project(
        self,
        context: CapabilityHealthProjectionContext,
    ) -> EffectiveCapabilityHealth:
        collected_facts: list[CapabilityHealthFact] = []
        provider_failures: list[CapabilityHealthProviderFailure] = []
        provenance: set[str] = set()

        for provider in self._providers:
            facts, failure = invoke_health_provider_safely(provider, context)
            if failure is not None:
                provider_failures.append(failure)
                collected_facts.append(
                    _provider_failure_fact(
                        context=context,
                        provider=provider,
                        reason=failure.reason,
                    ),
                )
                provenance.add(provider.source_provenance)
                continue
            if facts:
                provenance.add(provider.source_provenance)
            collected_facts.extend(facts)

        scoped_facts = tuple(
            fact
            for fact in collected_facts
            if _fact_matches_scope(fact, context)
        )
        return project_effective_capability_health(
            capability=context.capability,
            facts=scoped_facts,
            provider_failures=provider_failures,
            provenance=provenance,
            effective_profile_revision_id=context.effective_profile_revision_id,
            effective_profile_fingerprint=context.effective_profile_fingerprint,
        )


def _fact_matches_scope(
    fact: CapabilityHealthFact,
    context: CapabilityHealthProjectionContext,
) -> bool:
    if (
        context.scope_application_id is not None
        and fact.scope_application_id is not None
        and fact.scope_application_id != context.scope_application_id
    ):
        return False
    if (
        context.scope_tenant_id is not None
        and fact.scope_tenant_id is not None
        and fact.scope_tenant_id != context.scope_tenant_id
    ):
        return False
    return True


def _provider_failure_fact(
    *,
    context: CapabilityHealthProjectionContext,
    provider: CapabilityHealthProvider,
    reason: str,
) -> CapabilityHealthFact:
    from intergrax.applications.contracts.capability_health.fact import (
        CapabilityHealthConditionKind,
    )

    return CapabilityHealthFact(
        capability=context.capability,
        source=provider.source_provenance,
        condition_kind=CapabilityHealthConditionKind.PROVIDER_FAILURE,
        condition_ref=provider.provider_id,
        scope_application_id=context.scope_application_id,
        scope_tenant_id=context.scope_tenant_id,
        status=CapabilityHealthFactStatus.UNKNOWN,
        blocking=True,
        reason=CapabilityHealthReason(
            reason_code="provider.failure",
            source=provider.source_provenance,
            subject_ref=provider.provider_id,
            detail=reason,
        ),
        provider_id=provider.provider_id,
    )
