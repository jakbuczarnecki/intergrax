# © Artur Czarnecki. All rights reserved.

"""Tier-3 observability wiring (Phase OBS-1 · OBS-EVOL-9.10)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.event_subscription_registry import (
    EventSubscriptionHandler,
    require_event_subscription_handler,
)
from intergrax.applications._shared.observability_runtime_bridge import (
    ObservabilityWiringOptions,
    resolve_observability_wiring_options,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    EventSubscriptionSpec,
    ObservabilityProfile,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.observability_wiring import NexusObservabilityStores, wire_nexus_observability


class EventSubscriptionWiringError(ValueError):
    """Raised when declarative event subscriptions cannot be wired."""


@dataclass(frozen=True, slots=True)
class ApplicationObservabilityWiring:
    """Resolved observability stores and options for a Tier-3 host."""

    options: ObservabilityWiringOptions
    stores: NexusObservabilityStores


@dataclass(frozen=True, slots=True)
class EventSubscriptionWiring:
    """Bus subscription ids registered from ``ObservabilityProfile``."""

    subscription_ids: tuple[str, ...] = ()


def wire_observability_event_subscriptions(
    bus: RuntimeEventBus,
    profile: ObservabilityProfile,
    *,
    extra_handlers: dict[str, EventSubscriptionHandler] | None = None,
) -> EventSubscriptionWiring:
    """Register declarative taxonomy subscriptions on the runtime event bus."""
    registered: list[str] = []
    for spec in profile.event_subscriptions:
        if not spec.enabled:
            continue
        handler = _resolve_subscription_handler(spec, extra_handlers=extra_handlers)
        bus.subscribe(
            handler,
            event_types=set(spec.event_types) if spec.event_types else None,
            categories=set(spec.categories) if spec.categories else None,
            kind_prefix=spec.kind_prefix,
            ops_hints=set(spec.ops_hints) if spec.ops_hints else None,
            priority=spec.priority,
            subscription_id=spec.subscription_id,
        )
        registered.append(spec.subscription_id)
    return EventSubscriptionWiring(subscription_ids=tuple(registered))


def _resolve_subscription_handler(
    spec: EventSubscriptionSpec,
    *,
    extra_handlers: dict[str, EventSubscriptionHandler] | None,
) -> EventSubscriptionHandler:
    if extra_handlers is not None and spec.handler_id in extra_handlers:
        return extra_handlers[spec.handler_id]
    try:
        return require_event_subscription_handler(spec.handler_id)
    except KeyError as exc:
        raise EventSubscriptionWiringError(
            f"subscription {spec.subscription_id!r} references unknown handler_id "
            f"{spec.handler_id!r}"
        ) from exc


def wire_application_observability(
    env: ApplicationEnvironmentProfile,
    *,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    integration_profile: IntegrationProfile | None = None,
) -> ApplicationObservabilityWiring:
    """Materialize Nexus observability stores from environment profile."""
    options = resolve_observability_wiring_options(env.observability_profile)
    stores = wire_nexus_observability(
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        use_in_memory_trace=options.use_in_memory_trace,
        enable_runtime_events=options.enable_runtime_events,
        integration_profile=integration_profile or env.integration_profile,
    )
    return ApplicationObservabilityWiring(options=options, stores=stores)
