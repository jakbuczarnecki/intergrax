# © Artur Czarnecki. All rights reserved.

"""Registry for declarative observability event subscription handlers (OBS-EVOL-9.10)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Union

from intergrax.runtime.events.runtime_event import RuntimeEvent

EventSubscriptionHandler = Callable[[RuntimeEvent], Union[None, Awaitable[None]]]

_REGISTRY: dict[str, EventSubscriptionHandler] = {}


def register_event_subscription_handler(
    handler_id: str,
    handler: EventSubscriptionHandler,
    *,
    replace: bool = False,
) -> None:
    """Register a Tier-3 handler referenced by ``ObservabilityProfile.event_subscriptions``."""
    key = handler_id.strip()
    if not key:
        raise ValueError("handler_id must be non-empty")
    if key in _REGISTRY and not replace:
        raise ValueError(f"duplicate event subscription handler_id: {key!r}")
    _REGISTRY[key] = handler


def get_event_subscription_handler(handler_id: str) -> EventSubscriptionHandler | None:
    return _REGISTRY.get(handler_id.strip())


def require_event_subscription_handler(handler_id: str) -> EventSubscriptionHandler:
    handler = get_event_subscription_handler(handler_id)
    if handler is None:
        raise KeyError(
            f"unregistered event subscription handler_id: {handler_id!r} "
            "— call register_event_subscription_handler() at host bootstrap"
        )
    return handler


def list_event_subscription_handlers() -> list[str]:
    return sorted(_REGISTRY.keys())


def clear_event_subscription_handlers() -> None:
    """Test helper — clears registered subscription handlers."""
    _REGISTRY.clear()
