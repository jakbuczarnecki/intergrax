# © Artur Czarnecki. All rights reserved.

"""Pyright-focused callback contract typing checks."""

from __future__ import annotations

from intergrax.hosting import HostedApplicationContext, HostedApplicationEvent
from intergrax.hosting.contracts.events import HostedApplicationEventHandler
from intergrax.hosting.contracts.hooks import HostedApplicationHookCallback


def _sync_hook_handler(context: HostedApplicationContext) -> None:
    return None


async def _async_hook_handler(context: HostedApplicationContext) -> None:
    return None


def _sync_event_handler(event: HostedApplicationEvent) -> None:
    return None


async def _async_event_handler(event: HostedApplicationEvent) -> None:
    return None


def test_sync_hook_handler_matches_protocol() -> None:
    callback: HostedApplicationHookCallback = _sync_hook_handler
    assert callable(callback)


def test_async_hook_handler_matches_protocol() -> None:
    callback: HostedApplicationHookCallback = _async_hook_handler
    assert callable(callback)


def test_sync_event_handler_matches_protocol() -> None:
    handler: HostedApplicationEventHandler = _sync_event_handler
    assert callable(handler)


def test_async_event_handler_matches_protocol() -> None:
    handler: HostedApplicationEventHandler = _async_event_handler
    assert callable(handler)
