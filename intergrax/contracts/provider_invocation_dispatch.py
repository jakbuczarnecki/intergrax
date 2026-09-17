# © Artur Czarnecki. All rights reserved.

"""Dispatch gate after durable invocation intent (GR-7-A3)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from intergrax.contracts.enterprise_reliability.provider_invocation_reliability_emission import (
    ProviderInvocationReliabilityDispatchContext,
)
from intergrax.contracts.provider_invocation import ProviderInvocation

T = TypeVar("T")


class ProviderInvocationDispatchPort(Protocol):
    """Persist canonical intent, then run provider-bound execute callback."""

    def dispatch_after_intent_persisted(
        self,
        invocation: ProviderInvocation,
        execute: Callable[[], T],
        *,
        reliability_dispatch: ProviderInvocationReliabilityDispatchContext | None = None,
    ) -> T:
        """Fail closed without calling ``execute`` when intent persistence fails."""


__all__ = ["ProviderInvocationDispatchPort"]
