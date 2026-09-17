# © Artur Czarnecki. All rights reserved.

"""Composition port: durable intent dispatch with optional reliability evidence context."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from intergrax.contracts.provider_invocation import ProviderInvocation
from intergrax.contracts.provider_invocation_dispatch import ProviderInvocationDispatchPort
from intergrax.runtime.enterprise_reliability.provider_invocation_reliability_dispatch_context import (
    ProviderInvocationReliabilityDispatchContext,
)

T = TypeVar("T")


class ProviderInvocationReliabilityAwareDispatchPort(ProviderInvocationDispatchPort, Protocol):
    """``ProviderInvocationDispatchPort`` plus optional GR-7-A8 early lifecycle context."""

    def dispatch_after_intent_persisted(
        self,
        invocation: ProviderInvocation,
        execute: Callable[[], T],
        *,
        reliability_dispatch: ProviderInvocationReliabilityDispatchContext | None = None,
    ) -> T:
        """Fail closed without calling ``execute`` when intent persistence fails."""


__all__ = ["ProviderInvocationReliabilityAwareDispatchPort"]
