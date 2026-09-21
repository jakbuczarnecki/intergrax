# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Strongly-typed agent class references for Tier-3 bindings."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.applications.contracts.factory import CanonicalAgentFactory
from intergrax.contracts.tier2_agent import Tier2Agent


def qualname_for_agent(agent_type: type[Tier2Agent]) -> str:
    """Fully-qualified class name: ``package.module.ClassName``."""
    if not isinstance(agent_type, type) or not issubclass(agent_type, Tier2Agent):
        raise TypeError(f"Expected Tier2Agent subclass, got {agent_type!r}")
    return f"{agent_type.__module__}.{agent_type.__qualname__}"


def qualname_for_callable(fn: Callable[..., object]) -> str:
    """Fully-qualified callable: ``package.module.function``."""
    module = getattr(fn, "__module__", None) or type(fn).__module__
    qualname = getattr(fn, "__qualname__", None) or type(fn).__qualname__
    if not module or not qualname:
        raise ValueError(f"Cannot derive qualname for callable {fn!r}")
    return f"{module}.{qualname}"


def qualname_for_agent_factory(fn: CanonicalAgentFactory) -> str:
    """Fully-qualified name for a canonical Tier-3 agent factory."""
    return qualname_for_callable(fn)


__all__ = ["qualname_for_agent", "qualname_for_agent_factory", "qualname_for_callable"]
