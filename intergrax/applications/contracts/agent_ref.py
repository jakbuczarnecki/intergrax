# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Strongly-typed agent class references for Tier-3 bindings."""

from __future__ import annotations
from intergrax.utils import attribute_access

import importlib
from typing import Any, Callable, TypeVar

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.errors import AgentImportError

AgentT = TypeVar("AgentT", bound=Agent)


def qualname_for_agent(agent_type: type[Agent]) -> str:
    """Fully-qualified class name: ``package.module.ClassName``."""
    if not isinstance(agent_type, type) or not issubclass(agent_type, Agent):
        raise TypeError(f"Expected Agent subclass, got {agent_type!r}")
    return f"{agent_type.__module__}.{agent_type.__qualname__}"


def qualname_for_callable(fn: Callable[..., Any]) -> str:
    """Fully-qualified callable: ``package.module.function``."""
    module = attribute_access.optional(fn, "__module__", None)
    qualname = attribute_access.optional(fn, "__qualname__", None)
    if not module or not qualname:
        raise ValueError(f"Cannot derive qualname for callable {fn!r}")
    return f"{module}.{qualname}"


def resolve_agent_type(*, agent_type: type[Agent] | None, import_path: str | None) -> type[Agent]:
    """Resolve Tier-2 agent class from typed or serialized binding fields."""
    if agent_type is not None:
        if import_path is not None and qualname_for_agent(agent_type) != import_path:
            raise ValueError(
                f"agent_type {agent_type!r} does not match import_path {import_path!r}"
            )
        return agent_type

    if import_path is None:
        raise ValueError("AgentBinding requires agent_type or import_path")

    module_path, _, class_name = import_path.rpartition(".")
    if not module_path or not class_name:
        raise ValueError(f"Invalid import_path: {import_path!r}")

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise AgentImportError(
            f"Cannot import module {module_path!r} for {import_path!r}"
        ) from exc

    try:
        resolved = attribute_access.optional(module, class_name)
    except AttributeError as exc:
        raise AgentImportError(
            f"Module {module_path!r} has no attribute {class_name!r}"
        ) from exc

    if not isinstance(resolved, type) or not issubclass(resolved, Agent):
        raise AgentImportError(f"{import_path!r} is not an Agent subclass")
    return resolved
