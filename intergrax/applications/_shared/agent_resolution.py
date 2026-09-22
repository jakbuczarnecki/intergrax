# © Artur Czarnecki. All rights reserved.

"""Runtime resolution for declarative agent bindings (composition layer)."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from typing import TypeVar

from intergrax.applications.contracts.agent_ref import qualname_for_agent
from intergrax.applications.contracts.errors import AgentImportError
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.tier2_agent import Tier2Agent

AgentT = TypeVar("AgentT", bound=Tier2Agent)


def resolve_agent_type(
    *,
    agent_type: type[Tier2Agent] | None,
    import_path: str | None,
) -> type[Tier2Agent]:
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
        resolved = getattr(module, class_name)
    except AttributeError as exc:
        raise AgentImportError(
            f"Module {module_path!r} has no attribute {class_name!r}"
        ) from exc

    if not isinstance(resolved, type) or not issubclass(resolved, Tier2Agent):
        raise AgentImportError(f"{import_path!r} is not a Tier2Agent subclass")
    return resolved


def resolve_agent_type_from_binding(binding: AgentBinding) -> type[Tier2Agent]:
    return resolve_agent_type(
        agent_type=binding.agent_type,
        import_path=binding.import_path,
    )


def _invoke_build_agent_contract(
    builder: object,
    agent_type: type[Tier2Agent],
) -> object:
    if not callable(builder):
        raise AgentImportError("build_agent_contract must be callable")
    params = list(inspect.signature(builder).parameters.values())
    if not params:
        return builder()
    if len(params) == 1:
        return builder(agent_type)
    raise AgentImportError(
        "build_agent_contract must accept zero arguments or a single agent type argument"
    )


def resolve_agent_contract_from_binding(binding: AgentBinding) -> AgentContract:
    """
    Resolve agent contract metadata without materializing runtime dependencies.

    Requires declarative ``<package>.contract.build_agent_contract`` (zero-arg or
    single ``agent_type`` argument for multi-agent packages).
    """
    agent_type = resolve_agent_type_from_binding(binding)
    package = agent_type.__module__.rsplit(".", 1)[0]
    contract_module_name = f"{package}.contract"
    if importlib.util.find_spec(contract_module_name) is None:
        raise AgentImportError(
            f"Agent {qualname_for_agent(agent_type)!r} has no declarative contract module "
            f"{contract_module_name!r}"
        )
    try:
        contract_module = importlib.import_module(contract_module_name)
    except Exception as exc:
        raise AgentImportError(
            f"Failed to import declarative contract module {contract_module_name!r}"
        ) from exc

    builder = getattr(contract_module, "build_agent_contract", None)
    if not callable(builder):
        raise AgentImportError(
            f"{contract_module_name!r} must define callable build_agent_contract()"
        )

    try:
        contract = _invoke_build_agent_contract(builder, agent_type)
    except Exception as exc:
        raise AgentImportError(
            f"build_agent_contract() failed for {contract_module_name!r}"
        ) from exc

    if not isinstance(contract, AgentContract):
        raise AgentImportError(
            f"build_agent_contract() for {contract_module_name!r} must return AgentContract, "
            f"got {type(contract)!r}"
        )
    return contract


__all__ = [
    "resolve_agent_type",
    "resolve_agent_type_from_binding",
    "resolve_agent_contract_from_binding",
]
