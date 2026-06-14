# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Framework agent bridge (Tier-1 → Tier-2)."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.utils.lazy_export import export_from_import_path

__all__ = ["Agent", "AgentEngine", "UAEPExecutor", "supports_uaep", "UAEPAgent"]

_LAZY: dict[str, tuple[str, str]] = {
    "AgentEngine": ("intergrax.agents.agent_engine", "AgentEngine"),
    "UAEPExecutor": ("intergrax.agents.uaep", "UAEPExecutor"),
    "UAEPAgent": ("intergrax.agents.uaep_protocol", "UAEPAgent"),
    "supports_uaep": ("intergrax.agents.uaep_protocol", "supports_uaep"),
}


def __getattr__(name: str) -> object:
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = spec
    return export_from_import_path(module_path, attr)
