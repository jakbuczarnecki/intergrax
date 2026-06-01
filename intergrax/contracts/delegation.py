# © Artur Czarnecki. All rights reserved.

"""Graph delegation spec (architecture §42.14.3, Phase R-Delegate)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.context_assembly import TaskContextAssemblyOptions


class DelegationSpec(BaseModel):
    """Nexus graph child run — subagent-equivalent semantics without nested harness."""

    model_config = ConfigDict(extra="forbid")

    child_agent_id: str
    isolated_memory_namespace: str = ""
    context_assembly: TaskContextAssemblyOptions | None = None
    inherit_tool_policy: bool = True
    parent_run_id: str | None = None
    parent_node_id: str | None = None

    def resolved_memory_namespace(self, *, task_id: str, node_id: str) -> str:
        if self.isolated_memory_namespace.strip():
            return self.isolated_memory_namespace.strip()
        return f"{task_id}/delegation/{node_id}"
