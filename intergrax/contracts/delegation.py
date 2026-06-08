# © Artur Czarnecki. All rights reserved.

"""Graph delegation spec (architecture §42.14.3, Phase R-Delegate)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.context_assembly import TaskContextAssemblyOptions


class ExploreDelegationProfile(BaseModel):
    """Explore / discovery delegation — synthesis-only return (Phase MEM-DEPTH-4.1)."""

    model_config = ConfigDict(extra="forbid")

    parallel_search_budget: int = Field(default=4, ge=1, le=16)
    max_child_context_tokens: int = Field(default=8_000, ge=512)
    synthesis_only_return: bool = True
    enable_hybrid_retrieval: bool = True


class DelegationSpec(BaseModel):
    """Nexus graph child run — subagent-equivalent semantics without nested harness."""

    model_config = ConfigDict(extra="forbid")

    child_agent_id: str
    objective: str = ""
    permission_scopes: tuple[str, ...] = ()
    isolated_memory_namespace: str = ""
    context_assembly: TaskContextAssemblyOptions | None = None
    inherit_tool_policy: bool = False
    parent_run_id: str | None = None
    parent_node_id: str | None = None
    max_llm_calls: int | None = Field(default=None, ge=0)
    max_tool_calls: int | None = Field(default=None, ge=0)
    explore: ExploreDelegationProfile | None = None

    def resolved_memory_namespace(self, *, task_id: str, node_id: str) -> str:
        if self.isolated_memory_namespace.strip():
            return self.isolated_memory_namespace.strip()
        return f"{task_id}/delegation/{node_id}"
