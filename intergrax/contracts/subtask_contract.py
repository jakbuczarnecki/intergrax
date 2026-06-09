# © Artur Czarnecki. All rights reserved.

"""Formal subtask contract for graph delegation (FAUDIT-SUB.1)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.delegation import DelegationSpec


class SubtaskContract(BaseModel):
    """Child-run contract aligned with IDEAL Harness subagent semantics."""

    model_config = ConfigDict(extra="forbid")

    child_agent_id: str
    objective: str = ""
    permission_scopes: tuple[str, ...] = Field(default_factory=tuple)
    isolated_memory_namespace: str = ""
    inherit_tool_policy: bool = False
    allowed_tools: tuple[str, ...] = Field(default_factory=tuple)
    max_llm_calls: int | None = Field(default=None, ge=0)
    max_tool_calls: int | None = Field(default=None, ge=0)

    def to_delegation_spec(
        self,
        *,
        parent_run_id: str | None = None,
        parent_node_id: str | None = None,
    ) -> DelegationSpec:
        return DelegationSpec(
            child_agent_id=self.child_agent_id,
            isolated_memory_namespace=self.isolated_memory_namespace,
            inherit_tool_policy=self.inherit_tool_policy,
            parent_run_id=parent_run_id,
            parent_node_id=parent_node_id,
            permission_scopes=self.permission_scopes,
            objective=self.objective,
            max_llm_calls=self.max_llm_calls,
            max_tool_calls=self.max_tool_calls,
        )
