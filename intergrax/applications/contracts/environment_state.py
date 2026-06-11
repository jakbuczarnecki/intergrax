# © Artur Czarnecki. All rights reserved.

"""Typed application environment state for ApplicationHost hooks (APP-CON-2)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.execution_mode import ExecutionMode


class ApplicationEnvironmentState(BaseModel):
    """
    Host-scoped state surfaced on ``HookContext.runtime_state`` for Tier-3 hooks.

    Authors MAY persist host-local facts across hook invocations within one task
    by returning ``HookResult(action=MODIFY, modified_payload=...)`` — the harness
    merges keys into ``runtime_state`` (RFC 7396-style shallow update at hook registry).

  Wire format key: ``app_env_state.v1`` inside ``HookContext.runtime_state``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "app_env_state.v1"
    app_id: str
    profile_id: str
    execution_mode: ExecutionMode = ExecutionMode.BALANCED
    organization_id: str | None = None
    active_scenario_id: str | None = None
    budget_warn_emitted: bool = False
    budget_exceeded: bool = False
    custom: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_runtime_state(cls, runtime_state: dict[str, Any]) -> ApplicationEnvironmentState | None:
        raw = runtime_state.get("app_env_state.v1")
        if raw is None:
            return None
        if isinstance(raw, ApplicationEnvironmentState):
            return raw
        if isinstance(raw, dict):
            return cls.model_validate(raw)
        return None

    def apply_to_runtime_state(self, runtime_state: dict[str, Any]) -> dict[str, Any]:
        merged = dict(runtime_state)
        merged["app_env_state.v1"] = self.model_dump(mode="json")
        return merged

    def patch_runtime_state(self) -> dict[str, Any]:
        """Payload for ``HookResult.modified_payload`` when updating host state."""
        return {"app_env_state.v1": self.model_dump(mode="json")}


def seed_application_environment_state(
    *,
    app_id: str,
    profile_id: str,
    execution_mode: ExecutionMode,
    organization_id: str | None = None,
    active_scenario_id: str | None = None,
) -> dict[str, Any]:
    """Bootstrap ``HookContext.runtime_state`` for task intake hooks."""
    state = ApplicationEnvironmentState(
        app_id=app_id,
        profile_id=profile_id,
        execution_mode=execution_mode,
        organization_id=organization_id,
        active_scenario_id=active_scenario_id,
    )
    return state.apply_to_runtime_state({})
