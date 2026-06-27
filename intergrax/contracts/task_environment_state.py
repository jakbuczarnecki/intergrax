# © Artur Czarnecki. All rights reserved.

"""Task-scoped environment state keys and isolation refs for runtime cleanup."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

APP_ENV_STATE_RUNTIME_KEY = "app_env_state.v1"


class WorkspaceIsolationRef(BaseModel):
    """Active shadow workspace handle for a task."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    tenant_id: str
    task_id: str
    root_path: str | None = None


class SandboxIsolationRef(BaseModel):
    """Active sandbox session handle for a task."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    tenant_id: str
    task_id: str
    root_path: str | None = None


class TaskEnvironmentState(BaseModel):
    """Minimal persisted task environment state used by runtime isolation cleanup."""

    model_config = ConfigDict(extra="forbid")

    app_id: str = "unknown"
    shadow_workspace: WorkspaceIsolationRef | None = None
    sandbox_session: SandboxIsolationRef | None = None

    @classmethod
    def from_runtime_blob(cls, raw: object) -> TaskEnvironmentState | None:
        if raw is None:
            return None
        if isinstance(raw, TaskEnvironmentState):
            return raw
        if isinstance(raw, dict):
            payload = dict(raw)
            return cls.model_validate(
                {
                    "app_id": payload.get("app_id", "unknown"),
                    "shadow_workspace": payload.get("shadow_workspace"),
                    "sandbox_session": payload.get("sandbox_session"),
                },
            )
        return None
