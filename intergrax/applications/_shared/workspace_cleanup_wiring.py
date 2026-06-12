# © Artur Czarnecki. All rights reserved.

"""Shadow/sandbox lifespan cleanup and env-state isolation refs (APP-CON-8 · APP-PROD-8)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from intergrax.applications._shared.fastapi_mcp import LifespanFn, apply_lifespans, make_scheduler_lifespan
from intergrax.applications.contracts.environment_state import (
    ApplicationEnvironmentState,
    SandboxIsolationRef,
    WorkspaceIsolationRef,
)
from intergrax.runtime.hooks.hook_context import HookContext
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_SESSION_ID_KEY
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.workspace.shadow_workspace import SHADOW_WORKSPACE_ID_KEY

if TYPE_CHECKING:
    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime

REQUIRED_FACTORY_MARKER = "build_factory_lifespans"


def _tenant_id(ctx: HookContext) -> str:
    value = ctx.runtime_state.get("tenant_id")
    return value if isinstance(value, str) and value else "default"


def _resolve_root_path(manager: ShadowWorkspaceManager | SandboxSessionManager | None, handle_id: str) -> str | None:
    if manager is None:
        return None
    handle = manager.get(handle_id)
    if handle is None:
        return None
    root = getattr(handle, "root", None)
    return str(root) if root is not None else None


def sync_isolation_refs_for_hook(
    ctx: HookContext,
    state: ApplicationEnvironmentState,
    *,
    shadow_manager: ShadowWorkspaceManager | None = None,
    sandbox_manager: SandboxSessionManager | None = None,
) -> ApplicationEnvironmentState:
    """Mirror active shadow/sandbox handles into ``ApplicationEnvironmentState``."""
    tenant_id = _tenant_id(ctx)
    workspace_id = ctx.runtime_state.get(SHADOW_WORKSPACE_ID_KEY)
    session_id = ctx.runtime_state.get(SANDBOX_SESSION_ID_KEY)

    shadow_ref = state.shadow_workspace
    if isinstance(workspace_id, str) and workspace_id:
        shadow_ref = WorkspaceIsolationRef(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            task_id=ctx.task_id,
            root_path=_resolve_root_path(shadow_manager, workspace_id),
        )
    elif workspace_id is None and ctx.runtime_state.get("clear_isolation_refs"):
        shadow_ref = None

    sandbox_ref = state.sandbox_session
    if isinstance(session_id, str) and session_id:
        sandbox_ref = SandboxIsolationRef(
            session_id=session_id,
            tenant_id=tenant_id,
            task_id=ctx.task_id,
            root_path=_resolve_root_path(sandbox_manager, session_id),
        )
    elif session_id is None and ctx.runtime_state.get("clear_isolation_refs"):
        sandbox_ref = None

    if shadow_ref is state.shadow_workspace and sandbox_ref is state.sandbox_session:
        return state
    return state.model_copy(
        update={
            "shadow_workspace": shadow_ref,
            "sandbox_session": sandbox_ref,
        }
    )


def purge_all_workspace_sessions(
    *,
    shadow_manager: ShadowWorkspaceManager | None,
    sandbox_manager: SandboxSessionManager | None,
) -> tuple[int, int]:
    """Dispose all active shadow workspaces and sandbox sessions."""
    shadow_disposed = shadow_manager.dispose_all_active() if shadow_manager is not None else 0
    sandbox_disposed = sandbox_manager.dispose_all_active() if sandbox_manager is not None else 0
    return shadow_disposed, sandbox_disposed


def make_workspace_cleanup_lifespan(
    shadow_manager: ShadowWorkspaceManager | None,
    sandbox_manager: SandboxSessionManager | None,
) -> LifespanFn:
    """FastAPI lifespan that purges lingering isolation sessions on host shutdown."""

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        purge_all_workspace_sessions(
            shadow_manager=shadow_manager,
            sandbox_manager=sandbox_manager,
        )

    return _lifespan


def build_factory_lifespans(
    runtime: HarnessHostRuntime,
    *,
    schedulers: Sequence[Any] | None = None,
) -> list[LifespanFn]:
    """Standard Tier-3 factory lifespans: workspace cleanup + optional schedulers."""
    lifespans: list[LifespanFn] = [
        make_workspace_cleanup_lifespan(
            runtime.env_wiring.shadow_manager,
            runtime.env_wiring.sandbox_manager,
        )
    ]
    for scheduler in schedulers or ():
        if scheduler is not None:
            lifespans.append(make_scheduler_lifespan(scheduler))
    return lifespans


def apply_factory_lifespans(
    app: FastAPI,
    runtime: HarnessHostRuntime,
    *,
    schedulers: Sequence[Any] | None = None,
) -> FastAPI:
    """Merge workspace cleanup and scheduler lifespans onto a FastAPI app."""
    lifespans = build_factory_lifespans(runtime, schedulers=schedulers)
    return apply_lifespans(app, *lifespans)


def check_factory_workspace_cleanup(factory_path: Path) -> list[str]:
    """Return violations when a host factory omits workspace cleanup lifespan wiring."""
    rel = factory_path.as_posix()
    text = factory_path.read_text(encoding="utf-8")
    if REQUIRED_FACTORY_MARKER not in text:
        return [f"{rel}: must call {REQUIRED_FACTORY_MARKER} for workspace lifespan cleanup"]
    return []


def check_all_factory_workspace_cleanup(applications_root: Path) -> list[str]:
    violations: list[str] = []
    if not applications_root.is_dir():
        return [f"missing applications root: {applications_root}"]
    for path in sorted(applications_root.glob("*_application/host/factory.py")):
        violations.extend(check_factory_workspace_cleanup(path))
    return violations
