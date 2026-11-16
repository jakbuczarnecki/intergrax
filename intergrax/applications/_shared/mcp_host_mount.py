# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Single engine-owned path for mounting composed MCP on harness FastAPI hosts (MCP-P3-01)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastmcp import FastMCP

from intergrax.applications._shared.fastapi_lifespan import LifespanFn
from intergrax.applications._shared.mcp_server_engine import (
    McpServerCompositionDeps,
    compose_mcp_server_from_definition,
)
from intergrax.contracts.mcp_server_definition import McpServerDefinition

if TYPE_CHECKING:
    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


class McpHostMountError(ValueError):
    """Fail-closed MCP host configuration (mount path / enabled surface)."""


def normalize_mcp_mount_path(mount_path: str) -> str:
    """Normalize mount path; reject values that cannot be mounted safely."""
    raw = (mount_path or "").strip()
    if not raw:
        raise McpHostMountError("mcp_mount_path must be non-empty when MCP is enabled")
    if ".." in raw:
        raise McpHostMountError("mcp_mount_path must be an absolute path segment without traversal")
    path = raw if raw.startswith("/") else f"/{raw}"
    if path in {"/", ""} or path.rstrip("/") == "":
        raise McpHostMountError("mcp_mount_path must not be the API root (use a dedicated prefix such as /mcp)")
    normalized = path.rstrip("/")
    return normalized or "/mcp"


def resolve_mcp_mount_path_for_host(mount_path: str, *, enabled: bool) -> str:
    """Validate mount path when MCP is enabled; return normalized path or passthrough when disabled."""
    if not enabled:
        return mount_path
    return normalize_mcp_mount_path(mount_path)


def compose_mcp_fastmcp_for_definition(
    definition: McpServerDefinition,
    deps: McpServerCompositionDeps,
) -> FastMCP:
    """Compose FastMCP from definition; invalid definitions fail at startup (MCP-P3-05)."""
    return compose_mcp_server_from_definition(definition, deps)


def couple_mcp_fastmcp_on_harness_host(
    fastapi_app: FastAPI,
    mcp: FastMCP,
    *,
    runtime: HarnessHostRuntime | None = None,
    env: ApplicationEnvironmentProfile | None = None,
    mount_path: str = "/mcp",
    extra_lifespans: list[LifespanFn] | None = None,
    mcp_enabled: bool = True,
) -> FastAPI:
    """Mount an already-composed FastMCP via the canonical harness adapter (lifespan supervision P3-03)."""
    from intergrax.applications._shared.fastapi_mcp import couple_harness_host_fastapi_with_mcp

    resolved_mount = resolve_mcp_mount_path_for_host(mount_path, enabled=mcp_enabled)
    return couple_harness_host_fastapi_with_mcp(
        fastapi_app,
        mcp,
        runtime=runtime,
        env=env,
        mount_path=resolved_mount,
        extra_lifespans=extra_lifespans,
    )


def mount_mcp_definition_on_harness_host(
    fastapi_app: FastAPI,
    definition: McpServerDefinition,
    deps: McpServerCompositionDeps,
    *,
    runtime: HarnessHostRuntime | None = None,
    env: ApplicationEnvironmentProfile | None = None,
    mount_path: str = "/mcp",
    extra_lifespans: list[LifespanFn] | None = None,
    mcp_enabled: bool = True,
) -> FastAPI:
    """Compose definition → FastMCP → streamable HTTP mount (single public engine mount entry)."""
    mcp = compose_mcp_fastmcp_for_definition(definition, deps)
    return couple_mcp_fastmcp_on_harness_host(
        fastapi_app,
        mcp,
        runtime=runtime,
        env=env,
        mount_path=mount_path,
        extra_lifespans=extra_lifespans,
        mcp_enabled=mcp_enabled,
    )


__all__ = [
    "McpHostMountError",
    "compose_mcp_fastmcp_for_definition",
    "couple_mcp_fastmcp_on_harness_host",
    "mount_mcp_definition_on_harness_host",
    "normalize_mcp_mount_path",
    "resolve_mcp_mount_path_for_host",
]
