# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sandbox substrate resolution by isolation tier (ECC-4)."""

from __future__ import annotations

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.runtime.codecraft.substrate import CraftSandboxResolution, resolve_craft_sandbox
from intergrax.runtime.sandbox.contracts import SandboxExecCapable
from intergrax.tools.registry.wiring import ToolWiringContext


def resolve_craft_sandbox_session(
    ctx: ToolWiringContext,
    profile: CodeCraftProfile,
    *,
    tenant_id: str,
    task_id: str,
) -> SandboxExecCapable | None:
    """Route execution substrate per ``CodeCraftProfile.isolation_tier``."""
    resolution = resolve_craft_sandbox(ctx, profile, tenant_id=tenant_id, task_id=task_id)
    return resolution.session


def resolve_craft_sandbox_with_evidence(
    ctx: ToolWiringContext,
    profile: CodeCraftProfile,
    *,
    tenant_id: str,
    task_id: str,
) -> CraftSandboxResolution:
    """Resolve substrate and return typed capability evidence for qualification."""
    return resolve_craft_sandbox(ctx, profile, tenant_id=tenant_id, task_id=task_id)
