# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R5 helpers — production-shaped catalog tool invocation composition."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications._shared.uca6c_codecraft_qualified_execution_composition import (
    build_execution_bound_catalog_tool_invoker_for_qualified_capability,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.environment_profile.bundles import IsolationBundle
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    PolicyRulesProfile,
    SandboxProfile,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.nexus_execution_bound_catalog_tool_invoker import (
    NexusExecutionBoundCatalogToolInvoker,
)
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.registry import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext


def sandbox_env_profile() -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="uca6c-r5-sandbox")
    return profile.model_copy(
        update={
            "isolation": IsolationBundle(
                sandbox=SandboxProfile(enable_exec_tool=True),
            ),
            "policy_rules": PolicyRulesProfile(
                inline_rules=[],
                policy_enforcement_mode="enforce",
            ),
        },
    )


def build_sandbox_session(
    tmp_path: Path, *, tenant_id: str, task_id: str
) -> SandboxSession:
    return SandboxSession.create(
        tmp_path,
        tenant_id=tenant_id,
        task_id=task_id,
        allowed_operations=frozenset(
            {"echo", "write_file", "read_file", "list_files", "run_python"},
        ),
    )


def build_r5_production_catalog_tool_invoker(
    ctx: ToolWiringContext,
    *,
    tenant_id: str,
    caller_agent_id: str = "worker-uca6c-r5",
) -> tuple[ExecutionBoundCatalogToolInvoker, ToolRegistry, RuntimeToolInvoker]:
    """Production composition builder (lab profile, non-STRICT governance)."""
    registry = ToolRegistry()
    tool_wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled_bundles=frozenset({"sandbox"})),
        wiring_context=ctx,
        registry=registry,
    )
    invoker = build_execution_bound_catalog_tool_invoker_for_qualified_capability(
        tool_wiring,
        sandbox_env_profile(),
        caller_agent_id=caller_agent_id,
        tenant_id=tenant_id,
    )
    assert isinstance(invoker, NexusExecutionBoundCatalogToolInvoker)
    return invoker, registry, invoker.tool_invoker


# Backward-compatible alias for tests migrating off manual CatalogToolInvocationBinding.
build_r5_catalog_tool_binding = build_r5_production_catalog_tool_invoker
