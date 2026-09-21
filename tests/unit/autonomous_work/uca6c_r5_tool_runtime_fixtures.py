# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R5 helpers — canonical RuntimeToolInvoker + host RuntimeState binding."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.environment_profile.bundles import IsolationBundle
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    PolicyRulesProfile,
    SandboxProfile,
)
from intergrax.runtime.nexus.tools.catalog_tool_invocation_port import (
    CatalogToolInvocationBinding,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.sandbox.isolation_gate import sandbox_availability_provider
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.tools.providers.sandbox.bundle import (
    CODE_EXEC_TOOL_ID,
    register_sandbox_tools,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.builder import build_runtime_state_for_tests


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


def build_r5_catalog_tool_binding(
    ctx: ToolWiringContext,
    *,
    run_seed: str,
    allowed_tool_ids: set[str] | None = None,
) -> tuple[CatalogToolInvocationBinding, ToolRegistry, RuntimeToolInvoker]:
    registry = ToolRegistry()
    register_sandbox_tools(registry, ctx)
    allowed = allowed_tool_ids if allowed_tool_ids is not None else {CODE_EXEC_TOOL_ID}
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        sandbox_availability=sandbox_availability_provider(ctx),
        scope_policy=StaticToolScopePolicy(allowed_tools=allowed),
    )
    state = build_runtime_state_for_tests(run_id=run_seed)
    policy_env = sandbox_env_profile()
    cfg = replace(
        state.context.config,
        policy_bundle=wire_policy_bundle(policy_env),
        tool_invoker=invoker,
    )
    state = replace(state, context=replace(state.context, config=cfg))
    binding = CatalogToolInvocationBinding(
        tool_invoker=invoker,
        state_supplier=lambda: state,
        caller_agent_id=state.request.agent_id,
    )
    return binding, registry, invoker
