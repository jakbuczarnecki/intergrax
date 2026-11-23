# © Artur Czarnecki. All rights reserved.

"""Build ``ACPSessionHostContext`` for direct ``agent.run()`` from harness hosts (PAT-2)."""

from __future__ import annotations

from typing import Any

from intergrax.agents.authoring.acp_session_host import ACPSessionHostContext
from intergrax.applications._shared.acp_runtime_session_hooks import build_nexus_acp_runtime_session_hooks
from intergrax.applications._shared.agent_runtime_governance_wiring import (
    capability_grants_from_application_manifest,
)
from intergrax.applications._shared.declarative_tool_wiring import (
    declarative_catalog_tools_enabled,
)
from intergrax.applications._shared.harness_host_composition import (
    resolve_harness_host_decision_flow_gate,
    resolve_harness_host_declarative_tool_invoker,
    resolve_harness_host_execution_budget_ledger_factory,
)
from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
from intergrax.applications._shared.runtime_boundary_adapters import (
    agent_binding_to_run_binding,
    application_profile_to_runtime_profile,
)
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedgerFactory


def build_acp_session_host_context(
    *,
    app_profile: Any,
    binding: AgentBinding | None = None,
    declarative_tool_invoker: Any = None,
    decision_flow_gate: Any = None,
    execution_budget_ledger_factory: ExecutionBudgetLedgerFactory | None = None,
    runtime_session_hooks: Any = None,
) -> ACPSessionHostContext:
    hooks = runtime_session_hooks or build_nexus_acp_runtime_session_hooks()
    return ACPSessionHostContext(
        runtime_profile=application_profile_to_runtime_profile(app_profile),
        binding=agent_binding_to_run_binding(binding),
        declarative_tool_invoker=declarative_tool_invoker,
        decision_flow_gate=decision_flow_gate,
        execution_budget_ledger_factory=execution_budget_ledger_factory,
        runtime_session_hooks=hooks,
    )


def build_acp_session_host_from_harness(
    runtime: HarnessHostRuntime,
    *,
    binding: AgentBinding | None = None,
) -> ACPSessionHostContext:
    """Attach Decision flow gate and declarative tool invoker from a harness host."""
    tool_wiring = runtime.env_wiring.tool_wiring
    production_mode = runtime.environment.execution_mode.value == "strict"
    if production_mode and declarative_catalog_tools_enabled(tool_wiring.profile):
        capability_grants_from_application_manifest(
            runtime.manifest,
            tenant_id=runtime.tenant_id,
            agent_registry=runtime.registry,
        )
    invoker = resolve_harness_host_declarative_tool_invoker(runtime)
    return build_acp_session_host_context(
        app_profile=runtime.environment,
        binding=binding,
        declarative_tool_invoker=invoker,
        decision_flow_gate=resolve_harness_host_decision_flow_gate(runtime),
        execution_budget_ledger_factory=resolve_harness_host_execution_budget_ledger_factory(
            runtime,
        ),
    )
