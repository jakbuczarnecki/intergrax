# © Artur Czarnecki. All rights reserved.

"""Wire Nexus-backed ACP session hooks for application hosts."""

from __future__ import annotations

from intergrax.agents.authoring.acp_runtime_session_ports import AcpRuntimeSessionHooks
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.agents.run_environment import EffectiveAgentRunEnvironment
from intergrax.contracts.agent_run import AgentRunRequest
from intergrax.runtime.kernel.step_kernel import StepKernelContext


def build_nexus_acp_runtime_session_hooks() -> AcpRuntimeSessionHooks:
    from intergrax.runtime.nexus.agents.acp_routing_trace_bridge import (
        record_acp_routing_rule_evaluation,
    )
    from intergrax.runtime.nexus.agents.acp_uaep_shim import (
        attach_acp_catalog_exec_ctx,
        close_acp_catalog_exec_ctx,
    )
    from intergrax.runtime.nexus.agents.shared_context_bridge import (
        load_view,
        persist_view,
        view_from_task_metadata,
    )

    def _apply_kernel_wiring(
        *,
        host,
        kernel_ctx_holder: list[StepKernelContext],
        merged: EffectiveAgentRunEnvironment,
        request: AgentRunRequest,
    ) -> LLMAdapter | None:
        if host is None or host.runtime_profile is None:
            return None
        from intergrax.llm_adapters.routing.context_bridge import build_routing_context_from_runtime
        from intergrax.runtime.attestation.kernel_wiring import apply_boundary_export_to_kernel
        from intergrax.runtime.nexus.config import RuntimeConfig
        from intergrax.runtime.wiring.attestation_runtime_bridge import (
            apply_attestation_profile_to_runtime_config,
        )
        from intergrax.runtime.wiring.context_runtime_bridge import (
            apply_context_profile_to_runtime_config,
        )
        from intergrax.runtime.wiring.llm_resolver import resolve_llm_adapter

        acp_routing_context = build_routing_context_from_runtime(
            tenant_id=merged.tenant_id,
            agent_id=merged.agent_id,
            metadata=request.metadata,
            budget_limits=merged.resolved_budget_limits,
        )
        router_runtime_config = RuntimeConfig(
            llm_adapter=resolve_llm_adapter(
                host.runtime_profile,
                routing_context=acp_routing_context,
            ),
            production_mode=host.runtime_profile.execution_mode.value == "strict",
            llm_routing_context=acp_routing_context,
        )
        apply_context_profile_to_runtime_config(
            router_runtime_config,
            host.runtime_profile.context_profile,
        )
        apply_attestation_profile_to_runtime_config(
            router_runtime_config,
            host.runtime_profile,
        )
        apply_boundary_export_to_kernel(kernel_ctx_holder[0], router_runtime_config)
        return router_runtime_config.llm_adapter

    return AcpRuntimeSessionHooks(
        apply_runtime_profile_kernel_wiring=_apply_kernel_wiring,
        attach_acp_catalog_exec_ctx=attach_acp_catalog_exec_ctx,
        close_acp_catalog_exec_ctx=close_acp_catalog_exec_ctx,
        on_llm_routing_evaluated=record_acp_routing_rule_evaluation,
        load_shared_context_view=load_view,
        persist_shared_context_view=persist_view,
        view_shared_context_for_task=view_from_task_metadata,
    )
