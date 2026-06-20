# © Artur Czarnecki. All rights reserved.

"""Nexus runtime wiring for live LLM routing context (M-LLM-X.11.2 · M-LLM-X.12)."""

from __future__ import annotations

from typing import Any

from intergrax.applications._shared.routing_evaluating_adapter import RoutingContextProvider
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.routing.context_bridge import (
    LLMRoutingRuntimeSnapshot,
    build_routing_context_from_runtime,
)
from intergrax.llm_adapters.routing.contracts import RoutingContext
from intergrax.llm_adapters.routing.runtime_sync import refresh_config_routing_snapshot
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


def init_llm_routing_on_config(
    config: RuntimeConfig,
    routing_context: RoutingContext,
    request: RuntimeRequest,
) -> None:
    """Seed mutable routing snapshot on ``RuntimeConfig``."""
    metadata = dict(request.metadata)
    config.llm_routing_snapshot = LLMRoutingRuntimeSnapshot(
        task_class=routing_context.task_class,
        agent_id=routing_context.agent_id or request.agent_id,
        step_index=routing_context.step_index,
        budget_degrade_active=routing_context.budget_degrade_active,
        metadata=metadata,
    )
    config.llm_routing_context = routing_context


def make_config_routing_context_provider(config: RuntimeConfig) -> RoutingContextProvider:
    """Return provider that refreshes snapshot then reads latest context from ``RuntimeConfig``."""

    def _provider() -> RoutingContext:
        refresh_config_routing_snapshot(
            config,
            tenant_id=config.tenant_id,
            run_id=str(config.metadata.get("run_id", "")) or None,
            metadata=dict(config.llm_routing_snapshot.metadata)
            if config.llm_routing_snapshot is not None
            else None,
            budget_degrade_active=(
                config.llm_routing_snapshot.budget_degrade_active
                if config.llm_routing_snapshot is not None
                else None
            ),
        )
        snapshot = config.llm_routing_snapshot
        if snapshot is None:
            return config.llm_routing_context or RoutingContext()
        return build_routing_context_from_runtime(
            tenant_id=config.tenant_id,
            agent_id=snapshot.agent_id,
            task_class=snapshot.task_class,
            step_index=snapshot.step_index,
            budget_degrade_active=snapshot.budget_degrade_active,
            metadata=snapshot.metadata,
            budget_limits=snapshot.budget_limits,
            invocation_usage=snapshot.invocation_usage,
        )

    return _provider


def sync_llm_routing_snapshot_for_state(state: RuntimeState) -> RoutingContext:
    """Refresh routing snapshot from live Nexus ``RuntimeState`` (budget, metadata)."""
    config = state.context.config
    metadata: dict[str, Any] = dict(state.request.metadata)
    step_index = metadata.get("step_index")
    if not isinstance(step_index, int):
        step_index = None
    degrade_raw = metadata.get("budget_degrade_active")
    if not isinstance(degrade_raw, bool):
        degrade_raw = config.metadata.get("budget_degrade_active")
    budget_degrade = bool(degrade_raw) if isinstance(degrade_raw, bool) else None
    context = refresh_config_routing_snapshot(
        config,
        tenant_id=state.tenant_id,
        agent_id=state.request.agent_id or _optional_str(metadata.get("agent_id")),
        task_class=_optional_str(metadata.get("task_class")),
        step_index=step_index,
        metadata=metadata,
        run_id=state.run_id,
        budget_degrade_active=budget_degrade,
    )
    return context or config.llm_routing_context or RoutingContext()


def wire_llm_routing_observability_on_state(state: RuntimeState) -> None:
    """Attach per-evaluation trace observers to evaluating adapter."""
    state.configure_llm_tracker()


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def maybe_wrap_secondary_routing_adapter(
    adapter: LLMAdapter | None,
    env: ApplicationEnvironmentProfile,
    config: RuntimeConfig,
) -> LLMAdapter | None:
    """Opt-in evaluating wrap for secondary LLM surfaces (M-LLM-X.14.5)."""
    if adapter is None:
        return None
    if not env.llm_routing_evaluating_secondary or env.llm_routing_profile is None:
        return adapter
    from intergrax.applications._shared.llm_resolver import create_adapter_for_routing_evaluation
    from intergrax.applications._shared.routing_evaluating_adapter import (
        RoutingEvaluatingLLMAdapter,
        wrap_routing_evaluating_adapter,
    )

    if isinstance(adapter, RoutingEvaluatingLLMAdapter):
        return adapter
    provider = make_config_routing_context_provider(config)
    return wrap_routing_evaluating_adapter(
        adapter,
        env,
        context_provider=provider,
        adapter_factory=lambda evaluation, ctx: create_adapter_for_routing_evaluation(
            env,
            evaluation,
            ctx,
        ),
    )


def wire_secondary_llm_routing_evaluating(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> None:
    """Wrap secondary LLM adapters when host opts into mid-run routing (M-LLM-X.14.5)."""
    if not env.llm_routing_evaluating_secondary or env.llm_routing_profile is None:
        return
    if config.llm_routing_snapshot is None and config.llm_routing_context is None:
        return

    from intergrax.runtime.nexus.tools.catalog_tool_planner import CatalogToolPlanner

    tool_planner = config.tool_planner
    if isinstance(tool_planner, CatalogToolPlanner):
        wrapped = maybe_wrap_secondary_routing_adapter(tool_planner.llm, env, config)
        if wrapped is not None and wrapped is not tool_planner.llm:
            tool_planner._service.llm = wrapped

    websearch_config = config.websearch_config
    if websearch_config is not None and websearch_config.llm is not None:
        llm_cfg = websearch_config.llm
        llm_cfg.map_adapter = maybe_wrap_secondary_routing_adapter(llm_cfg.map_adapter, env, config)
        llm_cfg.reduce_adapter = maybe_wrap_secondary_routing_adapter(
            llm_cfg.reduce_adapter,
            env,
            config,
        )
        llm_cfg.rerank_adapter = maybe_wrap_secondary_routing_adapter(
            llm_cfg.rerank_adapter,
            env,
            config,
        )
