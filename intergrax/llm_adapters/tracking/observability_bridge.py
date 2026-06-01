# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""
Bridge LLM adapter metrics into Nexus runtime observability (§5.2, §7.1).

Registers a runtime plugin that exports OTLP-style JSON snapshots on ``TASK_COMPLETED``.
Optional Pushgateway push and governance cost signals.
"""

from __future__ import annotations

import logging
from typing import Optional

from intergrax.llm_adapters.governance.llm_cost import evaluate_llm_run_cost
from intergrax.llm_adapters.tracking.exposition import render_otlp_json
from intergrax.llm_adapters.tracking.metrics import get_llm_metrics_collector, is_metrics_enabled
from intergrax.llm_adapters.tracking.prometheus_push import push_llm_metrics_to_gateway
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.plugins.contract import PolicyEngineLike, RuntimeEventBusLike, RuntimePlugin
from intergrax.runtime.policy.policy_engine import PolicyEngine, coerce_policy_engine

logger = logging.getLogger(__name__)


def make_llm_metrics_runtime_plugin() -> RuntimePlugin:
    """Runtime plugin — log/export LLM metrics when a task completes."""

    def _register(
        event_bus: RuntimeEventBusLike,
        _hook_registry: HookRegistry,
        policy_engine: PolicyEngineLike,
    ) -> None:
        if not is_metrics_enabled():
            return
        policy = coerce_policy_engine(policy_engine)  # type: ignore[arg-type]

        async def _export_llm_metrics(event: RuntimeEvent) -> None:
            if event.event_type != RuntimeEventType.TASK_COMPLETED:
                return
            tenant = event.tenant_id or "_platform"
            run_id = event.run_id or ""
            collector = get_llm_metrics_collector()
            per_tenant = collector.snapshot_for_tenant(tenant)
            if not per_tenant:
                return

            cost, governance = policy.evaluate_llm_cost_on_task_completed(
                tenant_id=tenant,
                run_id=run_id,
            )
            otlp = render_otlp_json()
            logger.info(
                "llm_metrics_export tenant=%s run_id=%s task_id=%s tokens=%s calls=%s",
                tenant,
                run_id,
                event.task_id,
                cost.total_tokens,
                cost.total_calls,
                extra={
                    "run_id": run_id,
                    "task_id": event.task_id,
                    "tenant_id": tenant,
                    "llm_metrics": per_tenant,
                    "llm_otlp": otlp,
                    "llm_cost_evaluation": {
                        "total_tokens": cost.total_tokens,
                        "warn": cost.warn_threshold_exceeded,
                        "reasons": cost.reasons,
                        "policy_action": governance.decision.value,
                    },
                },
            )
            if governance.decision.value == "warn" or cost.warn_threshold_exceeded:
                logger.warning(
                    "llm_governance_warn tenant=%s run_id=%s reasons=%s",
                    tenant,
                    run_id,
                    cost.reasons,
                )

            push_llm_metrics_to_gateway(grouping_key=f"tenant/{tenant}")

        event_bus.subscribe(
            _export_llm_metrics,
            event_types={RuntimeEventType.TASK_COMPLETED},
            subscription_id="plugin.llm_metrics_export",
        )

    return RuntimePlugin(
        plugin_id="runtime.llm_metrics_export",
        version="1.0.0",
        register=_register,
    )


def register_llm_observability_plugin(
    plugins: list[RuntimePlugin],
    *,
    enabled: Optional[bool] = None,
) -> list[RuntimePlugin]:
    """Append LLM metrics plugin to an existing plugin list (lab/product bootstrap)."""
    if enabled is False:
        return plugins
    if enabled is None and not is_metrics_enabled():
        return plugins
    plugins.append(make_llm_metrics_runtime_plugin())
    return plugins
