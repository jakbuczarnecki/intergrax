# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""
Bridge LLM adapter metrics into Nexus runtime observability (§5.2, §7.1).

Registers a runtime plugin that exports OTLP-style JSON snapshots on ``TASK_COMPLETED``.
Prometheus ``observability_backend`` Integration can scrape ``GET /metrics/llm`` separately.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from intergrax.llm_adapters.tracking.exposition import render_otlp_json
from intergrax.llm_adapters.tracking.metrics import get_llm_metrics_collector, is_metrics_enabled
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.plugins.contract import PolicyEngineLike, RuntimeEventBusLike, RuntimePlugin

logger = logging.getLogger(__name__)


def make_llm_metrics_runtime_plugin() -> RuntimePlugin:
    """Runtime plugin — log/export LLM metrics when a task completes."""

    def _register(
        event_bus: RuntimeEventBusLike,
        _hook_registry: HookRegistry,
        _policy_engine: PolicyEngineLike,
    ) -> None:
        if not is_metrics_enabled():
            return

        async def _export_llm_metrics(event: RuntimeEvent) -> None:
            if event.event_type != RuntimeEventType.TASK_COMPLETED:
                return
            tenant = event.tenant_id or "_platform"
            collector = get_llm_metrics_collector()
            per_tenant = collector.snapshot_for_tenant(tenant)
            if not per_tenant:
                return
            otlp = render_otlp_json()
            logger.info(
                "llm_metrics_export tenant=%s run_id=%s providers=%s",
                tenant,
                event.run_id,
                list(per_tenant.keys()),
                extra={"llm_metrics": per_tenant, "llm_otlp": otlp},
            )

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
