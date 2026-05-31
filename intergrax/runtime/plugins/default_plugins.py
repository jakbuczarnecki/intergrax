# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default Tier-3 runtime plugins for lab and product hosts."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.governance.contracts.metrics_store import ExecutionMetricsStore
from intergrax.runtime.governance.in_memory_metrics_store import InMemoryMetricsStore
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.metrics.export import persist_run_metrics
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader
from intergrax.runtime.plugins.contract import PolicyEngineLike, RuntimeEventBusLike, RuntimePlugin
from intergrax.runtime.schema.registry import current_runtime_version

logger = logging.getLogger(__name__)


def default_lab_plugins(
    *,
    trace_store: Optional[RunTraceReader] = None,
    metrics_store: Optional[ExecutionMetricsStore] = None,
) -> List[RuntimePlugin]:
    store = metrics_store or InMemoryMetricsStore()
    return [
        RuntimePlugin(
            plugin_id="runtime.compatibility",
            version="1.0.0",
            register=_register_compatibility_probe,
        ),
        RuntimePlugin(
            plugin_id="runtime.metrics_export",
            version="1.0.0",
            register=_make_metrics_plugin_register(trace_store=trace_store, metrics_store=store),
        ),
    ]


def _register_compatibility_probe(
    event_bus: RuntimeEventBusLike,
    _hook_registry: HookRegistry,
    _policy_engine: PolicyEngineLike,
) -> None:
    info = current_runtime_version()
    logger.info(
        "runtime plugin bootstrap contract_bundle=%s schemas=%s",
        info.contract_bundle,
        len(info.supported_schemas),
    )

    def _log_first_task(event: RuntimeEvent) -> None:
        if event.event_type == RuntimeEventType.TASK_CREATED:
            logger.debug(
                "runtime event stream started task=%s schema=%s",
                event.task_id,
                event.schema_version,
            )

    event_bus.subscribe(_log_first_task, event_types={RuntimeEventType.TASK_CREATED})


def _make_metrics_plugin_register(
    *,
    trace_store: Optional[RunTraceReader],
    metrics_store: ExecutionMetricsStore,
) -> Callable[[RuntimeEventBusLike, HookRegistry, PolicyEngineLike], None]:
    def _register(
        event_bus: RuntimeEventBusLike,
        _hook_registry: HookRegistry,
        _policy_engine: PolicyEngineLike,
    ) -> None:
        if trace_store is None:
            return

        async def _persist_metrics(event: RuntimeEvent) -> None:
            if event.event_type != RuntimeEventType.TASK_COMPLETED:
                return
            tenant = event.tenant_id or "default"
            agent_id = event.agent_id or "unknown"
            try:
                persisted = trace_store.read_run(event.run_id, tenant)
            except (ValueError, KeyError):
                return
            persist_run_metrics(
                store=metrics_store,
                persisted=persisted,
                agent_id=agent_id,
            )

        event_bus.subscribe(
            _persist_metrics,
            event_types={RuntimeEventType.TASK_COMPLETED},
            subscription_id="plugin.metrics_export",
        )

    return _register
