# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Bridge unified run journal into Nexus runtime observability (OBS-BUS-6).

Registers a runtime plugin that dual-writes OTLP-style JSON snapshots on
``TASK_COMPLETED`` and links parser trace export from persisted trace rows.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.nexus.tracing.parser_trace_flush import export_parser_traces_from_events
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader
from intergrax.runtime.observability.journal_export import (
    build_journal_export_snapshot,
    render_journal_otlp_json,
)
from intergrax.runtime.plugins.contract import PolicyEngineLike, RuntimeEventBusLike, RuntimePlugin

logger = logging.getLogger(__name__)


def is_journal_export_enabled() -> bool:
    """Return whether journal OTLP export is active (default: on)."""
    return os.environ.get("INTERGRAX_EXPORT_JOURNAL", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def make_journal_export_runtime_plugin(
    *,
    trace_store: RunTraceReader | None,
    runtime_event_store: RuntimeEventPersistence | None = None,
) -> RuntimePlugin:
    """Runtime plugin — export unified journal + parser traces when a task completes."""

    def _register(
        event_bus: RuntimeEventBusLike,
        _hook_registry: HookRegistry,
        _policy_engine: PolicyEngineLike,
    ) -> None:
        if trace_store is None or not is_journal_export_enabled():
            return

        async def _export_journal(event: RuntimeEvent) -> None:
            if event.event_type != RuntimeEventType.TASK_COMPLETED:
                return
            tenant = event.tenant_id or "default"
            run_id = event.run_id or event.task_id
            try:
                persisted = trace_store.read_run(run_id, tenant)
            except (KeyError, ValueError):
                return

            snapshot = build_journal_export_snapshot(
                persisted,
                runtime_store=runtime_event_store,
            )
            otlp = render_journal_otlp_json(snapshot)
            export_parser_traces_from_events(persisted.events)

            logger.info(
                "journal_export tenant=%s run_id=%s task_id=%s events=%s parser_traces=%s",
                tenant,
                run_id,
                event.task_id,
                snapshot.event_count,
                snapshot.parser_trace_count,
                extra={
                    "run_id": run_id,
                    "task_id": event.task_id,
                    "tenant_id": tenant,
                    "journal_export": snapshot.to_dict(),
                    "journal_otlp": otlp,
                    "journal_ref": event.payload.get("journal_ref"),
                },
            )

        event_bus.subscribe(
            _export_journal,
            event_types={RuntimeEventType.TASK_COMPLETED},
            subscription_id="plugin.journal_export",
        )

    return RuntimePlugin(
        plugin_id="runtime.journal_export",
        version="1.0.0",
        register=_register,
    )


def register_journal_export_plugin(
    plugins: list[RuntimePlugin],
    *,
    trace_store: RunTraceReader | None,
    runtime_event_store: RuntimeEventPersistence | None = None,
    enabled: Optional[bool] = None,
) -> list[RuntimePlugin]:
    """Append journal export plugin to an existing plugin list (lab/product bootstrap)."""
    if enabled is False:
        return plugins
    if enabled is None and not is_journal_export_enabled():
        return plugins
    plugins.append(
        make_journal_export_runtime_plugin(
            trace_store=trace_store,
            runtime_event_store=runtime_event_store,
        )
    )
    return plugins
