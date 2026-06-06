# © Artur Czarnecki. All rights reserved.

"""Recommend-only adaptation scheduler skeleton (Phase W-ADAPT-2.12)."""

from __future__ import annotations

from collections import defaultdict

from intergrax.runtime.adaptive.adaptation_engine import AdaptationEngine
from intergrax.runtime.adaptive.adaptation_models import AdaptationEngineContext, AdaptationEngineRunResult
from intergrax.runtime.adaptive.signal_store import SignalStore


class AdaptationScheduler:
    """
    Hourly recommend-only scheduler entry point.

    Does not invoke AdaptationExecutor — L4-R scope only.
    """

    def __init__(
        self,
        *,
        engine: AdaptationEngine,
        signal_store: SignalStore,
    ) -> None:
        self._engine = engine
        self._signal_store = signal_store

    def run_adaptation_engine(
        self,
        *,
        tenant_id: str | None = None,
        signal_limit: int = 500,
    ) -> list[AdaptationEngineRunResult]:
        """Group recent signals by tenant/task_class and run recommend cycles."""
        signals = self._signal_store.list_signals(tenant_id=tenant_id, limit=signal_limit)
        grouped: dict[tuple[str, str], list] = defaultdict(list)
        for signal in signals:
            grouped[(signal.tenant_id, signal.task_class)].append(signal)

        results: list[AdaptationEngineRunResult] = []
        for (resolved_tenant, task_class), bucket in grouped.items():
            context = AdaptationEngineContext(
                tenant_id=resolved_tenant,
                task_class=task_class,
                signals=bucket,
            )
            results.append(self._engine.run(context))
        return results
