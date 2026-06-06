# © Artur Czarnecki. All rights reserved.

"""Adaptation scheduler for recommend and verify jobs (Phase W-ADAPT-2.12, W-ADAPT-5.12)."""

from __future__ import annotations

from collections import defaultdict

from intergrax.runtime.adaptive.adaptation_engine import AdaptationEngine
from intergrax.runtime.adaptive.adaptation_models import AdaptationEngineContext, AdaptationEngineRunResult
from intergrax.runtime.adaptive.signal_store import SignalStore
from intergrax.runtime.adaptive.process_pattern_miner import (
    ProcessPatternMiner,
    ProcessPatternMinerConfig,
    ProcessPatternMinerResult,
)
from intergrax.runtime.adaptive.verification_loop import VerificationLoop
from intergrax.runtime.adaptive.verification_models import VerificationContext, VerificationReport


class AdaptationScheduler:
    """
    Scheduler entry point for adaptation engine and verification loop jobs.
    """

    def __init__(
        self,
        *,
        engine: AdaptationEngine,
        signal_store: SignalStore,
        verification_loop: VerificationLoop | None = None,
    ) -> None:
        self._engine = engine
        self._signal_store = signal_store
        self._verification_loop = verification_loop
        self._pattern_miner: ProcessPatternMiner | None = None

    def attach_pattern_miner(self, miner: ProcessPatternMiner) -> None:
        """Register daily pattern miner job target (W-ADAPT-6.5)."""
        self._pattern_miner = miner

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

    def run_verification_loop(
        self,
        *,
        context: VerificationContext,
        tenant_id: str | None = None,
    ) -> VerificationReport:
        """Continuous verify on active canaries (AHIA §9.9)."""
        if self._verification_loop is None:
            raise ValueError("VerificationLoop is not configured on AdaptationScheduler")
        return self._verification_loop.verify_active_profiles(
            context=context,
            tenant_id=tenant_id,
        )

    def run_pattern_miner(
        self,
        *,
        tenant_id: str,
        config: ProcessPatternMinerConfig | None = None,
        run_limit: int = 200,
    ) -> ProcessPatternMinerResult:
        """Daily pattern mining job (AHIA §9.9)."""
        if self._pattern_miner is None:
            raise ValueError("ProcessPatternMiner is not configured on AdaptationScheduler")
        return self._pattern_miner.mine(
            tenant_id=tenant_id,
            config=config,
            run_limit=run_limit,
        )
