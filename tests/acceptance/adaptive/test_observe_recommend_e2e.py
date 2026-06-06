# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-7.6: end-to-end observe -> recommend acceptance (no apply)."""

from __future__ import annotations

import pytest

from intergrax.runtime.adaptive.adaptation_engine import AdaptationEngine
from intergrax.runtime.adaptive.adaptation_models import AdaptationEngineContext
from intergrax.runtime.adaptive.adaptation_scheduler import AdaptationScheduler
from intergrax.runtime.adaptive.bandit_state_store import InMemoryBanditStateStore
from intergrax.runtime.adaptive.governance_pipeline import AdaptationGovernancePipeline
from intergrax.runtime.adaptive.proposal_builder import ProposalBuilder
from intergrax.runtime.adaptive.proposal_cooldown_store import InMemoryProposalCooldownStore
from intergrax.runtime.adaptive.proposal_store import InMemoryProposalStore
from intergrax.runtime.adaptive.routing_tuning_engine import RoutingTuningEngine
from intergrax.runtime.adaptive.signal_collector import SignalAssemblyInput, SignalCollector
from intergrax.runtime.adaptive.signal_store import InMemorySignalStore

pytestmark = [pytest.mark.integration, pytest.mark.gate]


def test_observe_recommend_e2e_without_apply() -> None:
    signal_store = InMemorySignalStore()
    collector = SignalCollector(signal_store, application_id="lab.harness")
    collector.record(
        SignalAssemblyInput(
            run_id="run-e2e-1",
            tenant_id="tenant-a",
            application_id="lab.harness",
            agent_id="echo",
            task_class="echo.basic",
            validation_passed=True,
            actual_tokens=500,
            hitl_interventions=0,
        )
    )
    signals = signal_store.list_signals(limit=10)
    assert len(signals) == 1
    assert signals[0].utility is not None

    bandit_store = InMemoryBanditStateStore()
    governance = AdaptationGovernancePipeline()
    proposal_store = InMemoryProposalStore()
    engine = AdaptationEngine(
        sub_engines=[RoutingTuningEngine(bandit_store, utility_threshold=0.99)],
        proposal_builder=ProposalBuilder(governance),
        bandit_store=bandit_store,
        cooldown_store=InMemoryProposalCooldownStore(),
        proposal_store=proposal_store,
    )
    scheduler = AdaptationScheduler(engine=engine, signal_store=signal_store)
    recommend_results = scheduler.run_adaptation_engine(tenant_id="tenant-a")
    assert recommend_results
    assert recommend_results[0].packages

    stored_runs = proposal_store.list_runs(limit=10)
    assert stored_runs
    assert stored_runs[0].packages
