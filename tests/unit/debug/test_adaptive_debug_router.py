# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-7.5: adaptive debug read-only routes."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.debug.adaptive_debug_router import create_adaptive_debug_router
from intergrax.runtime.adaptive.adaptation_models import AdaptationEngineRunResult
from intergrax.runtime.adaptive.contracts import HarnessOutcomeSignal
from intergrax.runtime.adaptive.proposal_store import InMemoryProposalStore
from intergrax.runtime.adaptive.signal_store import InMemorySignalStore

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_adaptive_debug_router_lists_signals_and_proposals() -> None:
    signal_store = InMemorySignalStore()
    signal_store.append(
        HarnessOutcomeSignal(
            run_id="run-1",
            tenant_id="tenant-a",
            application_id="lab",
            agent_id="echo",
            task_class="echo.basic",
            utility=0.7,
        )
    )
    proposal_store = InMemoryProposalStore()
    proposal_store.append_run(
        AdaptationEngineRunResult(tenant_id="tenant-a", task_class="echo.basic")
    )
    app = FastAPI()
    app.include_router(
        create_adaptive_debug_router(
            signal_store=signal_store,
            proposal_store=proposal_store,
        )
    )
    client = TestClient(app)
    signals = client.get("/debug/adaptive/signals")
    assert signals.status_code == 200
    assert signals.json()["count"] == 1
    proposals = client.get("/debug/adaptive/proposals")
    assert proposals.status_code == 200
    assert proposals.json()["count"] == 0
