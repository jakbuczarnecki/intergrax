# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-6: process pattern miner, trace reader, scheduler, and skill stub tests."""

from __future__ import annotations

import json

import pytest

from intergrax.runtime.adaptive.adaptation_engine import AdaptationEngine
from intergrax.runtime.adaptive.adaptation_scheduler import AdaptationScheduler
from intergrax.runtime.adaptive.bandit_state_store import InMemoryBanditStateStore
from intergrax.runtime.adaptive.contracts import ProcessPatternAction
from intergrax.runtime.adaptive.governance_pipeline import AdaptationGovernancePipeline
from intergrax.runtime.adaptive.pattern_skill_stub import build_skill_stub_draft, write_skill_stub_draft
from intergrax.runtime.adaptive.process_pattern_miner import (
    ProcessPatternMiner,
    ProcessPatternMinerConfig,
)
from intergrax.runtime.adaptive.proposal_builder import ProposalBuilder
from intergrax.runtime.adaptive.proposal_cooldown_store import InMemoryProposalCooldownStore
from intergrax.runtime.adaptive.signal_store import InMemorySignalStore
from intergrax.runtime.adaptive.trace_sequence_reader import (
    PersistedTraceSequenceReader,
    ProcessSequenceToken,
    RunProcessSequence,
    extract_tokens_from_run,
)
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunMetadata, RunStats
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _seed_trace_store(*, tenant_id: str = "tenant-a") -> InMemoryRunTraceStore:
    store = InMemoryRunTraceStore()
    for index in range(3):
        run_id = f"run-{index}"
        store.append_event(
            TraceEvent(
                event_id=TraceEvent.new_id(),
                run_id=run_id,
                seq=1,
                ts_utc="2026-06-02T10:00:00+00:00",
                level=TraceLevel.INFO,
                component=TraceComponent.TOOLS,
                step="tool.invoke",
                message="tool completed",
                tags={
                    "capability": "echo.basic",
                    "agent_id": "echo",
                    "tool_id": "harness.echo",
                },
            )
        )
        store.finalize_run(
            run_id,
            RunMetadata(
                run_id=run_id,
                session_id="s1",
                user_id="u1",
                tenant_id=tenant_id,
                started_at_utc="2026-06-02T10:00:00+00:00",
                stats=RunStats(duration_ms=10, llm_usage={}),
            ),
        )
    return store


def test_extract_tokens_from_run_includes_tool_step() -> None:
    persisted = PersistedRun(
        metadata=RunMetadata(
            run_id="run-1",
            session_id="s1",
            user_id="u1",
            tenant_id="tenant-a",
            started_at_utc="2026-06-02T10:00:00+00:00",
            stats=RunStats(duration_ms=10, llm_usage={}),
        ),
        events=[
            {
                "step": "tool.invoke",
                "component": "tool_runtime",
                "tags": {"capability": "echo.basic", "agent_id": "echo", "tool_id": "harness.echo"},
            }
        ],
    )
    tokens = extract_tokens_from_run(persisted)
    assert len(tokens) == 1
    assert tokens[0].tool_id == "harness.echo"
    assert tokens[0].task_class == "echo.basic"


def test_process_pattern_miner_finds_repeated_ngram() -> None:
    trace_store = _seed_trace_store()
    reader = PersistedTraceSequenceReader(trace_store)
    miner = ProcessPatternMiner(sequence_reader=reader)
    result = miner.mine(
        tenant_id="tenant-a",
        config=ProcessPatternMinerConfig(min_support=2, ngram_size=1),
    )
    assert result.scanned_run_count == 3
    assert result.proposals
    assert result.proposals[0].support_count >= 2


def test_skill_stub_generator_writes_draft_file(tmp_path) -> None:
    from intergrax.runtime.adaptive.contracts import ProcessPatternProposal

    proposal = ProcessPatternProposal(
        pattern_id="pat_test123",
        description="task=echo.basic agents=echo tools=harness.echo hitl=False",
        suggested_action=ProcessPatternAction.CREATE_SKILL_DRAFT,
        evidence_run_ids=["run-1"],
        support_count=3,
    )
    draft = build_skill_stub_draft(proposal)
    assert draft is not None
    assert draft.skill_id == "mined.pat_test123"
    written = write_skill_stub_draft(proposal, output_dir=tmp_path)
    assert written is not None
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["tool_ids"] == ["harness.echo"]


def test_scheduler_run_pattern_miner_delegates() -> None:
    sequences = [
        RunProcessSequence(
            run_id=f"run-{index}",
            tenant_id="tenant-a",
            tokens=[
                ProcessSequenceToken(
                    task_class="echo.basic",
                    agent_id="echo",
                    tool_id="harness.echo",
                )
            ],
        )
        for index in range(3)
    ]

    class _StubReader:
        def load_sequences(self, *, tenant_id: str, limit: int = 100) -> list[RunProcessSequence]:
            return sequences

    miner = ProcessPatternMiner(sequence_reader=_StubReader())
    bandit_store = InMemoryBanditStateStore()
    governance = AdaptationGovernancePipeline()
    engine = AdaptationEngine(
        sub_engines=[],
        proposal_builder=ProposalBuilder(governance),
        bandit_store=bandit_store,
        cooldown_store=InMemoryProposalCooldownStore(),
    )
    scheduler = AdaptationScheduler(
        engine=engine,
        signal_store=InMemorySignalStore(),
    )
    scheduler.attach_pattern_miner(miner)
    result = scheduler.run_pattern_miner(
        tenant_id="tenant-a",
        config=ProcessPatternMinerConfig(min_support=2, ngram_size=1),
    )
    assert result.scanned_run_count == 3
    assert result.proposals
