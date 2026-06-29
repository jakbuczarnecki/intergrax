# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.agent_run import AgentRunResult
from intergrax.contracts.agent_run_enums import AgentRunStatus, TerminalReason
from local_search.local_search_agent import LocalSearchAgent


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_search_exports_search_summary_for_graph_handoff() -> None:
    agent = LocalSearchAgent()
    result = AgentRunResult(
        status=AgentRunStatus.SUCCEEDED,
        output={
            "search_summary": {
                "used": True,
                "reason": "retrieve_complete",
                "query": "pipeline query",
                "num_results": 1,
                "evidence": [
                    {
                        "text": "Handoff evidence chunk",
                        "source_path": "/data/fixture.txt",
                        "chunk_id": "chunk-1",
                    }
                ],
            }
        },
        run_id="run-handoff",
        terminal_reason=TerminalReason.GOAL_MET,
    )

    await agent.on_run_end(result)

    exported = result.structured_data.get("search_summary")
    assert isinstance(exported, dict)
    assert exported.get("num_results") == 1
    evidence = exported.get("evidence")
    assert isinstance(evidence, list) and len(evidence) == 1
    assert evidence[0]["text"] == "Handoff evidence chunk"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_search_skips_empty_search_summary_handoff() -> None:
    agent = LocalSearchAgent()
    result = AgentRunResult(
        status=AgentRunStatus.SUCCEEDED,
        output={"search_summary": {"used": False, "reason": "query_missing", "evidence": []}},
        run_id="run-empty",
        terminal_reason=TerminalReason.GOAL_MET,
    )

    await agent.on_run_end(result)

    assert "search_summary" not in result.structured_data
