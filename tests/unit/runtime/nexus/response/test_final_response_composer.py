# © Artur Czarnecki. All rights reserved.

"""IDEAL-9.2 — merge strategy test matrix."""

from __future__ import annotations

import json

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.orchestration_enums import MergeStrategy
from intergrax.runtime.nexus.response.final_response_composer import FinalResponseComposer

pytestmark = pytest.mark.gate


def _result(agent_id: str, summary: str, citations: list[dict[str, str]] | None = None) -> AgentExecutionResult:
    structured: dict = {}
    if citations is not None:
        structured["citations"] = citations
    return AgentExecutionResult(
        agent_id=agent_id,
        run_id="run-1",
        status=AgentExecutionStatus.COMPLETED,
        summary=summary,
        structured_data=structured,
    )


@pytest.mark.parametrize(
    "strategy,expected_substring",
    [
        (MergeStrategy.CONCAT, "[agent-a]"),
        (MergeStrategy.LAST_WINS, "second"),
        (MergeStrategy.STRUCTURED_JSON, '"agents"'),
        (MergeStrategy.CITATION_PRESERVING, "citations"),
    ],
)
def test_merge_strategy_matrix(strategy: MergeStrategy, expected_substring: str) -> None:
    composer = FinalResponseComposer(merge_strategy=strategy)
    results = [
        _result("agent-a", "first answer"),
        _result("agent-b", "second answer", citations=[{"id": "src-1"}]),
    ]
    summary = composer.compose_summary(results)
    assert expected_substring in summary
    if strategy is MergeStrategy.CITATION_PRESERVING:
        payload = json.loads(summary)
        assert payload["agents"][1]["citations"] == [{"id": "src-1"}]
