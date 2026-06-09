# © Artur Czarnecki. All rights reserved.

"""IDEAL-14.2 — citation preservation contract test."""

from __future__ import annotations

import json

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.orchestration_enums import MergeStrategy
from intergrax.runtime.nexus.response.final_response_composer import FinalResponseComposer

pytestmark = pytest.mark.gate


def test_citation_preservation_merge_retains_source_ids() -> None:
    composer = FinalResponseComposer(merge_strategy=MergeStrategy.CITATION_PRESERVING)
    result = AgentExecutionResult(
        agent_id="rag-agent",
        run_id="run-1",
        status=AgentExecutionStatus.COMPLETED,
        summary="answer",
        structured_data={"citations": [{"source_id": "doc-1", "excerpt": "fact"}]},
    )
    second = AgentExecutionResult(
        agent_id="secondary",
        run_id="run-1",
        status=AgentExecutionStatus.COMPLETED,
        summary="",
    )
    payload = json.loads(composer.compose_summary([result, second]))
    citations = payload["agents"][0]["citations"]
    assert citations[0]["source_id"] == "doc-1"
