# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest

from intergrax.runtime.nexus.planning.engine_plan_models import EngineNextStep, PlanIntent
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID
from intergrax.runtime.nexus.planning.engine_planner_parse import EnginePlanJsonParser

pytestmark = pytest.mark.gate


def test_parse_valid_generic_plan() -> None:
    raw = json.dumps(
        {
            "version": "1",
            "intent": "generic",
            "next_step": "synthesize",
            "reasoning_summary": "ok",
            "ask_clarifying_question": False,
            "clarifying_question": None,
            "use_websearch": False,
            "use_user_longterm_memory": False,
            "use_rag": True,
            "use_tools": False,
        }
    )
    plan = EnginePlanJsonParser.parse(raw)
    assert plan.intent == PlanIntent.GENERIC
    assert plan.next_step == EngineNextStep.SYNTHESIZE
    assert plan.use_rag is True
    assert RAG_RETRIEVE_TOOL_ID in plan.tool_ids


def test_parse_populates_tool_ids_from_explicit_list() -> None:
    raw = json.dumps(
        {
            "version": "1",
            "intent": "generic",
            "next_step": "synthesize",
            "reasoning_summary": "canonical tools",
            "ask_clarifying_question": False,
            "clarifying_question": None,
            "use_websearch": False,
            "use_user_longterm_memory": False,
            "use_rag": False,
            "use_tools": False,
            "tool_ids": [RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID],
        }
    )
    plan = EnginePlanJsonParser.parse(raw)
    assert plan.tool_ids == [RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID]
    assert plan.use_rag is True
    assert plan.use_websearch is True


def test_parse_rejects_invalid_intent() -> None:
    raw = json.dumps(
        {
            "version": "1",
            "intent": "not-a-real-intent",
            "next_step": "synthesize",
            "reasoning_summary": "x",
            "ask_clarifying_question": False,
            "clarifying_question": None,
            "use_websearch": False,
            "use_user_longterm_memory": False,
            "use_rag": False,
            "use_tools": False,
        }
    )
    with pytest.raises(ValueError, match="Invalid intent"):
        EnginePlanJsonParser.parse(raw)


def test_parse_tolerates_unknown_next_step_with_flag_fallback() -> None:
    raw = json.dumps(
        {
            "version": "1",
            "intent": "generic",
            "next_step": "not-a-step",
            "reasoning_summary": "x",
            "ask_clarifying_question": False,
            "clarifying_question": None,
            "use_websearch": True,
            "use_user_longterm_memory": False,
            "use_rag": False,
            "use_tools": False,
        }
    )
    plan = EnginePlanJsonParser.parse(raw)
    assert plan.next_step == EngineNextStep.WEBSEARCH
