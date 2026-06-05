# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Strict JSON parsing for :class:`~intergrax.runtime.nexus.planning.engine_planner.EnginePlanner` (Q+-P.1)."""

from __future__ import annotations

import json
from typing import Optional

from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID

from intergrax.runtime.nexus.planning.engine_plan_models import (
    DEFAULT_PLANNER_FALLBACK_CLARIFY_QUESTION,
    EngineNextStep,
    EnginePlan,
    PlanIntent,
    PlannerPromptConfig,
)


class EnginePlanJsonParser:
    """Parse LLM planner JSON into a typed :class:`EnginePlan` (no LLM calls)."""

    @staticmethod
    def extract_json_object(raw: str) -> str:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise ValueError("Planner did not return a JSON object.")
        return raw[start : end + 1]

    @classmethod
    def parse(cls, raw: str, *, prompt_config: Optional[PlannerPromptConfig] = None) -> EnginePlan:
        js = cls.extract_json_object(raw)

        try:
            data = json.loads(js)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("Planner output must be a JSON object.")

        def req_bool(key: str) -> bool:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
            value = data[key]
            if isinstance(value, bool):
                return value
            raise ValueError(f"Key '{key}' must be boolean.")

        def req_str(key: str) -> str:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
            value = data[key]
            if isinstance(value, str):
                return value
            raise ValueError(f"Key '{key}' must be string.")

        def req_str_or_null(key: str) -> Optional[str]:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
            value = data[key]
            if value is None:
                return None
            if isinstance(value, str):
                return value.strip()
            raise ValueError(f"Key '{key}' must be string or null.")

        def opt_version_str(key: str, default: str = "1.0") -> str:
            if key not in data:
                return default
            value = data[key]
            if value is None:
                return default
            if isinstance(value, str):
                stripped = value.strip()
                return stripped or default
            if isinstance(value, (int, float)):
                return str(value)
            raise ValueError(f"Key '{key}' must be string, number, or null.")

        version = opt_version_str("version", default="1.0")

        intent_raw = req_str("intent").strip()
        try:
            intent = PlanIntent(intent_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid intent '{intent_raw}'. Allowed: generic|freshness|project_architecture|clarify."
            ) from exc

        next_step_raw = req_str("next_step").strip()
        next_step: Optional[EngineNextStep]
        try:
            next_step = EngineNextStep(next_step_raw)
        except ValueError:
            next_step = None

        reasoning_summary = req_str("reasoning_summary").strip()
        ask_clarify = req_bool("ask_clarifying_question")
        clar_q = req_str_or_null("clarifying_question")

        use_web = req_bool("use_websearch")
        use_ltm = req_bool("use_user_longterm_memory")
        use_rag = req_bool("use_rag")
        use_tools = req_bool("use_tools")
        tool_ids = cls._optional_tool_ids(data)
        if tool_ids:
            use_rag = RAG_RETRIEVE_TOOL_ID in tool_ids or use_rag
            use_web = WEBSEARCH_QUERY_TOOL_ID in tool_ids or use_web
        elif use_rag or use_web:
            tool_ids = []
            if use_rag:
                tool_ids.append(RAG_RETRIEVE_TOOL_ID)
            if use_web:
                tool_ids.append(WEBSEARCH_QUERY_TOOL_ID)

        if intent == PlanIntent.CLARIFY:
            ask_clarify = True
            if not clar_q:
                clar_q = cls.fallback_clarify_question(prompt_config)
            use_web = use_ltm = use_rag = use_tools = False
            tool_ids = []
            next_step = EngineNextStep.CLARIFY
        else:
            if ask_clarify:
                intent = PlanIntent.CLARIFY
                if not clar_q:
                    clar_q = cls.fallback_clarify_question(prompt_config)
                use_web = use_ltm = use_rag = use_tools = False
                tool_ids = []
                next_step = EngineNextStep.CLARIFY
                reasoning_summary = "clarify_required"
            else:
                clar_q = None
                if next_step == EngineNextStep.CLARIFY:
                    next_step = None

        if next_step is None:
            if use_web:
                next_step = EngineNextStep.WEBSEARCH
            elif use_tools:
                next_step = EngineNextStep.TOOLS
            elif use_rag or use_ltm:
                next_step = EngineNextStep.RAG
            else:
                next_step = EngineNextStep.SYNTHESIZE

        return EnginePlan(
            version=version,
            intent=intent,
            reasoning_summary=reasoning_summary,
            ask_clarifying_question=ask_clarify,
            clarifying_question=clar_q,
            next_step=next_step,
            use_websearch=use_web,
            use_user_longterm_memory=use_ltm,
            use_rag=use_rag,
            use_tools=use_tools,
            tool_ids=tool_ids,
        )

    @staticmethod
    def _optional_tool_ids(data: dict) -> list[str]:
        raw = data.get("tool_ids")
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    @staticmethod
    def fallback_clarify_question(prompt_config: Optional[PlannerPromptConfig]) -> str:
        question = DEFAULT_PLANNER_FALLBACK_CLARIFY_QUESTION()
        if prompt_config is not None and prompt_config.fallback_clarify_question:
            question = prompt_config.fallback_clarify_question.strip()
        return question
