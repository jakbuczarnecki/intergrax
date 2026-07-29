# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Catalog tool planning without ``ToolsAgent`` (Phase T-Ops.5)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Union

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.exporters.openai import to_openai_tools
from intergrax.tools.exporters.schema import pydantic_parameters_schema
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.runtime import RegisteredTool
from intergrax.runtime.nexus.tools.tool_planning_config import ToolPlanningConfig
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision


def _registered_tools_for_planning(
    registry: ToolRegistry,
    allowed_tool_ids: Sequence[str] | None,
) -> list[RegisteredTool]:
    if allowed_tool_ids is None:
        return list(registry.list())
    allowed = frozenset(allowed_tool_ids)
    return [item for item in registry.list() if item.contract.tool_id in allowed]


def build_tool_planning_schema(
    registry: ToolRegistry,
    *,
    allowed_tool_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Export registered tools in deterministic lexicographic tool_id order."""
    registered = _registered_tools_for_planning(registry, allowed_tool_ids)
    ordered = sorted(registered, key=lambda item: item.contract.tool_id)
    return to_openai_tools(ordered)


def _expected_tool_ids(
    registry: ToolRegistry,
    allowed_tool_ids: Sequence[str] | None,
) -> frozenset[str]:
    return frozenset(
        item.contract.tool_id
        for item in _registered_tools_for_planning(registry, allowed_tool_ids)
    )


def _validate_prepared_tools_schema(
    prepared_tools_schema: Sequence[Mapping[str, Any]],
    *,
    expected_tool_ids: frozenset[str],
) -> List[Dict[str, Any]]:
    materialized: List[Dict[str, Any]] = [dict(entry) for entry in prepared_tools_schema]
    observed: list[str] = []
    for entry in materialized:
        function = entry.get("function")
        if not isinstance(function, dict):
            raise ValueError("prepared_tools_schema entry missing function object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("prepared_tools_schema entry missing function.name")
        if name in observed:
            raise ValueError(f"duplicate tool id in prepared_tools_schema: {name}")
        observed.append(name)
    observed_set = frozenset(observed)
    if observed_set != expected_tool_ids:
        missing = expected_tool_ids - observed_set
        unexpected = observed_set - expected_tool_ids
        if missing:
            raise ValueError(f"prepared_tools_schema missing expected tools: {sorted(missing)}")
        if unexpected:
            raise ValueError(
                f"prepared_tools_schema contains unexpected tools: {sorted(unexpected)}"
            )
    return materialized


def _build_openai_tools_schema(
    registry: ToolRegistry,
    *,
    allowed_tool_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    return build_tool_planning_schema(registry, allowed_tool_ids=allowed_tool_ids)


def _prune_messages_for_openai(messages: List[ChatMessage]) -> List[ChatMessage]:
    """
    OpenAI requires tool messages only after the last assistant message with tool_calls.
    """
    last_tc_idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if message.role == "assistant" and message.tool_calls:
            last_tc_idx = i
            break

    if last_tc_idx is None:
        return [message for message in messages if message.role in ("system", "user", "assistant")]

    pruned: List[ChatMessage] = []
    for i, message in enumerate(messages):
        if message.role == "tool":
            if i > last_tc_idx:
                pruned.append(message)
        else:
            pruned.append(message)
    return pruned


def _sync_routing_before_tool_planner_llm(
    routing_runtime_config: object | None,
    *,
    run_id: str | None,
) -> None:
    if routing_runtime_config is None:
        return
    from intergrax.runtime.nexus.config import RuntimeConfig
    from intergrax.runtime.nexus.context.routing_snapshot_sync import sync_routing_before_llm_call

    if not isinstance(routing_runtime_config, RuntimeConfig):
        return
    sync_routing_before_llm_call(routing_runtime_config, run_id=run_id)


class ToolPlanningService:
    """Plans tool calls from an LLM + catalog registry (planner-only, no execution)."""

    def __init__(
        self,
        llm: LLMAdapter,
        tools: ToolRegistry,
        *,
        config: Optional[ToolPlanningConfig] = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.cfg = config or ToolPlanningConfig.default()
        self._routing_runtime_config: object | None = None
        self._native_tools = False
        try:
            self._native_tools = bool(self.llm.supports_tools())
        except Exception:
            self._native_tools = False

    def attach_routing_runtime_config(self, config: object) -> None:
        """Wire live routing snapshot refresh before planner LLM calls (M-LLM-X.13.4)."""
        self._routing_runtime_config = config

    def plan_tools(
        self,
        input_data: Union[str, List[ChatMessage]],
        *,
        context: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        allowed_tool_ids: Sequence[str] | None = None,
        run_id: Optional[str] = None,
    ) -> ToolPlanDecision:
        if isinstance(input_data, list):
            base_messages: List[ChatMessage] = list(input_data)
            if context:
                ctx_msg = ChatMessage(
                    role="system",
                    content=self.cfg.system_context_template.format(context=context),
                )
                if base_messages and base_messages[-1].role == "user":
                    messages = base_messages[:-1] + [ctx_msg, base_messages[-1]]
                else:
                    base_messages.append(ctx_msg)
                    messages = base_messages
            else:
                messages = base_messages
        else:
            user_input: str = input_data
            if not user_input:
                raise ValueError("ToolPlanningService.plan_tools requires non-empty input_data.")

            sys = ChatMessage(
                role="system",
                content=self.cfg.system_instructions
                + (
                    f"\n\n{self.cfg.system_context_template.format(context=context)}"
                    if context
                    else ""
                ),
            )
            messages = [sys, ChatMessage(role="user", content=user_input)]

        calls: List[PlannedToolCall] = []

        allowed = frozenset(allowed_tool_ids) if allowed_tool_ids is not None else None

        if self._native_tools:
            llm_result, tool_plan = self.plan_native_round(
                messages,
                allowed_tool_ids=allowed_tool_ids,
                run_id=run_id,
                tool_choice=tool_choice,
            )
            _ = llm_result
            return ToolPlanDecision(
                final_answer=None,
                tool_plan=tool_plan,
                messages=[],
            )

        tools_desc = [
            {
                "name": rt.contract.tool_id,
                "description": rt.contract.description,
                "parameters": pydantic_parameters_schema(rt.contract.input_schema),
            }
            for rt in _registered_tools_for_planning(self.tools, allowed_tool_ids)
        ]

        plan_intro = ChatMessage(
            role="system",
            content=self.cfg.planner_instructions
            + "\nTOOLS=\n"
            + json.dumps(tools_desc, ensure_ascii=False),
        )

        if len(messages) and messages[0].role == "system":
            messages = [messages[0], plan_intro] + messages[1:]
        else:
            messages = [plan_intro] + messages

        _sync_routing_before_tool_planner_llm(
            self._routing_runtime_config,
            run_id=run_id,
        )
        plan_response = self.llm.generate_messages(
            messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_answer_tokens,
            run_id=run_id,
        )
        plan_text = plan_response.content

        plan_obj = None
        try:
            start, end = plan_text.find("{"), plan_text.rfind("}")
            if start != -1 and end > start:
                plan_obj = json.loads(plan_text[start : end + 1])
        except Exception:
            plan_obj = None

        if plan_obj and "call_tool" in plan_obj:
            call = plan_obj["call_tool"]
            name = call.get("name")
            if allowed is not None and name not in allowed:
                return ToolPlanDecision(
                    final_answer=None,
                    tool_plan=ToolCallPlan(calls=calls),
                    messages=[],
                )
            args = call.get("arguments", {}) or {}

            registered = self.tools.get(name)
            contract = registered.contract
            validated = contract.input_schema.model_validate(args)

            calls.append(
                PlannedToolCall(
                    step_id="tool",
                    tool_id=name,
                    input=validated,
                )
            )

        return ToolPlanDecision(
            final_answer=None,
            tool_plan=ToolCallPlan(calls=calls),
            messages=[],
        )

    def plan_native_round(
        self,
        messages: List[ChatMessage],
        *,
        allowed_tool_ids: Sequence[str] | None = None,
        run_id: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        prepared_tools_schema: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[LLMAdapterResponse, ToolCallPlan]:
        """One native LLM tool round — used by TOOL-ENG-6 multi-iteration loop."""
        if not self._native_tools:
            raise ValueError("plan_native_round requires an LLM adapter with native tool support")

        allowed = frozenset(allowed_tool_ids) if allowed_tool_ids is not None else None
        if prepared_tools_schema is not None:
            expected = _expected_tool_ids(self.tools, allowed_tool_ids)
            tools_schema = _validate_prepared_tools_schema(
                prepared_tools_schema,
                expected_tool_ids=expected,
            )
        else:
            tools_schema = _build_openai_tools_schema(
                self.tools,
                allowed_tool_ids=allowed_tool_ids,
            )
        pruned = _prune_messages_for_openai(list(messages))
        effective_tool_choice = tool_choice if tool_choice is not None else "auto"

        _sync_routing_before_tool_planner_llm(
            self._routing_runtime_config,
            run_id=run_id,
        )
        result = self.llm.generate_with_tools(
            pruned,
            tools_schema,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_answer_tokens,
            tool_choice=effective_tool_choice,
            run_id=run_id,
        )

        calls: List[PlannedToolCall] = []
        for tc in result.tool_calls:
            name = tc.name
            if allowed is not None and name not in allowed:
                continue
            args_json = tc.arguments_json or "{}"
            try:
                args = json.loads(args_json)
            except Exception:
                args = {}
            registered = self.tools.get(name)
            contract = registered.contract
            validated = contract.input_schema.model_validate(args)
            calls.append(
                PlannedToolCall(
                    step_id="tool",
                    tool_id=name,
                    input=validated,
                )
            )

        return result, ToolCallPlan(calls=calls)
