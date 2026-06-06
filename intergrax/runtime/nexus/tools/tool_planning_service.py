# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Catalog tool planning without ``ToolsAgent`` (Phase T-Ops.5)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.exporters.openai import to_openai_tools
from intergrax.tools.exporters.schema import pydantic_parameters_schema
from intergrax.tools.registry import ToolRegistry
from intergrax.runtime.nexus.tools.tool_planning_config import ToolPlanningConfig
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision


def _build_openai_tools_schema(tools: ToolRegistry) -> List[Dict[str, Any]]:
    return to_openai_tools(tools)


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
        self._native_tools = False
        try:
            self._native_tools = bool(self.llm.supports_tools())
        except Exception:
            self._native_tools = False

    def plan_tools(
        self,
        input_data: Union[str, List[ChatMessage]],
        *,
        context: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
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

        if self._native_tools:
            tools_schema = _build_openai_tools_schema(self.tools)
            messages = _prune_messages_for_openai(messages)
            effective_tool_choice = tool_choice if tool_choice is not None else "auto"

            result = self.llm.generate_with_tools(
                messages,
                tools_schema,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_answer_tokens,
                tool_choice=effective_tool_choice,
                run_id=run_id,
            )

            for tc in result.tool_calls:
                name = tc.name
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

            return ToolPlanDecision(
                final_answer=None,
                tool_plan=ToolCallPlan(calls=calls),
                messages=[],
            )

        tools_desc = [
            {
                "name": rt.contract.tool_id,
                "description": rt.contract.description,
                "parameters": pydantic_parameters_schema(rt.contract.input_schema),
            }
            for rt in self.tools._tools.values()
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
