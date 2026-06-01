# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, List, Optional, Union, Type

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageTracker
from intergrax.logging import IntergraxLogging
from intergrax.memory.conversational_memory import ConversationalMemory
from intergrax.llm.messages import ChatMessage
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.exporters.openai import to_openai_tools
from intergrax.tools.exporters.schema import pydantic_parameters_schema
from intergrax.tools.registry import ToolRegistry
from intergrax.tools._shared.output import limit_tool_output

# Backward-compatible alias (deprecated)
_limit_tool_output = limit_tool_output

logger = IntergraxLogging.get_logger(__name__, component="tools")


# =====================================================================
# PROMPTS
# =====================================================================

def PLANNER_PROMPT() -> str:
    registry = YamlPromptRegistry.create_default(load=True)
    return registry.resolve_localized("tools_agent_planner").system


def SYSTEM_PROMPT() -> str:
    registry = YamlPromptRegistry.create_default(load=True)
    return registry.resolve_localized("tools_agent_system").system


def SYSTEM_CONTEXT_TEMPLATE() -> str:
    """
    Returns legacy-compatible template containing `{context}` placeholder.
    Formatting is done later via `.format(context=...)`.
    """
    registry = YamlPromptRegistry.create_default(load=True)
    localized = registry.resolve_localized("tools_agent_context")
    return localized.user_template or ""


# =====================================================================
# RESULT MODELS
# =====================================================================

@dataclass(slots=True)
class ToolTrace:
    tool: str
    args: Dict[str, Any]
    output_preview: str
    output: Any


@dataclass(slots=True)
class ToolsAgentRunResult:
    """Completed tools-agent loop result (distinct from ``contracts.agent_execution_result``)."""

    final_answer: str
    tool_traces: List[ToolTrace]
    messages: List[ChatMessage]
    output_structure: Optional[Any]


# =====================================================================
# CONFIG
# =====================================================================

class ToolsAgentConfig:
    temperature: Optional[float] = None,
    max_answer_tokens: Optional[int] = None
    max_tool_iters: int = 6    
    system_instructions: str = SYSTEM_PROMPT()
    system_context_template: str = SYSTEM_CONTEXT_TEMPLATE()
    planner_instructions: str = PLANNER_PROMPT()


# =====================================================================
# HELPERS
# =====================================================================

def _maybe_import_pydantic_base() -> Optional[type]:
    try:
        from pydantic import BaseModel  # type: ignore
        return BaseModel
    except Exception:
        return None


def _instantiate_output_model(model_cls: Type, payload: Any) -> Any:
    """
    Creates an instance of output_model:
    - If it is Pydantic v2/v1 → model_cls(**payload)
    - Otherwise: model_cls(**payload) (duck-typing)
    """
    if payload is None:
        return None

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return None

    if not isinstance(payload, dict):
        try:
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump()
            elif hasattr(payload, "dict"):
                payload = payload.dict()
            else:
                return None
        except Exception:
            return None

    try:
        return model_cls(**payload)
    except Exception:
        try:
            base = _maybe_import_pydantic_base()
            if base and isinstance(model_cls, type) and issubclass(model_cls, base):
                return model_cls.model_validate(payload)  # v2 compat
        except Exception:
            pass
        return None


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Tolerant extraction of the first JSON object from text."""
    if not text:
        return None
    try:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
    except Exception:
        return None
    return None


def _build_openai_tools_schema(tools: ToolRegistry) -> List[Dict[str, Any]]:
    """Build OpenAI-compatible 'tools' schema from the runtime ToolRegistry."""
    return to_openai_tools(tools)


# =====================================================================
# TOOLS AGENT
# =====================================================================

class ToolsAgent:
    def __init__(
        self,
        llm: LLMAdapter,
        tools: ToolRegistry,
        *,
        memory: Optional[ConversationalMemory] = None,
        config: Optional[ToolsAgentConfig] = None,
    ):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.cfg = config or ToolsAgentConfig()

        # Does the LLM support native tools (OpenAI) or a JSON planner (Ollama)?
        self._native_tools = False
        try:
            self._native_tools = bool(self.llm.supports_tools())
        except Exception:
            self._native_tools = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _prune_messages_for_openai(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        OpenAI requires that tool messages appear only in response
        to the immediately preceding assistant message with tool_calls.
        Therefore, remove all older 'tool' messages and keep only those
        that appear after the *last* assistant with tool_calls.
        """
        last_tc_idx: Optional[int] = None
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if m.role == "assistant" and getattr(m, "tool_calls", None):
                last_tc_idx = i
                break

        if last_tc_idx is None:
            return [m for m in messages if m.role in ("system", "user", "assistant")]

        pruned: List[ChatMessage] = []
        for i, m in enumerate(messages):
            if m.role == "tool":
                if i > last_tc_idx:
                    pruned.append(m)
            else:
                pruned.append(m)
        return pruned

    def _build_output_structure(
        self,
        output_model: Optional[Type],
        answer_text: str,
        tool_traces: List[ToolTrace],
    ) -> Any:
        """
        Strategy:
        1) If there are tool_traces -> prefer last full output
        2) Otherwise -> try to extract JSON from answer_text
        3) Map to output_model (Pydantic / regular class)
        """
        if not output_model:
            return None

        if tool_traces:
            last = tool_traces[-1]
            full = last.output
            if full is not None:
                obj = _instantiate_output_model(output_model, full)
                if obj is not None:
                    return obj

            preview = last.output_preview
            if preview:
                try:
                    obj = _instantiate_output_model(output_model, json.loads(preview))
                    if obj is not None:
                        return obj
                except Exception:
                    pass

        data = _extract_json_from_text(answer_text)
        if data is not None:
            obj = _instantiate_output_model(output_model, data)
            if obj is not None:
                return obj

        return None

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def plan_tools(
        self,
        input_data: Union[str, List[ChatMessage]],
        *,
        context: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> ToolPlanDecision:
        """
        Planner-only mode.

        Uses LLM to decide WHICH tools to call and with WHAT arguments,
        but does NOT execute them.
        """
        from intergrax.runtime.nexus.tools.tool_planning_service import ToolPlanningService

        service = ToolPlanningService(
            self.llm,
            self.tools,
            config=self.cfg,
        )
        return service.plan_tools(
            input_data,
            context=context,
            tool_choice=tool_choice,
            run_id=run_id,
        )

    def run(
        self,
        input_data: Union[str, List[ChatMessage]],
        *,
        context: Optional[str] = None,
        stream: bool = False,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        output_model: Optional[Type] = None,
        run_id: Optional[str] = None,
        llm_usage_tracker: Optional[LLMUsageTracker] = None,
    ) -> ToolsAgentRunResult:
        """
        High-level tools orchestration entrypoint.

        Notes:
        - Tool execution requires run_id (ToolExecutionRequest requires it).
        - ToolsAgent is allowed to execute tools standalone (no runtime enforcement).
        """

        # --- Branch 1: caller provides full messages context (ChatGPT-like mode) ---
        if isinstance(input_data, list):
            base_messages: List[ChatMessage] = list(input_data)

            has_system = any(m.role == "system" for m in base_messages)
            if not has_system:
                base_messages.insert(
                    0,
                    ChatMessage(role="system", content=self.cfg.system_instructions),
                )

            if context:
                ctx_msg = ChatMessage(
                    role="system",
                    content=self.cfg.system_context_template.format(context=context),
                )
                if base_messages and base_messages[-1].role == "user":
                    base_messages = base_messages[:-1] + [ctx_msg, base_messages[-1]]
                else:
                    base_messages.append(ctx_msg)

            messages: List[ChatMessage] = base_messages

        # --- Branch 2: legacy mode – single user_input string ---
        else:
            user_input: str = input_data
            if not user_input:
                raise ValueError("ToolsAgent.run requires non-empty input_data.")

            if self.memory:
                self.memory.add("user", user_input)
                messages = self.memory.get_for_model(native_tools=self._native_tools)

                if not any(m.role == "system" for m in messages):
                    messages.insert(
                        0,
                        ChatMessage(role="system", content=self.cfg.system_instructions),
                    )

                if context:
                    ctx_msg = ChatMessage(
                        role="system",
                        content=self.cfg.system_context_template.format(context=context),
                    )
                    if messages and messages[-1].role == "user":
                        messages = messages[:-1] + [ctx_msg, messages[-1]]
                    else:
                        messages.append(ctx_msg)
            else:
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

        iterations = 0
        tool_traces: List[ToolTrace] = []
        last_call_fp = None  # anti-loop

        # ===== BRANCH A: Native tools (OpenAI, etc.) =====
        if self._native_tools:
            tools_schema = _build_openai_tools_schema(self.tools)

            while iterations < self.cfg.max_tool_iters:
                iterations += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[intergraxToolsAgent] Iteration {iterations} (native tools)")

                messages = self._prune_messages_for_openai(messages)

                effective_tool_choice = tool_choice if tool_choice is not None else "auto"

                # --- LLM call ---
                if stream:
                    chunks: List[Dict[str, Any]] = []
                    for ev in self.llm.stream_with_tools(
                        messages,
                        tools_schema,
                        temperature=self.cfg.temperature,
                        max_tokens=self.cfg.max_answer_tokens,
                        tool_choice=effective_tool_choice,
                        run_id=run_id,
                    ):
                        chunks.append(ev)
                    result = chunks[-1] if chunks else {"content": "", "tool_calls": []}
                else:
                    result = self.llm.generate_with_tools(
                        messages,
                        tools_schema,
                        temperature=self.cfg.temperature,
                        max_tokens=self.cfg.max_answer_tokens,
                        tool_choice=effective_tool_choice,
                        run_id=run_id,
                    )

                content = result.get("content") or ""
                tool_calls = result.get("tool_calls") or []

                messages.append(
                    ChatMessage(role="assistant", content=content, tool_calls=tool_calls)
                )

                # --- no tools → final ---
                if not tool_calls:
                    if content.strip():
                        if self.memory and messages is not None:
                            self.memory.add("assistant", content)
                        output_obj = self._build_output_structure(
                            output_model, content, tool_traces
                        )
                        return ToolsAgentRunResult(
                            final_answer=content,
                            tool_traces=tool_traces,
                            messages=messages,
                            output_structure=output_obj,
                        )

                    final = "(no tool call, empty content)"
                    if self.memory:
                        self.memory.add("assistant", final)
                    output_obj = self._build_output_structure(
                        output_model, final, tool_traces
                    )
                    return ToolsAgentRunResult(
                        final_answer=final,
                        tool_traces=tool_traces,
                        messages=messages,
                        output_structure=output_obj,
                    )

                # --- execute tools ---
                if run_id is None:
                    raise ValueError("ToolsAgent.run: run_id is required for tool execution.")

                for tool_idx, tc in enumerate(tool_calls):
                    fn = tc.get("function") or {}
                    name = fn.get("name") or tc.get("name")
                    call_id = tc.get("id")
                    args_json = fn.get("arguments") or tc.get("arguments") or "{}"

                    try:
                        args = json.loads(args_json)
                    except Exception:
                        args = {}

                    registered = self.tools.get(name)
                    contract = registered.contract
                    handler = registered.handler

                    validated_model = contract.input_schema.model_validate(args)
                    validated_dict = validated_model.model_dump()

                    fp = (name, json.dumps(validated_dict, sort_keys=True))
                    if fp == last_call_fp:
                        final = "Stopped repeated identical tool call."
                        output_obj = self._build_output_structure(
                            output_model, final, tool_traces
                        )
                        return ToolsAgentRunResult(
                            final_answer=final,
                            tool_traces=tool_traces,
                            messages=messages,
                            output_structure=output_obj,
                        )
                    last_call_fp = fp

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[intergraxToolsAgent] Calling tool: {name}({validated_dict})")

                    step_id = f"tools_agent/{iterations}/{tool_idx}"

                    try:
                        req = ToolExecutionRequest(
                            run_id=run_id,
                            step_id=step_id,
                            tool_id=name,
                            input=validated_model,
                        )
                        out_model = handler.execute(req)
                        out = out_model.model_dump()
                    except Exception as e:
                        out = f"[{name}] ERROR: {e}"

                    safe_out = _limit_tool_output(json.dumps(out, ensure_ascii=False))
                    tool_traces.append(
                        ToolTrace(
                            tool=name,
                            args=validated_dict,
                            output_preview=safe_out[:400],
                            output=out,
                        )
                    )

                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=json.dumps(
                                {"tool_name": name, "result": safe_out},
                                ensure_ascii=False,
                            ),
                            tool_call_id=call_id,
                            name=name,
                        )
                    )

                continue

            final = "Reached tool iteration limit."
            if self.memory:
                self.memory.add("assistant", final)
            output_obj = self._build_output_structure(output_model, final, tool_traces)
            return ToolsAgentRunResult(
                final_answer=final,
                tool_traces=tool_traces,
                messages=messages,
                output_structure=output_obj,
            )

        # ===== BRANCH B: JSON planner (e.g., Ollama) =====
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

        while iterations < self.cfg.max_tool_iters:
            iterations += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[intergraxToolsAgent] Iteration {iterations} (planner)")

            plan_text = self.llm.generate_messages(
                messages,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_answer_tokens,
                run_id=run_id,
            )

            plan_obj = None
            try:
                start, end = plan_text.find("{"), plan_text.rfind("}")
                if start != -1 and end > start:
                    plan_obj = json.loads(plan_text[start : end + 1])
            except Exception:
                plan_obj = None

            if not plan_obj:
                if self.memory:
                    self.memory.add("assistant", plan_text)
                output_obj = self._build_output_structure(
                    output_model, plan_text, tool_traces
                )
                return ToolsAgentRunResult(
                    final_answer=plan_text,
                    tool_traces=tool_traces,
                    messages=messages,
                    output_structure=output_obj,
                )

            if "final_answer" in plan_obj:
                final = str(plan_obj["final_answer"])
                if self.memory:
                    self.memory.add("assistant", final)
                output_obj = self._build_output_structure(output_model, final, tool_traces)
                return ToolsAgentRunResult(
                    final_answer=final,
                    tool_traces=tool_traces,
                    messages=messages,
                    output_structure=output_obj,
                )

            if "call_tool" in plan_obj:
                if run_id is None:
                    raise ValueError("ToolsAgent.run: run_id is required for tool execution.")

                call = plan_obj["call_tool"]
                name = call.get("name")
                args = call.get("arguments", {}) or {}

                registered = self.tools.get(name)
                contract = registered.contract
                handler = registered.handler

                validated_model = contract.input_schema.model_validate(args)
                validated_dict = validated_model.model_dump()

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[intergraxToolsAgent] Calling tool: {name}({validated_dict})")

                step_id = f"tools_agent/{iterations}/0"

                try:
                    req = ToolExecutionRequest(
                        run_id=run_id,
                        step_id=step_id,
                        tool_id=name,
                        input=validated_model,
                    )
                    out_model = handler.execute(req)
                    out = out_model.model_dump()
                except Exception as e:
                    out = f"[{name}] ERROR: {e}"

                safe_out = _limit_tool_output(json.dumps(out, ensure_ascii=False))
                tool_traces.append(
                    ToolTrace(
                        tool=name,
                        args=validated_dict,
                        output_preview=safe_out[:400],
                        output=out,
                    )
                )

                messages.append(
                    ChatMessage(
                        role="tool",
                        content=json.dumps(
                            {"tool_name": name, "result": safe_out},
                            ensure_ascii=False,
                        ),
                    )
                )

                messages.append(
                    ChatMessage(
                        role="user",
                        content="Use the TOOL RESULT above. Continue and return final_answer as JSON.",
                    )
                )
                continue

        final = "Reached planner iteration limit."
        if self.memory:
            self.memory.add("assistant", final)
        output_obj = self._build_output_structure(output_model, final, tool_traces)
        return ToolsAgentRunResult(
            final_answer=final,
            tool_traces=tool_traces,
            messages=messages,
            output_structure=output_obj,
        )


_DEPRECATED_TOOL_AGENT_ALIASES = {
    "AgentDecision": ToolPlanDecision,
    "AgentExecutionResult": ToolsAgentRunResult,
}


def __getattr__(name: str):
    import warnings

    if name in _DEPRECATED_TOOL_AGENT_ALIASES:
        warnings.warn(
            f"intergrax.tools.tools_agent.{name} is deprecated; "
            f"use { _DEPRECATED_TOOL_AGENT_ALIASES[name].__name__} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_TOOL_AGENT_ALIASES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
