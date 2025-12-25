# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Union, Type

from intergrax.llm_adapters.llm_adapter import LLMAdapter
from intergrax.memory.conversational_memory import ConversationalMemory
from intergrax.llm.messages import ChatMessage
from .tools_base import ToolRegistry, _limit_tool_output


PLANNER_PROMPT = (
                    "You do not have native tool-calling.\n"
                    "At each step, reply ONLY with strict JSON:\n"
                    '{\"call_tool\": {\"name\": \"<tool_name>\", \"arguments\": {...}}} '
                    'or {\"final_answer\": \"<text>\"}.\n'
                    "Use the declared TOOLS catalog to choose function name and arguments.\n"
                    "If tool result is insufficient, you may call another tool.\n"
                    "Never include commentary outside JSON."
                )

SYSTEM_PROMPT = (
        "You are a capable assistant. Use tools when helpful. "
        "If you call a tool, do not fabricate results—wait for tool outputs."
    )

SYSTEM_CONTEXT_TEMPLATE = "Session context:\n{context}"


class ToolsAgentConfig:
    temperature: float = 0.2
    max_answer_tokens: Optional[int] = None
    max_tool_iters: int = 6
    system_instructions: str = SYSTEM_PROMPT
    system_context_template: str = SYSTEM_CONTEXT_TEMPLATE
    planner_instructions : str = PLANNER_PROMPT


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
    # If payload is a JSON string → decode
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return None
    # If payload is not a dict but has .dict/.model_dump → use it
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

    # Pydantic / regular class with **kwargs
    try:
        return model_cls(**payload)
    except Exception:
        # Another attempt: if a pydantic model needs type conversion
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


class ToolsAgent:
    def __init__(
        self,
        llm: LLMAdapter,
        tools: ToolRegistry,
        *,
        memory: Optional[ConversationalMemory] = None,
        config: Optional[ToolsAgentConfig] = None,
        verbose: bool = False,
    ):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.cfg = config or ToolsAgentConfig()
        self.verbose = verbose

        # Does the LLM support native tools (OpenAI) or a JSON planner (Ollama)?
        self._native_tools = False
        if hasattr(self.llm, "supports_tools"):
            try:
                self._native_tools = bool(self.llm.supports_tools())
            except Exception:
                self._native_tools = False        
        
    # ----- helpers -----

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
            # No active tool_calls → remove all 'tool' messages
            return [m for m in messages if m.role in ("system", "user", "assistant")]

        pruned: List[ChatMessage] = []
        for i, m in enumerate(messages):
            if m.role == "tool":
                # Keep only tool messages after the last assistant.tool_calls
                if i > last_tc_idx:
                    pruned.append(m)
            else:
                pruned.append(m)
        return pruned

    def _build_output_structure(
        self,
        output_model: Optional[Type],
        answer_text: str,
        tool_traces: List[Dict[str, Any]],
    ) -> Any:
        """
        Strategy:
        1) If there are tool_traces and they have 'output' (full result) → use the last one.
        2) If there were no tools → try to extract JSON from answer_text.
        3) Map to output_model (Pydantic / regular class).
        """
        if not output_model:
            return None

        # 1) Prefer tools (full result — see change in append tool trace)
        if tool_traces:
            last = tool_traces[-1]
            full = last.get("output")  # full, not truncated result
            if full is not None:
                obj = _instantiate_output_model(output_model, full)
                if obj is not None:
                    return obj

            # If there's no 'output' but there is output_preview → try JSON
            preview = last.get("output_preview")
            if preview:
                try:
                    obj = _instantiate_output_model(output_model, json.loads(preview))
                    if obj is not None:
                        return obj
                except Exception:
                    pass

        # 2) Without tools: try to extract JSON from text
        data = _extract_json_from_text(answer_text)
        if data is not None:
            obj = _instantiate_output_model(output_model, data)
            if obj is not None:
                return obj

        return None
        

    # ----- PUBLIC API -----
    def run(
        self,
        input_data: Union[str, List[ChatMessage]],
        *,
        context: Optional[str] = None,
        stream: bool = False,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        output_model: Optional[Type] = None,
        run_id:Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        High-level tools orchestration entrypoint.

        Modes:
        - If `input_data` is a List[ChatMessage]:
            Use this list as the base conversation context (already built by the caller,
            e.g. runtime engine with RAG + websearch + history). Optionally inject
            system instructions and context.
        - If `input_data` is a str:
            Fall back to legacy mode with single user_input + optional memory.

        Returns:
            Dict with keys:
                - "answer": final answer from tools loop (or planner),
                - "tool_traces": list of executed tools (name, args, output, preview),
                - "messages": final message list used by the tools loop,
                - "output_structure": optional structured output (if output_model provided).
        """

        # --- Branch 1: caller provides full messages context (ChatGPT-like mode) ---
        if isinstance(input_data, list):
            # Treat input_data as List[ChatMessage]
            base_messages: List[ChatMessage] = list(input_data)

            # Ensure there is at least one system message with core instructions.
            has_system = any(m.role == "system" for m in base_messages)
            if not has_system:
                base_messages.insert(
                    0,
                    ChatMessage(role="system", content=self.cfg.system_instructions),
                )

            # Inject textual session context (RAG + websearch) as extra system message.
            if context:
                ctx_msg = ChatMessage(
                    role="system",
                    content=self.cfg.system_context_template.format(context=context),
                )
                # Wkładamy go tuż przed ostatnią wiadomością user, jeśli jest,
                # żeby model widział kontekst „tuż przed pytaniem”.
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
                        ChatMessage(
                            role="system",
                            content=self.cfg.system_instructions,
                        ),
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
        tool_traces: List[Dict[str, Any]] = []
        last_call_fp = None  # anti-loop

        # ===== BRANCH A: Native tools (OpenAI, etc.) =====
        if self._native_tools:
            tools_schema = self.tools.to_openai_tools()

            while iterations < self.cfg.max_tool_iters:
                iterations += 1
                if self.verbose:
                    print(f"[intergraxToolsAgent] Iteration {iterations} (native tools)")

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
                        run_id=run_id
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
                        run_id=run_id
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
                        return {
                            "answer": content,
                            "tool_traces": tool_traces,
                            "messages": messages,
                            "output_structure": output_obj,
                        }
                    final = "(no tool call, empty content)"
                    if self.memory:
                        self.memory.add("assistant", final)
                    output_obj = self._build_output_structure(
                        output_model, final, tool_traces
                    )
                    return {
                        "answer": final,
                        "tool_traces": tool_traces,
                        "messages": messages,
                        "output_structure": output_obj,
                    }

                # --- execute tools ---
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    name = fn.get("name") or tc.get("name")
                    call_id = tc.get("id")
                    args_json = fn.get("arguments") or tc.get("arguments") or "{}"

                    try:
                        args = json.loads(args_json)
                    except Exception:
                        args = {}

                    tool = self.tools.get(name)
                    validated = tool.validate_args(args)

                    fp = (name, json.dumps(validated, sort_keys=True))
                    if fp == last_call_fp:
                        final = "Stopped repeated identical tool call."
                        output_obj = self._build_output_structure(
                            output_model, final, tool_traces
                        )
                        return {
                            "answer": final,
                            "tool_traces": tool_traces,
                            "messages": messages,
                            "output_structure": output_obj,
                        }
                    last_call_fp = fp

                    if self.verbose:
                        print(
                            f"[intergraxToolsAgent] Calling tool: {name}({validated})"
                        )

                    try:
                        out = tool.run(**validated)  # full result
                    except Exception as e:
                        out = f"[{name}] ERROR: {e}"

                    safe_out = _limit_tool_output(json.dumps(out, ensure_ascii=False))
                    tool_traces.append(
                        {
                            "tool": name,
                            "args": validated,
                            "output_preview": safe_out[:400],
                            "output": out,
                        }
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
            output_obj = self._build_output_structure(
                output_model, final, tool_traces
            )
            return {
                "answer": final,
                "tool_traces": tool_traces,
                "messages": messages,
                "output_structure": output_obj,
            }

        # ===== BRANCH B: JSON planner (e.g., Ollama) =====
        tools_desc = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.get_parameters(),
            }
            for t in self.tools.list()
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
            if self.verbose:
                print(f"[intergraxToolsAgent] Iteration {iterations} (planner)")

            plan_text = self.llm.generate_messages(
                messages,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_answer_tokens,
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
                return {
                    "answer": plan_text,
                    "tool_traces": tool_traces,
                    "messages": messages,
                    "output_structure": output_obj,
                }

            if "final_answer" in plan_obj:
                final = str(plan_obj["final_answer"])
                if self.memory:
                    self.memory.add("assistant", final)
                output_obj = self._build_output_structure(
                    output_model, final, tool_traces
                )
                return {
                    "answer": final,
                    "tool_traces": tool_traces,
                    "messages": messages,
                    "output_structure": output_obj,
                }

            if "call_tool" in plan_obj:
                call = plan_obj["call_tool"]
                name = call.get("name")
                args = call.get("arguments", {}) or {}
                tool = self.tools.get(name)
                validated = tool.validate_args(args)

                if self.verbose:
                    print(
                        f"[intergraxToolsAgent] Calling tool: {name}({validated})"
                    )

                try:
                    out = tool.run(**validated)
                except Exception as e:
                    out = f"[{name}] ERROR: {e}"

                safe_out = _limit_tool_output(json.dumps(out, ensure_ascii=False))
                tool_traces.append(
                    {
                        "tool": name,
                        "args": validated,
                        "output_preview": safe_out[:400],
                        "output": out,
                    }
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
        return {
            "answer": final,
            "tool_traces": tool_traces,
            "messages": messages,
            "output_structure": output_obj,
        }


