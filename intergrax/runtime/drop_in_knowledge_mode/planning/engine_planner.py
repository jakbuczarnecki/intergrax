# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
import pprint
from typing import Any, Dict, List, Optional

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.llm_adapter import LLMAdapter
from intergrax.runtime.drop_in_knowledge_mode.config import RuntimeConfig
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeRequest


# -----------------------------
# Typed plan schema
# -----------------------------

@dataclass(frozen=True)
class PlanRetrieval:
    enabled: bool
    top_k: int
    max_chars: int

@dataclass(frozen=True)
class PlanTools:
    enabled: bool
    goal: str  # e.g. "none" | "execute_actions" | "compute" | "call_external_apis"
    tool_choice_hint: Optional[str]  # e.g. "auto" / tool name (optional)

@dataclass(frozen=False)
class EnginePlan:
    version: str
    intent: str                 # short: "answer_question", "debug_code", "write_doc", ...
    reasoning_summary: str       # high-level, no CoT
    ask_clarifying_question: bool
    clarifying_question: Optional[str]

    use_rag: PlanRetrieval
    use_user_longterm_memory: PlanRetrieval
    use_attachments: PlanRetrieval
    use_websearch: PlanRetrieval
    use_tools: PlanTools

    parallel_groups: List[List[str]]  # e.g. [["attachments","rag","websearch"],["tools"]]
    final_answer_style: str           # e.g. "concise_technical", "step_by_step", "long_form"

    debug: Dict[str, Any] = field(default_factory=dict)


    def print_pretty(self):
        pprint.pprint({
        "version": self.version,
        "intent": self.intent,
        "reasoning_summary": self.reasoning_summary,
        "ask_clarifying_question": self.ask_clarifying_question,
        "clarifying_question": self.clarifying_question,
        "use_rag": self.use_rag,
        "use_user_longterm_memory": self.use_user_longterm_memory,
        "use_attachments": self.use_attachments,
        "use_websearch": self.use_websearch,
        "use_tools": self.use_tools,
        "parallel_groups": self.parallel_groups,
        "final_answer_style": self.final_answer_style,
        "debug": self.debug,
    })


class EnginePlanner:
    """
    LLM-based planner that outputs a typed EnginePlan.

    IMPORTANT:
    - No heuristics: the LLM decides based on PlannerInput + capabilities.
    - Output must be JSON only (we parse & validate).
    """

    def __init__(self, *, llm_adapter: LLMAdapter) -> None:
        self._llm = llm_adapter

    async def plan(
        self,
        *,
        req: RuntimeRequest,
        state: RuntimeState,
        config: RuntimeConfig,
        run_id: Optional[str] = None,
    ) -> EnginePlan:
        messages = self._build_prompt(req=req, state=state, config=config)

        raw = self._llm.generate_messages(messages, run_id=run_id)
        if inspect.iscoroutine(raw):
            raw = await raw
        if not isinstance(raw, str):
            raw = str(raw)

        plan_json = self._extract_json_object(raw)
        plan_dict = json.loads(plan_json)

        plan = self._parse_plan(plan_dict)

        # Minimal execution-safety validation (no heuristics)
        self._validate_against_capabilities(plan=plan, state=state)

        plan.debug = {
            "planner_raw_len": len(raw),
            "planner_raw_preview": raw[:400],
            "planner_raw_tail_preview": raw[-400:],
            "planner_json_len": len(plan_json),
        }

        return plan

    def _validate_against_capabilities(self, *, plan: EnginePlan, state: RuntimeState) -> None:
        if plan.use_rag.enabled and not state.cap_rag_available:
            raise ValueError("Planner produced use_rag.enabled=true but RAG is not available.")
        if plan.use_user_longterm_memory.enabled and not state.cap_user_ltm_available:
            raise ValueError("Planner produced use_user_longterm_memory.enabled=true but user LTM is not available.")
        if plan.use_attachments.enabled and not state.cap_attachments_available:
            raise ValueError("Planner produced use_attachments.enabled=true but attachments retrieval is not available.")
        if plan.use_websearch.enabled and not state.cap_websearch_available:
            raise ValueError("Planner produced use_websearch.enabled=true but websearch is not available.")
        if plan.use_tools.enabled and not state.cap_tools_available:
            raise ValueError("Planner produced use_tools.enabled=true but tools are not available.")

    # -----------------------------
    # Prompting
    # -----------------------------

    def _build_prompt(
        self,
        *,
        req: RuntimeRequest,
        state: RuntimeState,
        config: RuntimeConfig,  # kept for signature consistency (may be used later)
    ) -> List[ChatMessage]:
        # Schema with explicit defaults (anchors the model to safe defaults).
        schema = {
            "version": "1.0",
            "intent": "string",
            "reasoning_summary": "string (high-level, no chain-of-thought)",
            "ask_clarifying_question": "boolean",
            "clarifying_question": "string|null",

            # Retrieval blocks: always present; default OFF
            "use_rag": {"enabled": False, "top_k": 0, "max_chars": 0},
            "use_user_longterm_memory": {"enabled": False, "top_k": 0, "max_chars": 0},
            "use_attachments": {"enabled": False, "top_k": 0, "max_chars": 0},
            "use_websearch": {"enabled": False, "top_k": 0, "max_chars": 0},

            # Tools block: always present; default OFF
            "use_tools": {"enabled": False, "goal": "", "tool_choice_hint": None},

            # Execution hints
            "parallel_groups": [],  # list[list[string]]
            "final_answer_style": "string",
        }

        # Capabilities must be computed by runtime and stored in state (strong typing).
        rag_av = bool(state.cap_rag_available)
        user_ltm_av = bool(state.cap_user_ltm_available)
        att_av = bool(state.cap_attachments_available)
        web_av = bool(state.cap_websearch_available)
        tools_av = bool(state.cap_tools_available)

        sys_lines: List[str] = []
        sys_lines.append("You are an execution planner for Intergrax Drop-In Knowledge Runtime.")
        sys_lines.append("Your job: produce an execution plan in STRICT JSON matching the schema below.")
        sys_lines.append("Return JSON only. No prose, no markdown, no comments.")
        sys_lines.append("Do NOT include chain-of-thought. Provide only a short high-level reasoning_summary.")
        sys_lines.append("Do NOT add extra keys that are not in the schema. Output must be a single JSON object.")
        sys_lines.append("")

        sys_lines.append("Capabilities (only plan steps that are available):")
        sys_lines.append(f"- rag_available: {rag_av}")
        sys_lines.append(f"- user_ltm_available: {user_ltm_av}")
        sys_lines.append(f"- attachments_available: {att_av}")
        sys_lines.append(f"- websearch_available: {web_av}")
        sys_lines.append(f"- tools_available: {tools_av}")
        sys_lines.append("")

        sys_lines.append("Hard rules (MUST FOLLOW):")
        sys_lines.append("- JSON typing rules:")
        sys_lines.append("  - All integers must be JSON numbers (no quotes). Example: top_k: 5, max_chars: 1200.")
        sys_lines.append("  - Always include all retrieval objects: use_rag, use_user_longterm_memory, use_attachments, use_websearch.")
        sys_lines.append("  - For enabled=false, top_k and max_chars MUST be 0 (integers).")
        sys_lines.append("  - For use_tools.enabled=false, still include goal as empty string and tool_choice_hint as null.")
        sys_lines.append("- Capability gating:")
        sys_lines.append("  - If a capability is false, you must set the corresponding use_* .enabled = false (and numeric fields to 0).")
        sys_lines.append("- Clarifying mode:")
        sys_lines.append("  - If ask_clarifying_question=true, then ALL use_* must be enabled=false and parallel_groups must be [].")
        sys_lines.append("- parallel_groups validity:")
        sys_lines.append("  - parallel_groups may contain only step names that are enabled=true in this plan.")
        sys_lines.append("")

        sys_lines.append("Retrieval policy (STRICT):")
        sys_lines.append("- Default: all retrieval steps disabled.")
        sys_lines.append("- Enable use_user_longterm_memory ONLY if the user query depends on user-specific context, prior decisions, preferences, or project-specific facts not present in the current message/history.")
        sys_lines.append("- For generic programming questions or general explanations, set use_user_longterm_memory.enabled=false.")
        sys_lines.append("- Enable use_rag/use_attachments ONLY if answering requires information from documents/attachments.")
        sys_lines.append("- Enable use_websearch ONLY if answering requires up-to-date or niche info not safely known.")
        sys_lines.append("- Retrieval steps must include conservative limits (top_k, max_chars) to avoid context overflow.")
        sys_lines.append("")

        sys_lines.append("Examples (MUST FOLLOW):")
        sys_lines.append("- Q: 'Explain async retry strategy in Python' -> use_user_longterm_memory.enabled=false")
        sys_lines.append("- Q: 'What is a mutex?' -> use_user_longterm_memory.enabled=false")
        sys_lines.append("- Q: 'Given my preferences for concise technical answers, how should you respond?' -> use_user_longterm_memory.enabled=true")
        sys_lines.append("- Q: 'Continue the Intergrax engine refactor we discussed earlier' -> use_user_longterm_memory.enabled=true")
        sys_lines.append("")

        sys_lines.append("Valid minimal example (JSON):")
        sys_lines.append(json.dumps(
            {
                "version": "1.0",
                "intent": "explain_async_retry_strategy",
                "reasoning_summary": "Answer is based on general Python async patterns; no external sources needed.",
                "ask_clarifying_question": False,
                "clarifying_question": None,
                "use_rag": {"enabled": False, "top_k": 0, "max_chars": 0},
                "use_user_longterm_memory": {"enabled": False, "top_k": 0, "max_chars": 0},
                "use_attachments": {"enabled": False, "top_k": 0, "max_chars": 0},
                "use_websearch": {"enabled": False, "top_k": 0, "max_chars": 0},
                "use_tools": {"enabled": False, "goal": "", "tool_choice_hint": None},
                "parallel_groups": [],
                "final_answer_style": "concise_technical",
            },
            ensure_ascii=False,
        ))
        sys_lines.append("")

        sys_lines.append("JSON schema (output must match keys and types):")
        sys_lines.append(json.dumps(schema, ensure_ascii=False))
        sys_lines.append("")

        # Request context
        attachments = req.attachments or []
        has_attachments = len(attachments) > 0

        user_lines: List[str] = []
        user_lines.append(f"session_id: {req.session_id}")
        user_lines.append(f"user_id: {req.user_id}")
        user_lines.append(f"has_attachments: {has_attachments}")
        user_lines.append(f"attachments_count: {len(attachments)}")
        user_lines.append("")

        # Instructions (runtime-like): request overrides + optional profile fragments.
        instr_parts: List[str] = []
        if req.instructions:
            instr_parts.append(req.instructions.strip())
        if state.profile_user_instructions:
            instr_parts.append(state.profile_user_instructions.strip())
        if state.profile_org_instructions:
            instr_parts.append(state.profile_org_instructions.strip())

        if instr_parts:
            user_lines.append("instructions (may affect style/policies):")
            user_lines.append("\n\n".join(instr_parts))
            user_lines.append("")

        # User query
        user_lines.append("user_query:")
        user_lines.append((req.message or "").strip())
        user_lines.append("")

        # Recent messages
        user_lines.append("recent_messages (most recent last):")
        recent = state.base_history or []
        for m in recent[-8:]:
            content = (m.content or "").strip()
            if len(content) > 400:
                content = content[:400]
            user_lines.append(f"- {m.role}: {content}")

        return [
            ChatMessage(role="system", content="\n".join(sys_lines)),
            ChatMessage(role="user", content="\n".join(user_lines)),
        ]


    # -----------------------------
    # Parsing & validation
    # -----------------------------

    def _extract_json_object(self, s: str) -> str:
        """
        Minimal safety: extract the first {...} block.
        This is not a heuristic planner; it's just robust parsing for LLM outputs.
        """
        start = s.find("{")
        end = s.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise ValueError("Planner did not return a JSON object.")
        return s[start : end + 1]

    def _req_str(self, d: Dict[str, Any], key: str) -> str:
        v = d.get(key)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Invalid or missing string field: {key}")
        return v.strip()

    def _opt_str(self, d: Dict[str, Any], key: str) -> Optional[str]:
        v = d.get(key)
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError(f"Invalid optional string field: {key}")
        vv = v.strip()
        return vv if vv else None

    def _req_bool(self, d: Dict[str, Any], key: str) -> bool:
        v = d.get(key)
        if not isinstance(v, bool):
            raise ValueError(f"Invalid or missing boolean field: {key}")
        return v

    def _req_int(self, d: Dict[str, Any], key: str) -> int:
        v = d.get(key)
        if not isinstance(v, int):
            raise ValueError(f"Invalid or missing int field: {key}")
        return int(v)

    def _parse_retrieval(self, d: Dict[str, Any], key: str) -> PlanRetrieval:
        obj = d.get(key)
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid or missing object field: {key}")
        enabled = obj.get("enabled")
        top_k = obj.get("top_k")
        max_chars = obj.get("max_chars")
        if not isinstance(enabled, bool) or not isinstance(top_k, int) or not isinstance(max_chars, int):
            raise ValueError(f"Invalid retrieval config for: {key}")
        # Ensure non-negative, but do not "auto-fix" anything beyond type safety.
        if top_k < 0 or max_chars < 0:
            raise ValueError(f"Invalid negative limits for: {key}")
        return PlanRetrieval(enabled=enabled, top_k=top_k, max_chars=max_chars)

    def _parse_tools(self, d: Dict[str, Any]) -> PlanTools:
        obj = d.get("use_tools")
        if not isinstance(obj, dict):
            raise ValueError("Invalid or missing object field: use_tools")
        enabled = obj.get("enabled")
        goal = obj.get("goal")
        tool_choice_hint = obj.get("tool_choice_hint")
        if not isinstance(enabled, bool) or not isinstance(goal, str):
            raise ValueError("Invalid tools config.")
        tch: Optional[str]
        if tool_choice_hint is None:
            tch = None
        elif isinstance(tool_choice_hint, str):
            tch = tool_choice_hint.strip() or None
        else:
            raise ValueError("Invalid tool_choice_hint type.")
        return PlanTools(enabled=enabled, goal=goal.strip(), tool_choice_hint=tch)

    def _parse_parallel_groups(self, d: Dict[str, Any]) -> List[List[str]]:
        pg = d.get("parallel_groups")
        if not isinstance(pg, list):
            raise ValueError("Invalid or missing parallel_groups.")
        out: List[List[str]] = []
        for group in pg:
            if not isinstance(group, list):
                raise ValueError("parallel_groups must be list[list[string]].")
            gg: List[str] = []
            for item in group:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("parallel_groups items must be non-empty strings.")
                gg.append(item.strip())
            out.append(gg)
        return out

    def _parse_plan(self, d: Dict[str, Any]) -> EnginePlan:
        version = self._req_str(d, "version")
        intent = self._req_str(d, "intent")
        reasoning_summary = self._req_str(d, "reasoning_summary")

        ask = self._req_bool(d, "ask_clarifying_question")
        cq = self._opt_str(d, "clarifying_question")
        if ask and not cq:
            raise ValueError("ask_clarifying_question=true requires clarifying_question.")

        use_rag = self._parse_retrieval(d, "use_rag")
        use_user_ltm = self._parse_retrieval(d, "use_user_longterm_memory")
        use_attachments = self._parse_retrieval(d, "use_attachments")
        use_websearch = self._parse_retrieval(d, "use_websearch")
        use_tools = self._parse_tools(d)

        parallel_groups = self._parse_parallel_groups(d)
        final_answer_style = self._req_str(d, "final_answer_style")

        return EnginePlan(
            version=version,
            intent=intent,
            reasoning_summary=reasoning_summary,
            ask_clarifying_question=ask,
            clarifying_question=cq,
            use_rag=use_rag,
            use_user_longterm_memory=use_user_ltm,
            use_attachments=use_attachments,
            use_websearch=use_websearch,
            use_tools=use_tools,
            parallel_groups=parallel_groups,
            final_answer_style=final_answer_style,
        )
