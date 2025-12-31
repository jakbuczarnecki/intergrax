# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import inspect
import json
from typing import List, Optional
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.llm_adapter import LLMAdapter
from intergrax.runtime.drop_in_knowledge_mode.config import RuntimeConfig
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.planning.engine_plan_models import DEFAULT_PLANNER_SYSTEM_PROMPT, EnginePlan, PlanIntent, PlannerPromptConfig
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeRequest



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
        prompt_config: Optional[PlannerPromptConfig] = None,
        run_id: Optional[str] = None,
    ) -> EnginePlan:
        messages = self._build_planner_messages(
            req=req, 
            state=state, 
            config=config,
            prompt_config=prompt_config,
        )

        raw = self._llm.generate_messages(messages, run_id=run_id)
        if inspect.iscoroutine(raw):
            raw = await raw
        if not isinstance(raw, str):
            raw = str(raw)

        # Single source of truth: parsing extracts JSON and loads it
        plan = self._parse_plan(raw)

        # Capability clamp
        plan = self._validate_against_capabilities(plan=plan, state=state)

        # Merge debug (do not overwrite parser debug)
        if plan.debug is None:
            plan.debug = {}

        plan.debug.update(
            {
                "planner_raw_len": len(raw),
                "planner_raw_preview": raw[:400],
                "planner_raw_tail_preview": raw[-400:],
            }
        )

        return plan
    

    def _validate_against_capabilities(self, *, plan: EnginePlan, state: RuntimeState) -> EnginePlan:
        """
        Hard capability clamp. No heuristics, no intent changes.
        Only disables flags that are not available in the current runtime.
        """

        if plan.use_websearch and not state.cap_websearch_available:
            plan.use_websearch = False

        if plan.use_user_longterm_memory and not state.cap_user_ltm_available:
            plan.use_user_longterm_memory = False

        if plan.use_rag and not state.cap_rag_available:
            plan.use_rag = False

        if plan.use_tools and not state.cap_tools_available:
            plan.use_tools = False

        # Optional: record clamp info for debugging
        if plan.debug is None:
            plan.debug = {}

        plan.debug["capability_clamp"] = {
            "websearch_available": state.cap_websearch_available,
            "user_ltm_available": state.cap_user_ltm_available,
            "rag_available": state.cap_rag_available,
            "tools_available": state.cap_tools_available,
            "use_websearch": plan.use_websearch,
            "use_user_longterm_memory": plan.use_user_longterm_memory,
            "use_rag": plan.use_rag,
            "use_tools": plan.use_tools,
        }

        return plan

    # -----------------------------
    # Prompting
    # -----------------------------

    def _build_planner_messages(
        self,
        *,
        req: RuntimeRequest,
        state: RuntimeState,
        config: RuntimeConfig,
        prompt_config: Optional[PlannerPromptConfig] = None,
    ) -> List[ChatMessage]:
        """
        Build minimal, low-variance planner messages.

        The model must output a SINGLE JSON object that matches the schema exactly.
        The output is parsed into EnginePlan (simplified).
        """

        # Capabilities are hard constraints (runtime will clamp again as a safety net).
        caps = {
            "websearch_available": state.cap_websearch_available,
            "user_ltm_available": state.cap_user_ltm_available,
            "rag_available": state.cap_rag_available,
            "tools_available": state.cap_tools_available,
            "attachments_present": req.attachments and len(req.attachments or []) > 0,
        }

        # Strict JSON schema (minimal surface area, no extra keys).
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "version",
                "intent",
                "reasoning_summary",
                "ask_clarifying_question",
                "clarifying_question",
                "use_websearch",
                "use_user_longterm_memory",
                "use_rag",
                "use_tools",
            ],
            "properties": {
                "version": {"type": "string"},
                "intent": {
                    "type": "string",
                    "enum": ["generic", "freshness", "project_architecture", "clarify"],
                },
                "reasoning_summary": {"type": "string"},
                "ask_clarifying_question": {"type": "boolean"},
                "clarifying_question": {"type": ["string", "null"]},
                "use_websearch": {"type": "boolean"},
                "use_user_longterm_memory": {"type": "boolean"},
                "use_rag": {"type": "boolean"},
                "use_tools": {"type": "boolean"},
            },
        }

        system_prompt = DEFAULT_PLANNER_SYSTEM_PROMPT

        if prompt_config is not None and prompt_config.system_prompt:
            system_prompt = prompt_config.system_prompt.strip()

        # User message: provide only needed context + schema.
        user_lines: List[str] = []
        user_lines.append("CAPABILITIES (hard constraints):")
        user_lines.append(json.dumps(caps, ensure_ascii=False))
        user_lines.append("")
        user_lines.append("USER QUERY:")
        user_lines.append((req.message or "").strip())
        user_lines.append("")
        user_lines.append("JSON SCHEMA:")
        user_lines.append(json.dumps(schema, ensure_ascii=False))
        user_lines.append("")
        user_lines.append("OUTPUT JSON:")

        return [
            ChatMessage(role="system", content=system_prompt),
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

    def _parse_plan(self, raw: str) -> EnginePlan:
        """
        Parse strict JSON (as specified by _build_planner_messages) into EnginePlan.

        Expected keys:
        version, intent, reasoning_summary, ask_clarifying_question, clarifying_question,
        use_websearch, use_user_longterm_memory, use_rag, use_tools
        """
        js = self._extract_json_object(raw)

        try:
            data = json.loads(js)
        except Exception as e:
            raise ValueError(f"Invalid JSON from LLM: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Planner output must be a JSON object.")

        def req_bool(k: str) -> bool:
            if k not in data:
                raise ValueError(f"Missing required key: {k}")
            v = data[k]
            if isinstance(v, bool):
                return v
            raise ValueError(f"Key '{k}' must be boolean.")

        def req_str(k: str) -> str:
            if k not in data:
                raise ValueError(f"Missing required key: {k}")
            v = data[k]
            if isinstance(v, str):
                return v
            raise ValueError(f"Key '{k}' must be string.")

        def req_str_or_null(k: str) -> Optional[str]:
            if k not in data:
                raise ValueError(f"Missing required key: {k}")
            v = data[k]
            if v is None:
                return None
            if isinstance(v, str):
                vv = v.strip()
                return vv
            raise ValueError(f"Key '{k}' must be string or null.")

        def opt_version_str(k: str, default: str = "1.0") -> str:
            """
            Version is metadata. Be tolerant: accept string/number/null/missing.
            Always return a non-empty string.
            """
            if k not in data:
                return default

            v = data[k]
            if v is None:
                return default

            if isinstance(v, str):
                s = v.strip()
                return s or default

            if isinstance(v, (int, float)):
                return str(v)

            raise ValueError(f"Key '{k}' must be string, number, or null.")

        version = opt_version_str("version", default="1.0")

        intent_raw = req_str("intent").strip()
        try:
            intent = PlanIntent(intent_raw)
        except Exception:
            raise ValueError(
                f"Invalid intent '{intent_raw}'. Allowed: generic|freshness|project_architecture|clarify."
            )

        reasoning_summary = req_str("reasoning_summary").strip()

        ask_clarify = req_bool("ask_clarifying_question")
        clar_q = req_str_or_null("clarifying_question")

        use_web = req_bool("use_websearch")
        use_ltm = req_bool("use_user_longterm_memory")
        use_rag = req_bool("use_rag")
        use_tools = req_bool("use_tools")

        # Deterministic consistency rules (independent from model compliance)
        if intent == PlanIntent.CLARIFY:
            ask_clarify = True
            if not clar_q:
                # If missing, force a safe generic clarifier.
                clar_q = "Could you clarify what you mean and what outcome you want?"
            # In clarify mode, retrieval is always off
            use_web = use_ltm = use_rag = use_tools = False
        else:
            # Non-clarify must not ask clarifying question
            if ask_clarify:
                # If model set it true incorrectly, force clarify intent
                intent = PlanIntent.CLARIFY
                if not clar_q:
                    clar_q = "Could you clarify what you mean and what outcome you want?"
                use_web = use_ltm = use_rag = use_tools = False
            else:
                clar_q = None

        return EnginePlan(
            version=version,
            intent=intent,
            reasoning_summary=reasoning_summary,
            ask_clarifying_question=ask_clarify,
            clarifying_question=clar_q,
            use_websearch=use_web,
            use_user_longterm_memory=use_ltm,
            use_rag=use_rag,
            use_tools=use_tools,
            debug={
                "raw_json": data,
                "planner_json_len": len(js),
            },
        )

