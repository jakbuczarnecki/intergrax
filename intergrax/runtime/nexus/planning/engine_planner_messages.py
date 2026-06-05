# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LLM prompt assembly for :class:`~intergrax.runtime.nexus.planning.engine_planner.EnginePlanner` (Q+-P.1)."""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.engine_plan_models import (
    DEFAULT_PLANNER_NEXT_STEP_RULES_PROMPT,
    DEFAULT_PLANNER_REPLAN_SYSTEM_PROMPT,
    DEFAULT_PLANNER_SYSTEM_PROMPT,
    PlannerPromptConfig,
)
from intergrax.runtime.nexus.planning.step_executor_models import ReplanContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tracing.plan.replan_context_injected import (
    PlannerReplanContextInjectedDiagV1,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class EnginePlannerMessageBuilder:
    """Build low-variance planner chat messages (strict JSON schema)."""

    @staticmethod
    def serialize_replan_ctx(replan_ctx: Optional[ReplanContext]) -> Optional[str]:
        if replan_ctx is None:
            return None
        return json.dumps(
            replan_ctx.to_prompt_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def build_messages(
        cls,
        *,
        req: RuntimeRequest,
        state: RuntimeState,
        prompt_config: Optional[PlannerPromptConfig] = None,
        replan_ctx: Optional[ReplanContext] = None,
    ) -> list[ChatMessage]:
        caps = {
            "websearch_available": state.cap_websearch_available,
            "user_ltm_available": state.cap_user_ltm_available,
            "rag_available": state.cap_rag_available,
            "tools_available": state.cap_tools_available,
            "attachments_present": bool(req.attachments and len(req.attachments or []) > 0),
        }

        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "version",
                "intent",
                "next_step",
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
                "next_step": {
                    "type": "string",
                    "enum": ["clarify", "websearch", "tools", "rag", "synthesize", "finalize"],
                },
                "reasoning_summary": {"type": "string"},
                "ask_clarifying_question": {"type": "boolean"},
                "clarifying_question": {"type": ["string", "null"]},
                "use_websearch": {"type": "boolean"},
                "use_user_longterm_memory": {"type": "boolean"},
                "use_rag": {"type": "boolean"},
                "use_tools": {"type": "boolean"},
                "tool_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Canonical catalog tool ids (e.g. rag.retrieve, websearch.query).",
                },
            },
        }

        prompt_registry = state.context.prompt_registry
        catalog_path = state.context.config.prompt_catalog_path
        prompt_kwargs = {
            "registry": prompt_registry,
            "catalog_path": catalog_path,
        }

        system_prompt = DEFAULT_PLANNER_SYSTEM_PROMPT(**prompt_kwargs)
        if prompt_config is not None and prompt_config.system_prompt:
            system_prompt = prompt_config.system_prompt.strip()

        replan_system_msg: Optional[ChatMessage] = None
        if replan_ctx is not None:
            replan_template = DEFAULT_PLANNER_REPLAN_SYSTEM_PROMPT(**prompt_kwargs)
            if prompt_config is not None and prompt_config.replan_system_prompt:
                replan_template = prompt_config.replan_system_prompt.strip()

            replan_json = cls.serialize_replan_ctx(replan_ctx)
            assert replan_json is not None
            replan_hash = hashlib.sha256(replan_json.encode("utf-8")).hexdigest()[:16]

            state.trace_event(
                component=TraceComponent.PLANNER,
                step="engine_planner",
                message="Replan context injected into planner prompt.",
                level=TraceLevel.INFO,
                payload=PlannerReplanContextInjectedDiagV1(
                    has_replan_ctx=True,
                    replan_reason=(replan_ctx.replan_reason or "").strip() or None,
                    replan_hash=replan_hash,
                    replan_json_len=len(replan_json),
                ),
            )

            replan_text = replan_template.format(replan_json=replan_json)
            replan_system_msg = ChatMessage(role="system", content=replan_text)

        next_step_rules_prompt = DEFAULT_PLANNER_NEXT_STEP_RULES_PROMPT(**prompt_kwargs)
        if prompt_config is not None and prompt_config.next_step_rules_prompt:
            next_step_rules_prompt = prompt_config.next_step_rules_prompt.strip()

        user_lines: list[str] = [
            "CAPABILITIES (hard constraints):",
            json.dumps(caps, ensure_ascii=False, sort_keys=True),
            "",
            "USER QUERY:",
            (req.message or "").strip(),
            "",
            next_step_rules_prompt,
            "",
            "JSON SCHEMA:",
            json.dumps(schema, ensure_ascii=False),
            "",
            "OUTPUT JSON:",
        ]

        messages: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]
        if replan_system_msg is not None:
            messages.append(replan_system_msg)
        messages.append(ChatMessage(role="user", content="\n".join(user_lines)))
        return messages
