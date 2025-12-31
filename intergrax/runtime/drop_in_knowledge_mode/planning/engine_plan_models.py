# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import pprint
from typing import Any, Dict, Literal, Optional


# -----------------------------
# Typed plan schema
# -----------------------------

class EngineNextStep(str, Enum):
    CLARIFY = "clarify"
    WEBSEARCH = "websearch"
    TOOLS = "tools"
    RAG = "rag"
    SYNTHESIZE = "synthesize"
    FINALIZE = "finalize"
    

class PlanIntent(str, Enum):
    GENERIC = "generic"
    FRESHNESS = "freshness"
    PROJECT_ARCHITECTURE = "project_architecture"
    CLARIFY = "clarify"
    

@dataclass(frozen=False)
class EnginePlan:
    version: str
    intent: PlanIntent

    # Debug / trace only
    reasoning_summary: str = ""

    # Clarify only
    ask_clarifying_question: bool = False
    clarifying_question: Optional[str] = None

    # Next action (policy routing for this iteration)
    next_step: Optional[EngineNextStep] = None

    # Soft preferences for this iteration (NOT hard constraints)
    use_websearch: bool = False
    use_user_longterm_memory: bool = False
    use_rag: bool = False
    use_tools: bool = False

    debug: Dict[str, Any] = field(default_factory=dict)


    def print_pretty(self) -> None:
        pprint.pprint({
            "version": self.version,
            "intent": self.intent.value if isinstance(self.intent, Enum) else str(self.intent),
            "reasoning_summary": self.reasoning_summary,
            "ask_clarifying_question": self.ask_clarifying_question,
            "clarifying_question": self.clarifying_question,
            "use_websearch": self.use_websearch,
            "use_user_longterm_memory": self.use_user_longterm_memory,
            "use_rag": self.use_rag,
            "use_tools": self.use_tools,
            "debug": self.debug,
        })
    

@dataclass(frozen=True)
class PlannerPromptConfig:
    version: str = "default"
    system_prompt: Optional[str] = None


BASE_PLANNER_SYSTEM_PROMPT = """You are EnginePlanner for Intergrax Drop-In Knowledge Runtime.
Return a SINGLE JSON object only. No prose. No markdown. No comments.
The JSON MUST match the provided JSON Schema EXACTLY (no extra keys).
Do NOT include chain-of-thought. Put a short high-level note in reasoning_summary.

Hard constraints:
- If a capability is unavailable, its corresponding use_* flag MUST be false.
- If intent is 'clarify', ask_clarifying_question MUST be true and clarifying_question MUST be a single question.
- If intent is not 'clarify', ask_clarifying_question MUST be false and clarifying_question MUST be null.

Intent definitions:
- generic: general answer using internal knowledge; no external retrieval needed.
- freshness: requires up-to-date info; prefer websearch if available.
- project_architecture: depends on user's project history/preferences; prefer user long-term memory if available.
- clarify: question is ambiguous/missing info; ask exactly one clarifying question.

Tools policy (STRICT):
- Default: use_tools=false.
- Set use_tools=true ONLY if the user explicitly requests tool usage or the task requires an external action/data source (e.g., web lookup, calling tools, operating on user resources).

Examples:
- Q: 'Explain async retry in Python' -> use_tools=false
- Q: 'What are the most recent changes to the OpenAI Responses API? Provide dates.' -> use_websearch=true (if available), use_tools=false
- Q: 'Search the web and summarize the latest changes to the OpenAI Responses API' -> use_websearch=true (if available), use_tools=true ONLY if your tools system is the websearch tool.
"""


DEFAULT_PLANNER_SYSTEM_PROMPT = BASE_PLANNER_SYSTEM_PROMPT + """
        Intent field constraints (STRICT):
        - intent MUST be EXACTLY one of: "generic", "freshness", "project_architecture", "clarify".
        - Do NOT output any other intent value (e.g., "compare", "choose", "optimize", "decision").

        User long-term memory policy (STRICT, HARD RULE):
        - use_user_longterm_memory MUST be true ONLY when intent is exactly "project_architecture" AND the capability is available.
        - For intents "generic", "freshness", and "clarify": use_user_longterm_memory MUST be false.

        Websearch vs Tools policy (STRICT, HARD RULE):
        - In this runtime, websearch is NOT part of tools. Websearch is a separate pipeline.
        - Therefore, if use_websearch=true, then use_tools MUST be false.
        - Set use_tools=true ONLY for non-websearch external actions handled by the tools pipeline.

        Clarify policy (STRICT, HARD RULE):
        - Use intent="clarify" when the user's request is ambiguous or missing required details to answer.
        - Triggers: the question asks to choose/compare/optimize ("which one", "compare", "what should I choose", "what should I change"),
        but the options and/or decision criteria are not provided.
        - If intent="clarify": ask_clarifying_question MUST be true and clarifying_question MUST be exactly one question.
        - If intent!="clarify": ask_clarifying_question MUST be false and clarifying_question MUST be null.
        """
