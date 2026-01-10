# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
import hashlib
import json
import pprint
from typing import Any, Dict, Optional


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

    def fingerprint(self) -> str:
      """
      Stable fingerprint of the *decision* part of EnginePlan.
      Excludes trace/debug and free-form reasoning to avoid false differences.
      """
      def _enum(v: Any) -> Any:
          return v.value if hasattr(v, "value") else v

      # Normalize clarify_question: strip and collapse whitespace
      cq: Optional[str] = self.clarifying_question
      if cq is not None:
          cq = " ".join(cq.strip().split()) or None

      payload: Dict[str, Any] = {
          "version": self.version,
          "intent": _enum(self.intent),
          "ask_clarifying_question": bool(self.ask_clarifying_question),
          "clarifying_question": cq,
          "next_step": _enum(self.next_step) if self.next_step is not None else None,
          "use_websearch": bool(self.use_websearch),
          "use_user_longterm_memory": bool(self.use_user_longterm_memory),
          "use_rag": bool(self.use_rag),
          "use_tools": bool(self.use_tools),
      }

      raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
      return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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

    def to_planner_dict(self) -> Dict[str, Any]:
      """
      Convert EnginePlan into the strict dict shape expected by EnginePlanner._parse_plan().
      This makes EnginePlan usable as a deterministic 'forced_plan' override.
      """
      def _enum(v: Any) -> Any:
          return v.value if hasattr(v, "value") else v

      return {
          "version": self.version,
          "intent": _enum(self.intent),
          "next_step": _enum(self.next_step) if self.next_step is not None else None,
          "reasoning_summary": self.reasoning_summary or "",
          "ask_clarifying_question": bool(self.ask_clarifying_question),
          "clarifying_question": self.clarifying_question,
          "use_websearch": bool(self.use_websearch),
          "use_user_longterm_memory": bool(self.use_user_longterm_memory),
          "use_rag": bool(self.use_rag),
          "use_tools": bool(self.use_tools),
      }
    
    @classmethod
    def clarify(
        cls,
        *,
        question: str,
        version: str = "1",
        reasoning_summary: str = "Deterministic clarify plan.",
    ) -> "EnginePlan":
        """
        Deterministic plan that forces a clarifying question.
        """
        return cls(
            version=version,
            intent=PlanIntent.CLARIFY,
            next_step=EngineNextStep.CLARIFY,
            reasoning_summary=reasoning_summary,
            ask_clarifying_question=True,
            clarifying_question=question,
            use_websearch=False,
            use_user_longterm_memory=False,
            use_rag=False,
            use_tools=False,
        )

    @classmethod
    def generic_finalize(
        cls,
        *,
        version: str = "1",
        reasoning_summary: str = "Deterministic generic finalize plan.",
        use_user_longterm_memory: bool = False,
    ) -> "EnginePlan":
        """
        Deterministic plan that goes straight to FINALIZE (no external retrieval).
        """
        return cls(
            version=version,
            intent=PlanIntent.GENERIC,
            next_step=EngineNextStep.FINALIZE,
            reasoning_summary=reasoning_summary,
            ask_clarifying_question=False,
            clarifying_question=None,
            use_websearch=False,
            use_user_longterm_memory=use_user_longterm_memory,
            use_rag=False,
            use_tools=False,
        )

    @classmethod
    def with_rag(
        cls,
        *,
        version: str = "1",
        reasoning_summary: str = "Deterministic RAG plan.",
        intent: PlanIntent = PlanIntent.GENERIC,
        use_user_longterm_memory: bool = False,
    ) -> "EnginePlan":
        """
        Deterministic plan that forces RAG retrieval.
        """
        return cls(
            version=version,
            intent=intent,
            next_step=EngineNextStep.RAG,
            reasoning_summary=reasoning_summary,
            ask_clarifying_question=False,
            clarifying_question=None,
            use_websearch=False,
            use_user_longterm_memory=use_user_longterm_memory,
            use_rag=True,
            use_tools=False,
        )

    @classmethod
    def with_websearch(
        cls,
        *,
        version: str = "1",
        reasoning_summary: str = "Deterministic websearch plan.",
    ) -> "EnginePlan":
        """
        Deterministic plan that forces websearch.
        """
        return cls(
            version=version,
            intent=PlanIntent.FRESHNESS,
            next_step=EngineNextStep.WEBSEARCH,
            reasoning_summary=reasoning_summary,
            ask_clarifying_question=False,
            clarifying_question=None,
            use_websearch=True,
            use_user_longterm_memory=False,
            use_rag=False,
            use_tools=False,
        )

    @classmethod
    def with_tools(
        cls,
        *,
        version: str = "1",
        reasoning_summary: str = "Deterministic tools plan.",
        intent: PlanIntent = PlanIntent.GENERIC,
    ) -> "EnginePlan":
        """
        Deterministic plan that forces tools execution.
        """
        return cls(
            version=version,
            intent=intent,
            next_step=EngineNextStep.TOOLS,
            reasoning_summary=reasoning_summary,
            ask_clarifying_question=False,
            clarifying_question=None,
            use_websearch=False,
            use_user_longterm_memory=False,
            use_rag=False,
            use_tools=True,
        )
    
    def with_flags(
      self,
      *,
      use_user_longterm_memory: Optional[bool] = None,
      use_rag: Optional[bool] = None,
      use_websearch: Optional[bool] = None,
      use_tools: Optional[bool] = None,
    ) -> "EnginePlan":
      return replace(
          self,
          use_user_longterm_memory=self.use_user_longterm_memory
          if use_user_longterm_memory is None
          else use_user_longterm_memory,
          use_rag=self.use_rag if use_rag is None else use_rag,
          use_websearch=self.use_websearch if use_websearch is None else use_websearch,
          use_tools=self.use_tools if use_tools is None else use_tools,
      )
    
    @classmethod
    def generic_synthesize(
        cls,
        *,
        version: str = "forced-v1",
        reasoning_summary: str = "forced plan for deterministic same-plan-repeat test",
        use_user_longterm_memory: bool = False,
    ) -> "EnginePlan":
        """
        Deterministic plan that forces SYNTHESIZE step (no retrieval).
        Useful for tests that validate repeated same-plan behavior.
        """
        return cls(
            version=version,
            intent=PlanIntent.GENERIC,
            next_step=EngineNextStep.SYNTHESIZE,
            reasoning_summary=reasoning_summary,
            ask_clarifying_question=False,
            clarifying_question=None,
            use_websearch=False,
            use_user_longterm_memory=use_user_longterm_memory,
            use_rag=False,
            use_tools=False,
        )

    

@dataclass(frozen=True)
class PlannerPromptConfig:
    version: str = "default"
    system_prompt: Optional[str] = None
    replan_system_prompt: Optional[str] = None
    next_step_rules_prompt: Optional[str] = None
    fallback_clarify_question: Optional[str] = None

    # Optional deterministic override:
    # - allows replaying a captured plan
    # - allows running the planner without LLM (e.g. offline / incident mode)
    forced_plan: Optional[EnginePlan] = None

    def __post_init__(self) -> None:
        if self.forced_plan is not None and not isinstance(self.forced_plan, EnginePlan):
          raise TypeError("forced_plan must be an EnginePlan")


BASE_PLANNER_SYSTEM_PROMPT = """You are EnginePlanner for Intergrax nexus Runtime.
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


# DEFAULT_PLANNER_SYSTEM_PROMPT = """
#         You are EnginePlanner for Intergrax nexus Runtime.
#         Return a SINGLE JSON object only. No prose. No markdown. No comments.
#         The JSON MUST match the provided JSON Schema EXACTLY (no extra keys).
#         Do NOT include chain-of-thought. Put a short high-level note in reasoning_summary.

#         Hard constraints:
#         - If a capability is unavailable, its corresponding use_* flag MUST be false.
#         - If intent is 'clarify', ask_clarifying_question MUST be true and clarifying_question MUST be a single question.
#         - If intent is not 'clarify', ask_clarifying_question MUST be false and clarifying_question MUST be null.

#         Intent definitions:
#         - generic: general answer using internal knowledge; no external retrieval needed.
#         - freshness: requires up-to-date info; prefer websearch if available.
#         - project_architecture: depends on user's project history/preferences; prefer user long-term memory if available.
#         - clarify: question is ambiguous/missing info; ask exactly one clarifying question.

#         Tools policy (STRICT):
#         - Default: use_tools=false.
#         - Set use_tools=true in TWO cases only:
#         (A) External action/data source is required AND is handled by the tools pipeline (NOT websearch).
#         (B) Deterministic transformation/extraction is required with a strict output contract, e.g.:
#             - "only output JSON", "exact list", "sorted", "unique", "no prose", "return only ..."
#             - parsing/transforming provided JSON/CSV/XML/text into a precise structured output
#         - If use_tools=true: next_step MUST be "tools".

#         Examples:
#         - Q: 'Explain async retry in Python' -> use_tools=false
#         - Q: 'What are the most recent changes to the OpenAI Responses API? Provide dates.' -> use_websearch=true (if available), use_tools=false
#         - Q: 'Search the web and summarize the latest changes to the OpenAI Responses API' -> use_websearch=true (if available), use_tools=true ONLY if your tools system is the websearch tool.

#         Intent field constraints (STRICT):
#         - intent MUST be EXACTLY one of: "generic", "freshness", "project_architecture", "clarify".
#         - Do NOT output any other intent value (e.g., "compare", "choose", "optimize", "decision").

#         User long-term memory policy (STRICT, HARD RULE):
#         - use_user_longterm_memory MUST be true ONLY when intent is exactly "project_architecture" AND the capability is available.
#         - For intents "generic", "freshness", and "clarify": use_user_longterm_memory MUST be false.

#         Websearch vs Tools policy (STRICT, HARD RULE):
#         - In this runtime, websearch is NOT part of tools. Websearch is a separate pipeline.
#         - Therefore, if use_websearch=true, then use_tools MUST be false.
#         - Set use_tools=true ONLY for non-websearch external actions handled by the tools pipeline.

#         Clarify policy (STRICT, HARD RULE):
#         - Use intent="clarify" when the user's request is ambiguous OR missing required details to answer safely/correctly.
#         - Missing-required-info triggers (NON-NEGOTIABLE). If ANY trigger matches => intent MUST be "clarify":
#         - user mentions an exception/error/bug but provides no traceback/stack trace
#         - no error message, no reproduction steps, no environment/version, no minimal example when needed
#         - "it doesn't work" / "I got an error" without the actual error text
#         - For missing traceback specifically: clarifying_question MUST ask for the full traceback and minimal repro in one sentence.
#         - If intent="clarify": next_step MUST be "clarify", ask_clarifying_question MUST be true, and clarifying_question MUST be exactly one question.
#         - If intent!="clarify": ask_clarifying_question MUST be false and clarifying_question MUST be null.

#         RAG policy (STRICT, HARD RULE):
#         - If intent is "generic" or "freshness" or "clarify": use_rag MUST be false.
#         - use_rag MAY be true only when intent is exactly "project_architecture" AND capability is available.
#         """


# DEFAULT_PLANNER_SYSTEM_PROMPT = """
# You are EnginePlanner for Intergrax nexus Runtime.

# Return a SINGLE JSON object only. No prose. No markdown. No comments.
# The JSON MUST match the provided JSON Schema EXACTLY (no extra keys).
# Do NOT include chain-of-thought. Put a short high-level note in reasoning_summary.

# Hard constraints:
# - If a capability is unavailable, its corresponding use_* flag MUST be false.
# - If intent is 'clarify': ask_clarifying_question MUST be true and clarifying_question MUST be a single question.
# - If intent is not 'clarify': ask_clarifying_question MUST be false and clarifying_question MUST be null.
# - If use_tools=true: next_step MUST be "tools".
# - If use_websearch=true: next_step MUST be "websearch".

# Intent definitions:
# - generic: general answer using internal knowledge; no external retrieval needed; no strict output contract.
# - freshness: requires up-to-date info; prefer websearch if available.
# - project_architecture: depends on user's project and preferences; can use user long-term memory and project RAG if available.
# - clarify: question is ambiguous/missing required info; ask exactly one clarifying question.

# Decision procedure (follow in order):
# 1) CLARIFY gate:
#    If the request is ambiguous OR missing required details to answer safely/correctly => intent="clarify".
# 2) FRESHNESS gate:
#    If the request requires recent/dated information not safely answerable from general knowledge => intent="freshness".
# 3) PROJECT_ARCHITECTURE gate:
#    If the request depends on the user's project/codebase/history/preferences => intent="project_architecture".
# 4) Otherwise => intent="generic".

# Tools policy (STRICT):
# - Default: use_tools=false.
# - Set use_tools=true ONLY in TWO cases:
#   (A) External action/data source is required AND is handled by the tools pipeline (NOT websearch).
#   (B) The user requires a deterministic transformation/extraction with a strict output contract (machine-readable output),
#       where format correctness matters more than prose quality.

# Strict output contract signals (non-exhaustive; treat as strong indicators for (B)):
# - The user explicitly demands format constraints such as:
#   "only output ...", "no prose", "no explanation", "return only", "exact", "verbatim", "must be valid JSON",
#   "JSON array only", "schema", "sorted", "unique", "deduplicate", "extract fields", "parse", "transform".
# - The input includes structured data (JSON/CSV/XML/logs) and the task is to compute/transform it precisely.
# If these signals are present, prefer use_tools=true even if the transformation seems simple.

# Websearch vs Tools policy (STRICT, HARD RULE):
# - In this runtime, websearch is NOT part of tools. Websearch is a separate pipeline.
# - Therefore, if use_websearch=true, then use_tools MUST be false.
# - Set use_tools=true ONLY for non-websearch external actions handled by the tools pipeline,
#   or for deterministic transformations under the strict output contract rule (B).

# User long-term memory policy (STRICT, HARD RULE):
# - use_user_longterm_memory MUST be true ONLY when intent is exactly "project_architecture" AND the capability is available.
# - For intents "generic", "freshness", and "clarify": use_user_longterm_memory MUST be false.

# RAG policy (STRICT, HARD RULE):
# - If intent is "generic" or "freshness" or "clarify": use_rag MUST be false.
# - use_rag MAY be true only when intent is exactly "project_architecture" AND capability is available.

# Clarify policy (STRICT, HARD RULE):
# - Missing-required-info triggers (NON-NEGOTIABLE). If ANY trigger matches => intent MUST be "clarify":
#   - user mentions an exception/error/bug but provides no traceback/stack trace
#   - no error message, no reproduction steps, no environment/version, no minimal example when needed
#   - "it doesn't work" / "I got an error" without the actual error text
# - For missing traceback specifically:
#   clarifying_question MUST ask for the full traceback and a minimal repro in one sentence.
# - If intent="clarify": next_step MUST be "clarify", ask_clarifying_question MUST be true,
#   and clarifying_question MUST be exactly one question.
# - If intent!="clarify": ask_clarifying_question MUST be false and clarifying_question MUST be null.

# Intent field constraints (STRICT):
# - intent MUST be EXACTLY one of: "generic", "freshness", "project_architecture", "clarify".
# - Do NOT output any other intent value.

# Examples (illustrative, not exhaustive):
# - Explanation request without strict output contract -> intent="generic", use_tools=false.
# - Up-to-date changes with dates -> intent="freshness", use_websearch=true (if available), use_tools=false.
# - Question about user's runtime/codebase -> intent="project_architecture", use_rag/use_user_longterm_memory as available.
# - "I have an error but no traceback" -> intent="clarify", next_step="clarify".
# - "Return only valid JSON / exact list / sorted unique values from provided data" -> use_tools=true, next_step="tools".
# """


DEFAULT_PLANNER_SYSTEM_PROMPT = """
You are EnginePlanner for Intergrax nexus Runtime.

Return a SINGLE JSON object only. No prose. No markdown. No comments.
The JSON MUST match the provided JSON Schema EXACTLY (no extra keys).
Do NOT include chain-of-thought. Put a short high-level note in reasoning_summary.

Core idea:
- intent describes the user's request type.
- use_* flags describe which capabilities are likely needed to answer well.
- next_step selects the FIRST execution step for StepPlanner (subsequent steps may be planned iteratively later).

Hard constraints:
- If a capability is unavailable, its corresponding use_* flag MUST be false.
- If intent is 'clarify': ask_clarifying_question MUST be true and clarifying_question MUST be a single question.
- If intent is not 'clarify': ask_clarifying_question MUST be false and clarifying_question MUST be null.
- If next_step is "tools": use_tools MUST be true.
- If next_step is "websearch": use_websearch MUST be true.
- If next_step is "clarify": intent MUST be "clarify".

Intent definitions (choose ONE):
- generic: can be answered from general knowledge or provided context; no requirement for up-to-date facts.
- freshness: the user explicitly asks for latest/recent/current info OR correctness depends on post-cutoff changes.
- project_architecture: the answer depends on the user's specific project/runtime/codebase/conventions/preferences
  (including Intergrax/Mooff details), not just general knowledge.
- clarify: essential information is missing to answer safely/correctly (for the user's stated goal).

Planning procedure (do NOT use rigid gates; evaluate needs):
1) Determine whether the user wants (a) diagnosis of a concrete situation OR (b) general guidance.
   - If diagnosis requires missing essentials => intent="clarify".
2) Determine whether the question requires up-to-date facts (post-training) => set intent="freshness".
3) Determine whether the question depends on the user's project specifics => set intent="project_architecture".
4) Otherwise => intent="generic".
Note: If both (2) and (3) apply, prefer intent="project_architecture" (because the answer must fit the user's system),
but still set use_websearch/use_rag/use_user_longterm_memory as needed.

Capability selection (truthful, flexible):
- use_user_longterm_memory:
  Set true when user preferences/history/project conventions are helpful to answer correctly or in the expected style,
  AND capability is available. This can apply to any intent except "clarify" (where you must first ask).
- use_rag:
  Set true when project/docs knowledgebase is likely needed to answer accurately (APIs, code, internal docs, specs),
  AND capability is available. This can apply to generic or project_architecture intents.
- use_websearch:
  Set true when the answer depends on current/up-to-date external information, AND capability is available.
- use_tools:
  Set true when the answer requires:
  (A) non-websearch external actions/data sources handled by tools pipeline, OR
  (B) high-reliability deterministic transformation/validation where format correctness is critical.

Deterministic transformation rule (STRONG):
- If the user provides structured input (JSON/CSV/XML/logs) AND asks to compute/transform it with exactness
  (e.g., "exact", "only output", "sorted", "unique", "deduplicate", "extract fields", "parse", "transform"),
  then set use_tools=true and next_step="tools" (unless tools capability is unavailable).
- Treat "Only output the JSON array ..." + "sorted/unique/exact" as a decisive signal for tools.
- Do NOT set use_tools=true for "please answer in JSON" if no transformation/validation is required.

Websearch and Tools relationship:
- Websearch is a separate pipeline from tools.
- They are NOT logically mutually exclusive in the overall solution.
- However, next_step MUST choose only one FIRST step. If both may be needed, pick the one that should happen first:
  - Prefer websearch first when external freshness is needed.
  - Prefer tools first when deterministic transformation/validation is needed on provided data.

Clarify policy:
- Use intent="clarify" ONLY when missing information blocks the user's stated goal.
- If the user reports an error and asks for diagnosis but provides no traceback/logs, ask for:
  full traceback + minimal repro + environment/version in ONE question.
- If the user asks for general debugging guidance (not diagnosis), do NOT force clarify.

Choosing next_step:
- If intent="clarify": next_step="clarify".
- Else if use_websearch=true and freshness is a key requirement: next_step="websearch".
- Else if use_tools=true and deterministic transformation/validation is the primary requirement: next_step="tools".
- Else: next_step="synthesize" (or whatever your schema uses for normal answering).

In reasoning_summary (short):
- State why the intent was chosen and which capability is needed first.
- If both websearch and tools are likely needed, mention the expected sequence in one short phrase.
"""


DEFAULT_PLANNER_REPLAN_SYSTEM_PROMPT: str = """
REPLAN FEEDBACK (structured JSON):
{replan_json}

REPLAN RULES:
- Do NOT repeat the same invalid plan.
- If a dependency was missing, add the producer step before the consumer.
- If a step output was invalid/missing, adjust the plan to produce it.
- If the only safe action is to ask the user, set intent='clarify' and provide clarifying_question.
- Keep the output JSON strictly within the schema (no extra keys).
""".strip()


DEFAULT_PLANNER_NEXT_STEP_RULES_PROMPT: str = """
RULES FOR next_step:
- If intent == "clarify": next_step MUST be "clarify".
- If intent != "clarify": next_step MUST NOT be "clarify".
Clarify intent should be used ONLY when the user request is ambiguous or missing critical information.
Do NOT choose clarify if you can answer with a reasonable technical/general response without asking follow-ups.
Do NOT choose clarify for broad/open questions; answer them as GENERIC and use next_step="synthesize".

- Choose exactly one next_step for THIS iteration.
- Use "websearch" for freshness/external information.
- Use "rag" for internal documents or user long-term memory.
- Use "tools" only when tool execution is required.
- Use "synthesize" when you have enough information to draft an answer.
- Use "finalize" only when you can return the final answer now.
""".strip()


DEFAULT_PLANNER_FALLBACK_CLARIFY_QUESTION: str = (
    "Could you clarify what you mean and what outcome you want?"
)
