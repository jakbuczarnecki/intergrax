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

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry


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


def BASE_PLANNER_SYSTEM_PROMPT()->str:
    registry = YamlPromptRegistry.create_default(load=True)
    return registry.resolve_localized(prompt_id="planner_base").system


def DEFAULT_PLANNER_SYSTEM_PROMPT()->str:
    registry = YamlPromptRegistry.create_default(load=True)
    return registry.resolve_localized(prompt_id="planner_default").system


def DEFAULT_PLANNER_REPLAN_SYSTEM_PROMPT()->str:
    registry = YamlPromptRegistry.create_default(load=True)
    localized = registry.resolve_localized(prompt_id="planner_replan_default")

    user_template = localized.user_template or ""
    system = localized.system or ""

    if user_template and system:
        return f"{user_template}\n\n{system}"
    if user_template:
        return user_template
    return system


def DEFAULT_PLANNER_NEXT_STEP_RULES_PROMPT()->str:
    registry = YamlPromptRegistry.create_default(load=True)
    return registry.resolve_localized(prompt_id="planner_next_step_rule").system


def DEFAULT_PLANNER_FALLBACK_CLARIFY_QUESTION()->str:
    registry = YamlPromptRegistry.create_default(load=True)
    return registry.resolve_localized(prompt_id="planner_fallback_clarify").system
