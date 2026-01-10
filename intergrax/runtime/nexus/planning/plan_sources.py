# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.planning.engine_plan_models import EngineNextStep, PlanIntent
from intergrax.runtime.nexus.planning.step_executor_models import ReplanContext


@dataclass(frozen=True)
class PlanSourceMeta:
    source_kind: str
    source_detail: Optional[str] = None


@dataclass(frozen=True)
class PlanRequest:
    llm_adapter: LLMAdapter
    messages: Sequence[ChatMessage]
    run_id: Optional[str]
    replan_ctx: Optional[ReplanContext]


@dataclass(frozen=True)
class PlanSourceResult:
    raw: str
    meta: PlanSourceMeta
    raw_hash16: Optional[str] = None


@dataclass(frozen=True)
class PlanSpec:
    """
    Strongly-typed plan spec for scripted/replay sources.

    Production goals:
    - No loose dict/JSON blobs in tests.
    - Contract is aligned with engine_plan_models enums (single source of truth).
    - Canonical JSON serialization for stable hashing/traces.
    """

    version: str
    intent: PlanIntent
    next_step: EngineNextStep

    # Debug / trace only
    reasoning_summary: str = ""

    # Clarify only
    ask_clarifying_question: bool = False
    clarifying_question: Optional[str] = None

    # Controls for pipeline decisions
    use_websearch: bool = False
    use_user_longterm_memory: bool = False
    use_rag: bool = False
    use_tools: bool = False

    # Optional forward-compatible debug payload (must be JSON-serializable)
    debug: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # Clarify contract: keep it strict to prevent invalid scripted plans.
        if self.next_step == EngineNextStep.CLARIFY:
            if not self.ask_clarifying_question:
                raise ValueError(
                    "PlanSpec invalid: next_step=CLARIFY requires ask_clarifying_question=True."
                )
            cq = (self.clarifying_question or "").strip()
            if not cq:
                raise ValueError(
                    "PlanSpec invalid: ask_clarifying_question=True but clarifying_question is empty."
                )
        else:
            if self.ask_clarifying_question:
                raise ValueError(
                    "PlanSpec invalid: ask_clarifying_question=True requires next_step=CLARIFY."
                )
            # Enforce cleanliness: do not allow stray text question.
            if self.clarifying_question is not None and self.clarifying_question.strip():
                raise ValueError(
                    "PlanSpec invalid: clarifying_question provided but ask_clarifying_question=False."
                )

    def to_json_obj(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "version": self.version,
            "intent": self.intent.value,
            "next_step": self.next_step.value,
            "reasoning_summary": self.reasoning_summary,
            "ask_clarifying_question": bool(self.ask_clarifying_question),
            "clarifying_question": self.clarifying_question,
            "use_websearch": bool(self.use_websearch),
            "use_user_longterm_memory": bool(self.use_user_longterm_memory),
            "use_rag": bool(self.use_rag),
            "use_tools": bool(self.use_tools),
        }
        if self.debug is not None:
            obj["debug"] = self.debug
        return obj

    def to_raw_json(self) -> str:
        # Canonical JSON for stable hashing and consistent traces
        return json.dumps(
            self.to_json_obj(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )




class PlanSource(ABC):
    @abstractmethod
    async def generate_plan_raw(
        self,
        *,
        req: PlanRequest
    ) -> PlanSourceResult:
        """
        Returns raw planner output (typically JSON text) and metadata.
        EnginePlanner is the single source of truth for parsing/validation.
        """
        raise NotImplementedError


class LLMPlanSource(PlanSource):
    async def generate_plan_raw(
        self,
        *,
        req: PlanRequest,
    ) -> PlanSourceResult:
        raw = req.llm_adapter.generate_messages(list(req.messages), run_id=req.run_id)

        import inspect
        if inspect.iscoroutine(raw):
            raw = await raw

        if not isinstance(raw, str):
            raw = str(raw)

        return PlanSourceResult(
            raw=raw, 
            meta=PlanSourceMeta(source_kind="llm", source_detail=type(req.llm_adapter).__name__)
        )



class ScriptedPlanSource(PlanSource):
    """
    Deterministic plan source for tests and incident replay.

    Selection rule:
      - index = req.replan_ctx.attempt if present else 0
    """

    def __init__(self, plans: Sequence[PlanSpec]) -> None:
        self._plans = list(plans)
        if not self._plans:
            raise ValueError("ScriptedPlanSource requires a non-empty plans sequence.")

    async def generate_plan_raw(
        self,
        *,
        req: PlanRequest,
    ) -> PlanSourceResult:
        attempt = 0
        if req.replan_ctx is not None:
            attempt = req.replan_ctx.attempt

        if attempt < 0 or attempt >= len(self._plans):
            raise IndexError(
                "ScriptedPlanSource attempt out of range "
                f"(attempt={attempt}, plans_count={len(self._plans)})."
            )

        spec = self._plans[attempt]
        raw = spec.to_raw_json()

        raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        meta = PlanSourceMeta(
            source_kind="scripted",
            source_detail=f"attempt={attempt};plans_count={len(self._plans)};hash={raw_hash}",
        )

        return PlanSourceResult(
            raw=raw,
            meta=meta,
            raw_hash16=raw_hash,
        )

