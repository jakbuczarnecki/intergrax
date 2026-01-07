# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Optional, Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.llm_adapter import LLMAdapter
from intergrax.runtime.drop_in_knowledge_mode.planning.step_executor_models import ReplanContext


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
    A serializable plan representation for scripted/replay sources.
    Keeps plan_sources.py decoupled from EnginePlan and avoids unions in public APIs.
    """
    json_obj: Optional[Mapping[str, Any]] = None
    raw_json: Optional[str] = None


    def __post_init__(self) -> None:
        if (self.json_obj is None) == (self.raw_json is None):
            # both None or both set -> invalid
            raise ValueError("PlanSpec requires exactly one of: json_obj or raw_json.")


    def to_raw_json(self) -> str:
        if self.raw_json is not None:
            raw = self.raw_json.strip()

            try:
                obj = json.loads(raw)
            except Exception as e:
                raise ValueError(f"PlanSpec.raw_json must be valid JSON: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError("PlanSpec.raw_json must be a JSON object (dict).")

            # Optionally re-canonicalize to ensure stable hashing/traces:
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

        if self.json_obj is None:
            raise ValueError("PlanSpec must have either raw_json or json_obj.")

        # Canonical JSON for stable hashing and consistent traces
        return json.dumps(
            self.json_obj,
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

