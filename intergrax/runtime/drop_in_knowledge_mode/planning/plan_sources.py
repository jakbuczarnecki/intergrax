# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.llm_adapter import LLMAdapter


@dataclass(frozen=True)
class PlanSourceMeta:
    source_kind: str
    source_detail: Optional[str] = None


class PlanSource(ABC):
    @abstractmethod
    async def generate_plan_raw(
        self,
        *,
        llm_adapter: LLMAdapter,
        messages: Sequence[ChatMessage],
        run_id: Optional[str] = None,
    ) -> tuple[str, PlanSourceMeta]:
        """
        Returns raw planner output (typically JSON text) and metadata.
        EnginePlanner is the single source of truth for parsing/validation.
        """
        raise NotImplementedError


class LLMPlanSource(PlanSource):
    async def generate_plan_raw(
        self,
        *,
        llm_adapter: LLMAdapter,
        messages: Sequence[ChatMessage],
        run_id: Optional[str] = None,
    ) -> tuple[str, PlanSourceMeta]:
        raw = llm_adapter.generate_messages(list(messages), run_id=run_id)

        # keep compatibility with adapters that may be async
        import inspect
        if inspect.iscoroutine(raw):
            raw = await raw

        if not isinstance(raw, str):
            raw = str(raw)

        return raw, PlanSourceMeta(source_kind="llm", source_detail=type(llm_adapter).__name__)
