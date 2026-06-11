# © Artur Czarnecki. All rights reserved.

"""Per-step LLM routing (architecture §33 · ACP-LLM-1)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_run_trace import GatewayCallStatus, LlmCallRecord


class LlmStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    model_id: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    call_record: LlmCallRecord


class LlmCompletePort(Protocol):
    async def complete(
        self,
        prompt: str,
        *,
        model_id: str,
        provider: str,
    ) -> tuple[str, int, int]: ...


@dataclass
class StepLLMRouter:
    """
    Resolve ``model_hint`` against host allowlist and record Plane B LLM calls.

    Tier-2 agents use this router — never vendor SDKs directly.
    """

    allowed_models: tuple[str, ...]
    default_model: str
    provider: str = "stub"
    llm_port: LlmCompletePort | None = None
    _pending_calls: list[LlmCallRecord] = field(default_factory=list, init=False, repr=False)
    _last_effective_model: str = field(default="", init=False, repr=False)

    def list_allowed_models(self) -> list[str]:
        return list(self.allowed_models)

    @property
    def effective_model(self) -> str:
        return self._last_effective_model or self.default_model

    def resolve_model(self, model_hint: str | None) -> str:
        if model_hint is None or model_hint == "":
            resolved = self.default_model
        elif model_hint in self.allowed_models:
            resolved = model_hint
        else:
            resolved = self.default_model
        self._last_effective_model = resolved
        return resolved

    def drain_pending_calls(self) -> list[LlmCallRecord]:
        drained = list(self._pending_calls)
        self._pending_calls.clear()
        return drained

    async def complete(self, prompt: str, *, model_hint: str | None = None) -> LlmStepResult:
        model_id = self.resolve_model(model_hint)
        started = time.perf_counter()
        if self.llm_port is not None:
            text, tokens_in, tokens_out = await self.llm_port.complete(
                prompt,
                model_id=model_id,
                provider=self.provider,
            )
        else:
            text = prompt
            tokens_in = len(prompt.split())
            tokens_out = len(text.split())
        latency_ms = int((time.perf_counter() - started) * 1000)
        call_record = LlmCallRecord(
            call_id=f"llm_{uuid.uuid4().hex[:12]}",
            model_id=model_id,
            provider=self.provider,
            status=GatewayCallStatus.SUCCEEDED,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            model_hint=model_hint,
        )
        self._pending_calls.append(call_record)
        return LlmStepResult(
            text=text,
            model_id=model_id,
            provider=self.provider,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            call_record=call_record,
        )
