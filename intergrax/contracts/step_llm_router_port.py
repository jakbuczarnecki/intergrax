# © Artur Czarnecki. All rights reserved.

"""Author-facing LLM router port (architecture §33 · ACP-LLM-1)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.agent_run_trace import LlmCallRecord


class LlmStepResultPort(Protocol):
    text: str
    model_id: str
    provider: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    call_record: LlmCallRecord


class StepLlmRouterPort(Protocol):
    def list_allowed_models(self) -> list[str]: ...

    @property
    def effective_model(self) -> str: ...

    def drain_pending_calls(self) -> list[LlmCallRecord]: ...

    async def complete(
        self,
        prompt: str,
        *,
        model_hint: str | None = None,
    ) -> LlmStepResultPort: ...
