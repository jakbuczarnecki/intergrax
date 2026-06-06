# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage

if TYPE_CHECKING:
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter, LLMCallStats


@dataclass(slots=True)
class LLMCallLifecycle:
    """Typed per-call lifecycle helper shared by provider adapters (M-LLM-R.2.6)."""

    adapter: LLMAdapter
    call: LLMCallStats
    success: bool = False
    error_type: str | None = None

    @classmethod
    def begin(
        cls,
        adapter: LLMAdapter,
        *,
        run_id: str | None,
    ) -> LLMCallLifecycle:
        return cls(
            adapter=adapter,
            call=adapter.usage.begin_call(run_id=run_id, adapter=adapter),
        )

    def mark_success(self) -> None:
        self.success = True
        self.error_type = None

    def mark_failure(self, exc: BaseException) -> None:
        self.success = False
        self.error_type = type(exc).__name__

    def end(self, usage: LLMTokenUsage | None) -> None:
        token_usage = usage or LLMTokenUsage()
        self.adapter.usage.end_call(
            self.call,
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
            success=self.success,
            error_type=self.error_type,
        )
