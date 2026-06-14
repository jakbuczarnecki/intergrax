# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Profile-chain failover wrapper for LLM adapters (M-LLM-X.4.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, TypeVar

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.retry import is_retriable_provider_error
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LLMRoutingAttemptRecord:
    """In-process record of a failover attempt (trace DTO deferred to M-LLM-X.4.4)."""

    profile_index: int
    provider: str
    model: str
    error: str


class FailoverLLMAdapter(LLMAdapter):
    """
    Try adapters in order on retriable provider errors (429, 5xx, timeout).

    Uses the primary adapter for context window and token estimation.
    """

    def __init__(self, adapters: Sequence[LLMAdapter]) -> None:
        super().__init__()
        if not adapters:
            raise ValueError("FailoverLLMAdapter requires at least one adapter")
        self._adapters = tuple(adapters)
        primary = adapters[0]
        self.provider = primary.provider
        self.model = primary.model
        self.model_name_for_token_estimation = primary.model_name_for_token_estimation
        self.call_config = primary.call_config
        self.routing_attempts: list[LLMRoutingAttemptRecord] = []

    @property
    def context_window_tokens(self) -> int:
        return self._adapters[0].context_window_tokens

    def _provider_model(self, adapter: LLMAdapter) -> tuple[str, str]:
        provider = adapter.provider
        slug = provider.value if hasattr(provider, "value") else str(provider)
        return slug, str(adapter.model or "")

    def _execute_with_failover(self, operation: Callable[[LLMAdapter], T]) -> T:
        last_exc: BaseException | None = None
        for index, adapter in enumerate(self._adapters):
            try:
                return operation(adapter)
            except BaseException as exc:
                last_exc = exc
                provider, model = self._provider_model(adapter)
                self.routing_attempts.append(
                    LLMRoutingAttemptRecord(
                        profile_index=index,
                        provider=provider,
                        model=model,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                is_last = index >= len(self._adapters) - 1
                if is_last or not is_retriable_provider_error(exc, adapter.call_config):
                    raise
        assert last_exc is not None
        raise last_exc

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        return self._execute_with_failover(
            lambda adapter: adapter.generate_messages(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                run_id=run_id,
            )
        )

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        return self._execute_with_failover(
            lambda adapter: adapter.generate_with_tools(
                messages,
                tools,
                temperature=temperature,
                max_tokens=max_tokens,
                run_id=run_id,
            )
        )

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> Iterable[LLMStreamEvent]:
        adapter = self._select_streaming_adapter()
        return adapter.stream_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )

    def stream_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> Iterable[LLMStreamEvent]:
        adapter = self._select_streaming_adapter()
        return adapter.stream_with_tools(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[T]:
        return self._execute_with_failover(
            lambda adapter: adapter.generate_structured(
                messages,
                output_model,
                temperature=temperature,
                max_tokens=max_tokens,
                run_id=run_id,
            )
        )

    def _select_streaming_adapter(self) -> LLMAdapter:
        for adapter in self._adapters:
            if adapter.supports_streaming():
                return adapter
        return self._adapters[0]

    def supports_streaming(self) -> bool:
        return any(adapter.supports_streaming() for adapter in self._adapters)

    def supports_structured_output(self) -> bool:
        return any(adapter.supports_structured_output() for adapter in self._adapters)

    def supports_vision(self) -> bool:
        return any(adapter.supports_vision() for adapter in self._adapters)

    def supports_audio_input(self) -> bool:
        return any(adapter.supports_audio_input() for adapter in self._adapters)

    def supports_audio_output(self) -> bool:
        return any(adapter.supports_audio_output() for adapter in self._adapters)
