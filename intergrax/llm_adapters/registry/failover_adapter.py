# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Profile-chain failover wrapper for LLM adapters (M-LLM-X.4.2)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable, Sequence, TypeVar

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.retry import is_retriable_provider_error
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.strict_tool_arguments import (
    StrictToolArgumentConformanceError,
    tools_schema_requires_strict_argument_conformance,
)
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent

T = TypeVar("T")


RoutingAttemptObserver = Callable[["LLMRoutingAttemptRecord"], None]


@dataclass(frozen=True, slots=True)
class LLMRoutingAttemptRecord:
    """In-process record of a failover attempt (M-LLM-X.4.4 trace bridge)."""

    profile_index: int
    profile_id: str
    provider: str
    model: str
    error: str


class FailoverLLMAdapter(LLMAdapter):
    """
    Try adapters in order on retriable provider errors (429, 5xx, timeout).

    Uses the primary adapter for context window and token estimation.
    """

    def __init__(
        self,
        adapters: Sequence[LLMAdapter],
        *,
        profile_ids: Sequence[str] | None = None,
        routing_attempt_observer: RoutingAttemptObserver | None = None,
    ) -> None:
        super().__init__()
        if not adapters:
            raise ValueError("FailoverLLMAdapter requires at least one adapter")
        self._adapters = tuple(adapters)
        if profile_ids is not None and len(profile_ids) != len(adapters):
            raise ValueError("profile_ids length must match adapters length")
        self._profile_ids = tuple(
            profile_ids
            if profile_ids is not None
            else tuple(str(index) for index in range(len(adapters)))
        )
        primary = adapters[0]
        self.provider = primary.provider
        self.model = primary.model
        self.model_name_for_token_estimation = primary.model_name_for_token_estimation
        self.call_config = primary.call_config
        self.routing_attempts: list[LLMRoutingAttemptRecord] = []
        self.routing_attempt_observer = routing_attempt_observer

    @property
    def context_window_tokens(self) -> int:
        return self._adapters[0].context_window_tokens

    def _provider_model(self, adapter: LLMAdapter) -> tuple[str, str]:
        provider = adapter.provider
        slug = provider.value if hasattr(provider, "value") else str(provider)
        return slug, str(adapter.model or "")

    def _eligible_adapter_chain(
        self,
        tools_schema: Sequence[dict] | None = None,
    ) -> tuple[tuple[LLMAdapter, ...], tuple[str, ...]]:
        """Return adapters eligible for dispatch; filter strict-ineligible children."""
        if tools_schema is None or not tools_schema_requires_strict_argument_conformance(
            tools_schema
        ):
            return self._adapters, self._profile_ids
        if not self._adapters[0].supports_strict_tool_argument_conformance():
            raise StrictToolArgumentConformanceError(
                "tools schema requires provider-enforced strict argument conformance but "
                f"{self._adapters[0].__class__.__name__} does not support it"
            )
        pairs = [
            (adapter, profile_id)
            for adapter, profile_id in zip(self._adapters, self._profile_ids)
            if adapter.supports_strict_tool_argument_conformance()
        ]
        adapters, profile_ids = zip(*pairs)
        return tuple(adapters), tuple(profile_ids)

    def _execute_with_failover(
        self,
        operation: Callable[[LLMAdapter], T],
        *,
        adapters: Sequence[LLMAdapter] | None = None,
        profile_ids: Sequence[str] | None = None,
    ) -> T:
        active_adapters = tuple(adapters) if adapters is not None else self._adapters
        active_profile_ids = (
            tuple(profile_ids) if profile_ids is not None else self._profile_ids
        )
        if len(active_adapters) != len(active_profile_ids):
            raise ValueError("adapters and profile_ids length must match")
        self.routing_attempts.clear()
        last_exc: BaseException | None = None
        for index, adapter in enumerate(active_adapters):
            try:
                return operation(adapter)
            except BaseException as exc:
                last_exc = exc
                provider, model = self._provider_model(adapter)
                record = LLMRoutingAttemptRecord(
                    profile_index=index,
                    profile_id=active_profile_ids[index],
                    provider=provider,
                    model=model,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.routing_attempts.append(record)
                if self.routing_attempt_observer is not None:
                    self.routing_attempt_observer(record)
                is_last = index >= len(active_adapters) - 1
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
        adapters, profile_ids = self._eligible_adapter_chain(tools)
        return self._execute_with_failover(
            lambda adapter: adapter.generate_with_tools(
                messages,
                tools,
                temperature=temperature,
                max_tokens=max_tokens,
                run_id=run_id,
            ),
            adapters=adapters,
            profile_ids=profile_ids,
        )

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> Iterable[LLMStreamEvent]:
        adapter = self._select_streaming_adapter_from(self._adapters)
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
        adapters, _profile_ids = self._eligible_adapter_chain(tools)
        adapter = self._select_streaming_adapter_from(adapters)
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

    def _select_streaming_adapter_from(self, adapters: Sequence[LLMAdapter]) -> LLMAdapter:
        for adapter in adapters:
            if adapter.supports_streaming():
                return adapter
        return adapters[0]

    def supports_streaming(self) -> bool:
        return any(adapter.supports_streaming() for adapter in self._adapters)

    def supports_structured_output(self) -> bool:
        return any(adapter.supports_structured_output() for adapter in self._adapters)

    def supports_strict_tool_argument_conformance(self) -> bool:
        """Primary must support strict; failover excludes strict-ineligible children."""
        return self._adapters[0].supports_strict_tool_argument_conformance()

    def supports_vision(self) -> bool:
        return any(adapter.supports_vision() for adapter in self._adapters)

    def supports_audio_input(self) -> bool:
        return any(adapter.supports_audio_input() for adapter in self._adapters)

    def supports_audio_output(self) -> bool:
        return any(adapter.supports_audio_output() for adapter in self._adapters)
