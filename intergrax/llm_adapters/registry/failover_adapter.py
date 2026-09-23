# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Profile-chain failover wrapper for LLM adapters (M-LLM-X.4.2)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters.contracts.llm_provider import llm_provider_slug
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.strict_tool_arguments import (
    CanonicalFunctionToolDefinition,
    StrictToolArgumentConformanceError,
    coerce_canonical_tool_definitions,
    tool_definitions_require_strict_argument_conformance,
)
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.native_tool_choice import NativeToolChoice
from intergrax.llm_adapters.contracts.failover_policy import (
    FailoverDecision,
    FailoverPolicy,
    FailoverProgressionContext,
)
from intergrax.llm_adapters.registry.failover_policy import default_failover_policy

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


@dataclass(frozen=True, slots=True)
class _FailoverChainEntry:
    """Atomic adapter + profile identity + failover eligibility policy."""

    adapter: LLMAdapter
    profile_id: str
    failover_retry_config: LLMCallConfig


class FailoverLLMAdapter(BaseLLMAdapter):
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
        failover_retry_config: LLMCallConfig | None = None,
        adapter_failover_retry_configs: Sequence[LLMCallConfig] | None = None,
        failover_policy: FailoverPolicy | None = None,
    ) -> None:
        super().__init__()
        self._failover_policy = (
            failover_policy if failover_policy is not None else default_failover_policy()
        )
        if failover_retry_config is not None:
            self.call_config = failover_retry_config
        if not adapters:
            raise ValueError("FailoverLLMAdapter requires at least one adapter")
        resolved_profile_ids = (
            tuple(profile_ids)
            if profile_ids is not None
            else tuple(str(index) for index in range(len(adapters)))
        )
        if len(resolved_profile_ids) != len(adapters):
            raise ValueError("profile_ids length must match adapters length")
        if adapter_failover_retry_configs is not None:
            if len(adapter_failover_retry_configs) != len(adapters):
                raise ValueError(
                    "adapter_failover_retry_configs length must match adapters length"
                )
            resolved_retry_configs = tuple(adapter_failover_retry_configs)
        else:
            resolved_retry_configs = tuple(self.call_config for _ in adapters)
        self._chain = tuple(
            _FailoverChainEntry(
                adapter=adapter,
                profile_id=profile_id,
                failover_retry_config=retry_config,
            )
            for adapter, profile_id, retry_config in zip(
                adapters, resolved_profile_ids, resolved_retry_configs
            )
        )
        primary = adapters[0]
        self.provider = primary.provider
        self.model = primary.model
        self.model_name_for_token_estimation = primary.model or None
        self.routing_attempts: list[LLMRoutingAttemptRecord] = []
        self.routing_attempt_observer = routing_attempt_observer

    @property
    def context_window_tokens(self) -> int:
        return self._chain[0].adapter.context_window_tokens

    def _provider_model(self, adapter: LLMAdapter) -> tuple[str, str]:
        return llm_provider_slug(adapter.provider), str(adapter.model or "")

    def _eligible_adapter_chain(
        self,
        tools: Sequence[CanonicalFunctionToolDefinition] | None = None,
    ) -> tuple[_FailoverChainEntry, ...]:
        """Return chain entries eligible for dispatch; filter strict-ineligible children."""
        if tools is None:
            return self._chain
        definitions = coerce_canonical_tool_definitions(tools)
        if not tool_definitions_require_strict_argument_conformance(definitions):
            return self._chain
        primary = self._chain[0].adapter
        if not primary.supports_strict_tool_argument_conformance():
            raise StrictToolArgumentConformanceError(
                "tools schema requires provider-enforced strict argument conformance but "
                f"{primary.__class__.__name__} does not support it"
            )
        return tuple(
            entry
            for entry in self._chain
            if entry.adapter.supports_strict_tool_argument_conformance()
        )

    def _execute_with_failover(
        self,
        operation: Callable[[LLMAdapter], T],
        *,
        chain: Sequence[_FailoverChainEntry] | None = None,
    ) -> T:
        active_chain = tuple(chain) if chain is not None else self._chain
        if not active_chain:
            raise ValueError("failover chain must contain at least one entry")
        self.routing_attempts.clear()
        last_exc: BaseException | None = None
        for index, entry in enumerate(active_chain):
            adapter = entry.adapter
            try:
                return operation(adapter)
            except BaseException as exc:
                last_exc = exc
                provider, model = self._provider_model(adapter)
                record = LLMRoutingAttemptRecord(
                    profile_index=index,
                    profile_id=entry.profile_id,
                    provider=provider,
                    model=model,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.routing_attempts.append(record)
                if self.routing_attempt_observer is not None:
                    self.routing_attempt_observer(record)
                is_last = index >= len(active_chain) - 1
                decision = self._failover_policy.decide_after_failure(
                    FailoverProgressionContext(
                        attempt_index=index,
                        is_last_candidate=is_last,
                        error=exc,
                        failover_retry_config=entry.failover_retry_config,
                    )
                )
                if decision is FailoverDecision.STOP:
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
        tools: Sequence[CanonicalFunctionToolDefinition],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: NativeToolChoice | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        chain = self._eligible_adapter_chain(tools)
        return self._execute_with_failover(
            lambda adapter: adapter.generate_with_tools(
                messages,
                tools,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                run_id=run_id,
            ),
            chain=chain,
        )

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> Iterable[LLMStreamEvent]:
        adapters = tuple(entry.adapter for entry in self._chain)
        adapter = self._select_streaming_adapter_from(adapters)
        return adapter.stream_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )

    def stream_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[CanonicalFunctionToolDefinition],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: NativeToolChoice | None = None,
        run_id: str | None = None,
    ) -> Iterable[LLMStreamEvent]:
        chain = self._eligible_adapter_chain(tools)
        adapters = tuple(entry.adapter for entry in chain)
        adapter = self._select_streaming_adapter_from(adapters)
        return adapter.stream_with_tools(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
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
        return any(entry.adapter.supports_streaming() for entry in self._chain)

    def supports_structured_output(self) -> bool:
        return any(entry.adapter.supports_structured_output() for entry in self._chain)

    def supports_strict_tool_argument_conformance(self) -> bool:
        """Primary must support strict; failover excludes strict-ineligible children."""
        return self._chain[0].adapter.supports_strict_tool_argument_conformance()

    def supports_vision(self) -> bool:
        return any(entry.adapter.supports_vision() for entry in self._chain)

    def supports_audio_input(self) -> bool:
        return any(entry.adapter.supports_audio_input() for entry in self._chain)

    def supports_audio_output(self) -> bool:
        return any(entry.adapter.supports_audio_output() for entry in self._chain)
