# © Artur Czarnecki. All rights reserved.

"""Per-step LLM routing (architecture §33 · ACP-LLM-1)."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.agent_run_trace import GatewayCallStatus, LlmCallRecord
from intergrax.llm.messages import (
    ChatMessage,
    copy_model_input_messages,
    replace_final_user_message,
    StructuredModelInputRequiredError,
)

if TYPE_CHECKING:
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
    from intergrax.runtime.nexus.config import RuntimeConfig


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


@runtime_checkable
class LlmMessagesCompletePort(Protocol):
    async def complete_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        model_id: str,
        provider: str,
    ) -> tuple[str, int, int]: ...


class LLMAdapterCompletePort:
    """Async bridge from Tier-0 ``LLMAdapter`` to ACP ``LlmCompletePort`` (M-LLM-X.5.4)."""

    def __init__(self, adapter: object) -> None:
        self._adapter = adapter

    async def complete(
        self,
        prompt: str,
        *,
        model_id: str,
        provider: str,
    ) -> tuple[str, int, int]:
        del model_id, provider

        def _call() -> tuple[str, int, int]:
            response = self._adapter.generate_messages(  # type: ignore[attr-defined]
                [ChatMessage(role="user", content=prompt)],
            )
            usage = response.usage
            tokens_in = int(usage.input_tokens) if usage is not None else 0
            tokens_out = int(usage.output_tokens) if usage is not None else 0
            return response.content, tokens_in, tokens_out

        return await asyncio.to_thread(_call)

    async def complete_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        model_id: str,
        provider: str,
    ) -> tuple[str, int, int]:
        del model_id, provider
        send_messages = copy_model_input_messages(messages)

        def _call() -> tuple[str, int, int]:
            response = self._adapter.generate_messages(  # type: ignore[attr-defined]
                list(send_messages),
            )
            usage = response.usage
            tokens_in = int(usage.input_tokens) if usage is not None else 0
            tokens_out = int(usage.output_tokens) if usage is not None else 0
            return response.content, tokens_in, tokens_out

        return await asyncio.to_thread(_call)


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
    llm_adapter: LLMAdapter | None = None
    runtime_config: RuntimeConfig | None = None
    require_real_llm: bool = False
    model_input_messages: tuple[ChatMessage, ...] = ()
    _pending_calls: list[LlmCallRecord] = field(default_factory=list, init=False, repr=False)
    _last_effective_model: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if self.model_input_messages:
            self.model_input_messages = copy_model_input_messages(
                self.model_input_messages
            )

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

    def _resolve_completion_port(self) -> LlmCompletePort | None:
        if self.llm_port is not None:
            return self.llm_port
        adapter = self.llm_adapter
        if adapter is None and self.runtime_config is not None:
            adapter = self.runtime_config.llm_adapter
        if adapter is not None:
            return LLMAdapterCompletePort(adapter)
        return None

    async def complete(self, prompt: str, *, model_hint: str | None = None) -> LlmStepResult:
        model_id = self.resolve_model(model_hint)
        started = time.perf_counter()
        provider = self.provider
        if self.runtime_config is not None and self.runtime_config.llm_adapter is not None:
            raw_provider = self.runtime_config.llm_adapter.provider
            provider = raw_provider.value if hasattr(raw_provider, "value") else str(raw_provider)

        if self.model_input_messages:
            prepared = replace_final_user_message(self.model_input_messages, prompt)
            send_messages = copy_model_input_messages(prepared)
            completion_port = self._resolve_completion_port()
            complete_messages = (
                getattr(completion_port, "complete_messages", None)
                if completion_port is not None
                else None
            )
            if callable(complete_messages):
                text, tokens_in, tokens_out = await complete_messages(
                    list(send_messages),
                    model_id=model_id,
                    provider=provider,
                )
            elif completion_port is not None:
                raise StructuredModelInputRequiredError()
            elif self.require_real_llm:
                raise RuntimeError("StepLLMRouter requires llm_port or llm_adapter in production mode")
            else:
                text = prompt
                tokens_in = sum(len((message.content or "").split()) for message in send_messages)
                tokens_out = len(text.split())
        else:
            if self.runtime_config is not None:
                from intergrax.runtime.nexus.context.compile_service import compile_prompt_text

                prompt = compile_prompt_text(prompt, self.runtime_config)  # type: ignore[arg-type]
            completion_port = self._resolve_completion_port()
            if completion_port is not None:
                text, tokens_in, tokens_out = await completion_port.complete(
                    prompt,
                    model_id=model_id,
                    provider=provider,
                )
            elif self.require_real_llm:
                raise RuntimeError("StepLLMRouter requires llm_port or llm_adapter in production mode")
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
