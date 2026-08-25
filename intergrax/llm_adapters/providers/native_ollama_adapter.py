# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, List, Optional, Protocol, Union, cast

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
    final_stream_event,
    partial_stream_event,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.provider_extensions import LLMProviderExtensions
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.contracts.tool_call import (
    LLMToolCall,
    finalize_accepted_tool_call_identities,
)
from intergrax.llm_adapters.providers._ollama_schema import (
    prepare_ollama_generation_schema,
)
from intergrax.utils import attribute_access
from intergrax.llm_adapters.providers.ollama_capabilities import (
    OllamaModelCapabilities,
    OllamaModelCapabilityResolver,
)
from intergrax.llm_adapters.registry.context_window import (
    init_adapter_context_window_tokens,
)


class _NativeOllamaClient(Protocol):
    """Minimal native client surface used by the adapter."""

    def chat(
        self,
        model: str = "",
        messages: Sequence[Mapping[str, object]] | None = None,
        *,
        tools: Sequence[Mapping[str, object]] | None = None,
        stream: bool = False,
        format: str | Mapping[str, object] | None = None,
        options: Mapping[str, object] | None = None,
    ) -> object:
        ...


_MISSING = object()


class NativeOllamaAdapter(LLMAdapter):
    """Ollama adapter using the official native Python client."""

    DEFAULT_MODEL = "llama3.1:latest"

    def __init__(
        self,
        client: _NativeOllamaClient | None = None,
        model: Optional[str] = None,
        context_window_tokens: Optional[int] = None,
        *,
        capability_resolver: OllamaModelCapabilityResolver | None = None,
        base_url: Optional[str] = None,
        **defaults: Any,
    ) -> None:
        super().__init__()
        self._apply_defaults_call_config(defaults)

        resolved_model = model or self.DEFAULT_MODEL
        if client is None:
            try:
                from ollama import Client
            except ModuleNotFoundError as exc:
                if exc.name != "ollama":
                    raise
                from intergrax.llm_adapters.llm_provider_registry import (
                    LLMAdapterDependencyError,
                )

                raise LLMAdapterDependencyError(
                    "LLM provider 'ollama' requires dependency 'ollama'. "
                    "Install it with 'Intergrax-ai[llm-ollama]' before selecting "
                    "this provider."
                ) from exc

            client = cast(
                _NativeOllamaClient,
                Client(host=base_url) if base_url else Client(),
            )

        self._client = client
        self.defaults = dict(defaults)
        if context_window_tokens is not None and int(context_window_tokens) > 0:
            self.defaults["context_window_tokens"] = int(context_window_tokens)
        self._base_url = base_url
        self._capability_resolver = capability_resolver or OllamaModelCapabilityResolver(
            base_url=base_url,
        )
        self._model_capabilities: OllamaModelCapabilities | None = None
        self._context_window_tokens = int(
            init_adapter_context_window_tokens(
                provider=LLMProvider.OLLAMA,
                model=resolved_model,
                constructor_kwargs=self.defaults,
            )
        )

        self.provider = LLMProvider.OLLAMA
        self.model = resolved_model

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens

    @staticmethod
    def _field(value: object, name: str, default: object = None) -> object:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return attribute_access.optional(value, name, default)

    @classmethod
    def _response_content(cls, response: object) -> str:
        message = cls._field(response, "message", _MISSING)
        source = message if message is not _MISSING and message is not None else response
        content = cls._field(source, "content", "")
        return cls._coerce_content(content)

    @staticmethod
    def _coerce_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if not content:
            return ""
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, Mapping) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    @staticmethod
    def _arguments_object(raw_arguments: object) -> dict[str, object]:
        if isinstance(raw_arguments, Mapping):
            return dict(raw_arguments)
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                raise ValueError("tool call arguments must be valid JSON") from None
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("tool call arguments must decode to an object")
        raise ValueError("tool call arguments must be a dictionary or JSON string")

    @classmethod
    def _normalize_input_tool_call(cls, tool_call: object) -> dict[str, object]:
        if not isinstance(tool_call, Mapping):
            raise ValueError("tool call must be a dictionary")

        if "name" in tool_call and "args" in tool_call:
            name = tool_call.get("name")
            raw_arguments = tool_call.get("args")
        else:
            function = tool_call.get("function")
            if not isinstance(function, Mapping):
                raise ValueError("tool call requires name")
            name = function.get("name")
            raw_arguments = function.get("arguments")

        if not name or not str(name).strip():
            raise ValueError("tool call requires name")
        arguments = cls._arguments_object(raw_arguments)
        return {
            "id": str(tool_call.get("id") or ""),
            "function": {
                "name": str(name),
                "arguments": arguments,
            },
        }

    def _map_messages(
        self,
        messages: Sequence[ChatMessage],
    ) -> list[dict[str, object]]:
        mapped: list[dict[str, object]] = []
        for message in messages:
            if message.role == "system":
                mapped.append({"role": "system", "content": message.content})
            elif message.role == "user":
                mapped.append({"role": "user", "content": message.content})
            elif message.role == "assistant":
                item: dict[str, object] = {
                    "role": "assistant",
                    "content": message.content,
                }
                if message.tool_calls:
                    item["tool_calls"] = [
                        self._normalize_input_tool_call(tool_call)
                        for tool_call in message.tool_calls
                    ]
                mapped.append(item)
            elif message.role == "tool":
                if not message.tool_call_id or not str(message.tool_call_id).strip():
                    raise ValueError("tool message requires tool_call_id")
                mapped.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": str(message.tool_call_id),
                    }
                )
            else:
                mapped.append(
                    {
                        "role": "system",
                        "content": f"[{message.role.upper()}]\n{message.content}",
                    }
                )
        return mapped

    @staticmethod
    def _validate_ollama_tool_choice(
        tool_choice: Optional[Union[str, Dict[str, Any]]],
    ) -> None:
        if tool_choice is None:
            return
        if isinstance(tool_choice, str) and tool_choice in {"auto", "required"}:
            return
        raise ValueError(
            "Ollama native tool calling supports only tool_choice=None, 'auto', or 'required'"
        )

    def _generation_options(
        self,
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> dict[str, object]:
        raw_options = self.defaults.get("options") or {}
        if not isinstance(raw_options, Mapping):
            raise ValueError("Ollama options must be a mapping")
        options = dict(raw_options)
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        return options

    def _chat(
        self,
        messages: list[dict[str, object]],
        *,
        stream: bool,
        options: Mapping[str, object],
        tools: Sequence[Mapping[str, object]] | None = None,
        format: Mapping[str, object] | None = None,
    ) -> object:
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if format is not None:
            kwargs["format"] = format
        return self._client.chat(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def _provider_counts(cls, response: object) -> tuple[int, int] | None:
        input_tokens = cls._field(response, "prompt_eval_count", None)
        output_tokens = cls._field(response, "eval_count", None)
        if (
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and input_tokens >= 0
            and isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
            and output_tokens >= 0
        ):
            return input_tokens, output_tokens
        return None

    def _usage_for_response(
        self,
        response: object,
        *,
        input_tokens: int,
        output_text: str,
    ) -> tuple[LLMTokenUsage, str]:
        counts = self._provider_counts(response)
        if counts is not None:
            return LLMTokenUsage.from_counts(
                input_tokens=counts[0],
                output_tokens=counts[1],
            ), "sdk"
        return LLMTokenUsage.from_counts(
            input_tokens=input_tokens,
            output_tokens=self.estimate_tokens_for_text(
                output_text,
                model_hint=self.model_name_for_token_estimation,
            ),
        ), "estimate"

    def _build_response(
        self,
        *,
        content: str,
        usage: LLMTokenUsage,
        usage_source: str,
        finish_reason: LLMFinishReason = LLMFinishReason.COMPLETED,
        tool_calls: tuple[LLMToolCall, ...] = (),
    ) -> LLMAdapterResponse:
        return build_adapter_response(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            model=self.model,
            provider=self._provider_slug(),
            tool_calls=tool_calls,
            provider_extensions=LLMProviderExtensions(usage_source=usage_source),  # type: ignore[arg-type]
        )

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        response: LLMAdapterResponse | None = None
        success = False
        error_type: str | None = None
        input_tokens = 0
        try:
            input_tokens = self.estimate_tokens_for_messages(
                messages,
                model_hint=self.model_name_for_token_estimation,
            )
            native_response = self._chat(
                self._map_messages(messages),
                stream=False,
                options=self._generation_options(
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
            content = self._response_content(native_response)
            usage, usage_source = self._usage_for_response(
                native_response,
                input_tokens=input_tokens,
                output_text=content,
            )
            response = self._build_response(
                content=content,
                usage=usage,
                usage_source=usage_source,
            )
            success = True
            return response
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            usage = (
                response.usage
                if success and response is not None and response.usage is not None
                else LLMTokenUsage()
            )
            self.usage.end_call(
                call,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                success=success,
                error_type=error_type,
            )

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        success = False
        error_type: str | None = None
        input_tokens = 0
        output_tokens = 0
        buffer: list[str] = []
        emitted_partial = False
        last_response: object | None = None
        try:
            input_tokens = self.estimate_tokens_for_messages(
                messages,
                model_hint=self.model_name_for_token_estimation,
            )
            mapped_messages = self._map_messages(messages)
            options = self._generation_options(
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                stream = self._chat(
                    mapped_messages,
                    stream=True,
                    options=options,
                )
                for chunk in stream:  # type: ignore[union-attr]
                    last_response = chunk
                    text = self._response_content(chunk)
                    if text:
                        buffer.append(text)
                        emitted_partial = True
                        yield partial_stream_event(delta_content=text)
            except Exception:
                if emitted_partial:
                    raise
                fallback = self._chat(
                    mapped_messages,
                    stream=False,
                    options=options,
                )
                last_response = fallback
                text = self._response_content(fallback)
                if text:
                    buffer.append(text)
                    emitted_partial = True
                    yield partial_stream_event(delta_content=text)

            content = "".join(buffer)
            usage, usage_source = self._usage_for_response(
                last_response,
                input_tokens=input_tokens,
                output_text=content,
            )
            output_tokens = usage.output_tokens
            success = True
            yield final_stream_event(
                response=self._build_response(
                    content=content,
                    usage=usage,
                    usage_source=usage_source,
                )
            )
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            self.usage.end_call(
                call,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=success,
                error_type=error_type,
            )

    @property
    def model_capabilities(self) -> OllamaModelCapabilities:
        if self._model_capabilities is None:
            self._model_capabilities = self._capability_resolver.resolve(self.model)
        return self._model_capabilities

    def refresh_model_capabilities(self) -> OllamaModelCapabilities:
        self._model_capabilities = self._capability_resolver.resolve(self.model)
        return self._model_capabilities

    def supports_streaming(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return self.model_capabilities.supports_tools

    def supports_structured_output(self) -> bool:
        return True

    @classmethod
    def _provider_tool_calls(cls, response: object) -> tuple[LLMToolCall, ...]:
        message = cls._field(response, "message", response)
        raw_calls = cls._field(message, "tool_calls", [])
        if raw_calls is None:
            return ()
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
            raise ValueError("Ollama returned malformed native tool calls")

        calls: list[LLMToolCall] = []
        for raw_call in raw_calls:
            function = cls._field(raw_call, "function", _MISSING)
            if function is _MISSING or function is None:
                raise ValueError("Ollama returned malformed native tool call")
            name = cls._field(function, "name", None)
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Ollama returned invalid native tool call name")
            raw_arguments = cls._field(function, "arguments", _MISSING)
            if raw_arguments is _MISSING:
                raise ValueError("Ollama returned malformed native tool arguments")
            arguments = cls._arguments_object(raw_arguments)
            arguments_json = json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            calls.append(
                LLMToolCall(
                    id=str(cls._field(raw_call, "id", "") or ""),
                    name=name,
                    arguments_json=arguments_json,
                )
            )
        return finalize_accepted_tool_call_identities(calls)

    @staticmethod
    def _estimate_tool_output_text(
        content: str,
        tool_calls: tuple[LLMToolCall, ...],
    ) -> str:
        parts = [content]
        for tool_call in tool_calls:
            parts.extend((tool_call.name, tool_call.arguments_json))
        return "".join(parts)

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        if not self.supports_tools():
            raise ValueError(
                f"Ollama model does not declare native tool support: {self.model}"
            )
        self._validate_ollama_tool_choice(tool_choice)

        call = self.usage.begin_call(run_id=run_id, adapter=self)
        response: LLMAdapterResponse | None = None
        success = False
        error_type: str | None = None
        input_tokens = 0
        try:
            input_tokens = self.estimate_tokens_for_messages(
                messages,
                model_hint=self.model_name_for_token_estimation,
            )
            native_response = self._execute(
                lambda: self._chat(
                    self._map_messages(messages),
                    stream=False,
                    options=self._generation_options(
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    tools=tools_schema,  # type: ignore[arg-type]
                )
            )
            invalid_tool_calls = self._field(native_response, "invalid_tool_calls", None)
            if invalid_tool_calls:
                raise ValueError("Ollama returned invalid native tool calls")
            tool_calls = self._provider_tool_calls(native_response)
            content = self._response_content(native_response)
            output_text = self._estimate_tool_output_text(content, tool_calls)
            usage, usage_source = self._usage_for_response(
                native_response,
                input_tokens=input_tokens,
                output_text=output_text,
            )
            response = self._build_response(
                content=content,
                finish_reason=(
                    LLMFinishReason.TOOL_CALLS
                    if tool_calls
                    else LLMFinishReason.COMPLETED
                ),
                usage=usage,
                usage_source=usage_source,
                tool_calls=tool_calls,
            )
            success = True
            return response
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            usage = (
                response.usage
                if success and response is not None and response.usage is not None
                else LLMTokenUsage()
            )
            self.usage.end_call(
                call,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                success=success,
                error_type=error_type,
            )

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMStructuredResult[Any]:
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        response: LLMAdapterResponse | None = None
        success = False
        error_type: str | None = None
        input_tokens = 0
        try:
            input_tokens = self.estimate_tokens_for_messages(
                messages,
                model_hint=self.model_name_for_token_estimation,
            )
            schema = prepare_ollama_generation_schema(output_model)
            native_response = self._chat(
                self._map_messages(messages),
                stream=False,
                options=self._generation_options(
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                format=schema,
            )
            raw_text = self._response_content(native_response)
            data = json.loads(raw_text)
            if hasattr(output_model, "model_validate"):
                parsed = output_model.model_validate(data)
            elif hasattr(output_model, "parse_obj"):
                parsed = output_model.parse_obj(data)
            else:
                parsed = self._validate_with_model(output_model, raw_text)

            usage, usage_source = self._usage_for_response(
                native_response,
                input_tokens=input_tokens,
                output_text=raw_text,
            )
            response = self._build_response(
                content=raw_text,
                usage=usage,
                usage_source=usage_source,
            )
            success = True
            return LLMStructuredResult(parsed=parsed, response=response)
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            usage = (
                response.usage
                if success and response is not None and response.usage is not None
                else LLMTokenUsage()
            )
            self.usage.end_call(
                call,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                success=success,
                error_type=error_type,
            )
