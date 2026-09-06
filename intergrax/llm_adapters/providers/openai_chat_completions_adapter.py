# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""
OpenAI Chat Completions API adapter for OpenAI-compatible HTTP endpoints.

Used by Groq, vLLM, and similar providers (same message/tools/stream shape).
"""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
    final_stream_event,
    partial_stream_event,
)
from intergrax.llm_adapters._shared.messages import map_chat_completion_messages, split_system_messages
from intergrax.llm_adapters._shared.openai_completion_mapping import (
    adapter_response_from_openai_chat_completion,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.contracts.tool_call import tool_calls_from_openai_dicts
from intergrax.llm_adapters.providers._openai_schema import prepare_openai_strict_generation_schema
from intergrax.llm_adapters.registry.context_window import init_adapter_context_window_tokens


class OpenAIChatCompletionsAdapter(LLMAdapter):
    """Chat Completions via ``openai.OpenAI`` (Groq, vLLM, local gateways)."""

    _CONTEXT_WINDOWS: Dict[str, int] = {}

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        provider: LLMProvider,
        context_windows: Optional[Dict[str, int]] = None,
        **defaults: Any,
    ) -> None:
        super().__init__()
        self._apply_defaults_call_config(defaults)
        self.client = client
        self.model = model
        self.provider = provider
        self.defaults = defaults
        self.model_name_for_token_estimation = model
        windows = context_windows if context_windows is not None else self._CONTEXT_WINDOWS
        self._context_window_tokens = init_adapter_context_window_tokens(
            provider=provider,
            model=model,
            constructor_kwargs=defaults,
            legacy_windows=windows,
        )

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens

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
        err_type = None
        try:
            system_text, convo = split_system_messages(messages)
            payload = self._build_chat_params(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            res: ChatCompletion = self._execute(
                lambda: self.client.chat.completions.create(**payload)
            )
            response = adapter_response_from_openai_chat_completion(
                res,
                model=self.model,
                provider=self._provider_slug(),
            )
            success = True
            return response
        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            usage = response.usage if (success and response and response.usage) else LLMTokenUsage()
            self.usage.end_call(
                call,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                success=success,
                error_type=err_type,
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
        err_type = None
        buf: List[str] = []
        in_tok = 0
        out_tok = 0
        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))
            system_text, convo = split_system_messages(messages)
            payload = self._build_chat_params(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            stream = self._execute(lambda: self.client.chat.completions.create(**payload))
            for chunk in stream:
                c: ChatCompletionChunk = chunk
                if not c.choices:
                    continue
                delta = c.choices[0].delta
                if delta and delta.content:
                    buf.append(delta.content)
                    yield partial_stream_event(delta_content=delta.content)
            out_tok = int(self.estimate_tokens_for_text("".join(buf), model_hint=self.model_name_for_token_estimation))
            success = True
            yield final_stream_event(
                response=build_adapter_response(
                    content="".join(buf),
                    usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                    model=self.model,
                    provider=self._provider_slug(),
                )
            )
        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            self.usage.end_call(
                call,
                input_tokens=in_tok,
                output_tokens=out_tok,
                success=success,
                error_type=err_type,
            )

    def supports_tools(self) -> bool:
        return True

    def supports_structured_output(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return True

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
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        response: LLMAdapterResponse | None = None
        success = False
        err_type = None
        try:
            system_text, convo = split_system_messages(messages)
            payload = self._build_chat_params(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                tools=tools_schema,
                tool_choice=tool_choice,
            )
            res: ChatCompletion = self._execute(
                lambda: self.client.chat.completions.create(**payload)
            )
            response = adapter_response_from_openai_chat_completion(
                res,
                model=self.model,
                provider=self._provider_slug(),
            )
            success = True
            return response
        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            usage = response.usage if (success and response and response.usage) else LLMTokenUsage()
            self.usage.end_call(
                call,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                success=success,
                error_type=err_type,
            )

    def stream_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        success = False
        err_type = None
        buf: List[str] = []
        tool_calls_acc: List[Dict[str, Any]] = []
        in_tok = 0
        out_tok = 0
        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))
            system_text, convo = split_system_messages(messages)
            payload = self._build_chat_params(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                tools=tools_schema,
                tool_choice=tool_choice,
            )
            stream = self._execute(lambda: self.client.chat.completions.create(**payload))
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    buf.append(delta.content)
                    yield partial_stream_event(delta_content=delta.content)
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index or 0
                        while len(tool_calls_acc) <= idx:
                            tool_calls_acc.append(
                                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                            )
                        acc = tool_calls_acc[idx]
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                acc["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                acc["function"]["arguments"] += tc.function.arguments
            tool_calls = tool_calls_from_openai_dicts(tool_calls_acc)
            finish = LLMFinishReason.TOOL_CALLS if tool_calls else LLMFinishReason.COMPLETED
            out_tok = int(self.estimate_tokens_for_text("".join(buf), model_hint=self.model_name_for_token_estimation))
            success = True
            yield final_stream_event(
                response=build_adapter_response(
                    content="".join(buf),
                    finish_reason=finish,
                    usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                    model=self.model,
                    provider=self._provider_slug(),
                    tool_calls=tool_calls,
                )
            )
        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            self.usage.end_call(
                call,
                input_tokens=in_tok,
                output_tokens=out_tok,
                success=success,
                error_type=err_type,
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
        system_text, convo = split_system_messages(messages)
        schema = prepare_openai_strict_generation_schema(output_model)
        payload = self._build_chat_params(
            system_text=system_text,
            convo=convo,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": attribute_access.optional(output_model, "__name__", "structured_output"),
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        res = self._execute(lambda: self.client.chat.completions.create(**payload))
        response = adapter_response_from_openai_chat_completion(
            res,
            model=self.model,
            provider=self._provider_slug(),
        )
        raw = response.content
        json_str = self._extract_json_object(raw) or raw.strip()
        parsed = self._validate_with_model(output_model, json_str)
        return LLMStructuredResult(parsed=parsed, response=response)

    def _build_chat_params(
        self,
        *,
        system_text: str,
        convo: Sequence[ChatMessage],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> dict:
        temp = temperature if temperature is not None else self.defaults.get("temperature")
        out_tokens = max_tokens if max_tokens is not None else self.defaults.get("max_tokens")
        include_multimodal = self.supports_vision() and _conversation_has_attachments(convo)
        mapped = map_chat_completion_messages(
            system_text=system_text,
            convo=convo,
            include_multimodal=include_multimodal,
        )
        payload: dict = {"model": self.model, "messages": mapped, "stream": stream}
        if temp is not None:
            payload["temperature"] = float(temp)
        if out_tokens is not None:
            payload["max_tokens"] = int(out_tokens)
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        return payload


def _conversation_has_attachments(convo: Sequence[ChatMessage]) -> bool:
    return any(message.attachments for message in convo)
