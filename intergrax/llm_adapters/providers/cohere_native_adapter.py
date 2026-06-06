# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Cohere API v2 native adapter (ClientV2 chat / chat_stream)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import cohere

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
    final_stream_event,
    partial_stream_event,
)
from intergrax.llm_adapters._shared.messages import split_system_messages
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.provider_extensions import LLMProviderExtensions
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall, tool_calls_from_openai_dicts


class CohereNativeChatAdapter(LLMAdapter):
    """Cohere ``ClientV2`` chat with optional tool definitions (v2 messages API)."""

    ENV_API_KEY = "COHERE_API_KEY"
    ENV_MODEL = "INTERGRAX_DEFAULT_COHERE_NATIVE_MODEL"
    DEFAULT_MODEL = "command-r-plus-08-2024"

    _CONTEXT_WINDOWS: Dict[str, int] = {
        "command-r-plus-08-2024": 128_000,
        "command-r-08-2024": 128_000,
    }

    def __init__(
        self,
        client: Optional[cohere.ClientV2] = None,
        model: Optional[str] = None,
        **defaults: Any,
    ) -> None:
        super().__init__()
        self._apply_defaults_call_config(defaults)
        api_key = os.getenv(self.ENV_API_KEY)
        if client is None and not api_key:
            raise RuntimeError(f"{self.ENV_API_KEY} not found in environment variables.")
        self.client = client or cohere.ClientV2(api_key=api_key)
        self.model = model or os.getenv(self.ENV_MODEL) or self.DEFAULT_MODEL
        self.provider = LLMProvider.COHERE_NATIVE
        self.defaults = defaults
        self.model_name_for_token_estimation = self.model
        self._context_window_tokens = int(self._CONTEXT_WINDOWS.get(self.model, 128_000))

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens

    def _map_messages(self, messages: Sequence[ChatMessage]) -> List[Dict[str, str]]:
        system_text, convo = split_system_messages(messages)
        out: List[Dict[str, str]] = []
        if system_text:
            out.append({"role": "system", "content": system_text})
        for m in convo:
            if m.role in {"user", "assistant"}:
                out.append({"role": m.role, "content": m.content or ""})
        return out

    def _parse_tool_blocks(self, blocks: Any) -> tuple[str, tuple[LLMToolCall, ...]]:
        text_parts: List[str] = []
        tool_calls_raw: List[Dict[str, Any]] = []
        for block in blocks or []:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", "") or "")
            if getattr(block, "type", None) == "tool_call":
                fn = getattr(block, "function", None)
                tool_calls_raw.append(
                    {
                        "id": getattr(block, "id", "") or "",
                        "type": "function",
                        "function": {
                            "name": getattr(fn, "name", "") if fn else "",
                            "arguments": json.dumps(getattr(fn, "arguments", {}) or {}),
                        },
                    }
                )
        return "".join(text_parts), tool_calls_from_openai_dicts(tool_calls_raw)

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
        in_tok = 0
        out_tok = 0
        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model))
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": self._map_messages(messages),
            }
            if temperature is not None:
                kwargs["temperature"] = float(temperature)
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            resp = self._execute(lambda: self.client.chat(**kwargs))
            text = ""
            if resp.message and resp.message.content:
                text = resp.message.content[0].text or ""
            out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model))
            response = build_adapter_response(
                content=text,
                usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                model=self.model,
                provider=self._provider_slug(),
                provider_extensions=LLMProviderExtensions(usage_source="estimate"),
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
        in_tok = 0
        out_tok = 0
        buf: List[str] = []
        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model))
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": self._map_messages(messages),
            }
            if temperature is not None:
                kwargs["temperature"] = float(temperature)
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            stream = self._execute(lambda: self.client.chat_stream(**kwargs))
            for event in stream:
                if getattr(event, "type", None) == "content-delta":
                    txt = event.delta.message.content.text or ""
                    if txt:
                        buf.append(txt)
                        yield partial_stream_event(delta_content=txt)
            out_tok = int(self.estimate_tokens_for_text("".join(buf), model_hint=self.model))
            success = True
            yield final_stream_event(
                response=build_adapter_response(
                    content="".join(buf),
                    usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                    model=self.model,
                    provider=self._provider_slug(),
                    provider_extensions=LLMProviderExtensions(usage_source="estimate"),
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
        del tool_choice
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        response: LLMAdapterResponse | None = None
        success = False
        err_type = None
        in_tok = 0
        out_tok = 0
        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model))
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": self._map_messages(messages),
                "tools": self._cohere_tools_payload(tools_schema),
            }
            if temperature is not None:
                kwargs["temperature"] = float(temperature)
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            resp = self._execute(lambda: self.client.chat(**kwargs))
            text = ""
            tool_calls: tuple[LLMToolCall, ...] = ()
            if resp.message and resp.message.content:
                text, tool_calls = self._parse_tool_blocks(resp.message.content)
            out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model))
            finish = LLMFinishReason.TOOL_CALLS if tool_calls else LLMFinishReason.COMPLETED
            response = build_adapter_response(
                content=text,
                finish_reason=finish,
                usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                model=self.model,
                provider=self._provider_slug(),
                tool_calls=tool_calls,
                provider_extensions=LLMProviderExtensions(usage_source="estimate"),
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

    def _cohere_tools_payload(self, tools_schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": (t.get("function") or {}).get("name"),
                    "description": (t.get("function") or {}).get("description") or "",
                    "parameters": (t.get("function") or {}).get("parameters") or {},
                },
            }
            for t in tools_schema
            if t.get("type") == "function"
        ]

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
        del tool_choice
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        success = False
        err_type = None
        in_tok = 0
        out_tok = 0
        buf: List[str] = []
        tool_calls: tuple[LLMToolCall, ...] = ()

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model))
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": self._map_messages(messages),
                "tools": self._cohere_tools_payload(tools_schema),
            }
            if temperature is not None:
                kwargs["temperature"] = float(temperature)
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)

            stream = self._execute(lambda: self.client.chat_stream(**kwargs))
            for event in stream:
                if getattr(event, "type", None) == "content-delta":
                    txt = event.delta.message.content.text or ""
                    if txt:
                        buf.append(txt)
                        yield partial_stream_event(delta_content=txt)
                if getattr(event, "type", None) == "message-end":
                    msg = getattr(event, "message", None)
                    if msg and getattr(msg, "content", None):
                        _, tool_calls = self._parse_tool_blocks(msg.content)

            if not buf and not tool_calls:
                result = self.generate_with_tools(
                    messages,
                    tools_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    run_id=run_id,
                )
                yield final_stream_event(response=result)
                success = True
                return

            out_tok = int(self.estimate_tokens_for_text("".join(buf), model_hint=self.model))
            finish = LLMFinishReason.TOOL_CALLS if tool_calls else LLMFinishReason.COMPLETED
            success = True
            yield final_stream_event(
                response=build_adapter_response(
                    content="".join(buf),
                    finish_reason=finish,
                    usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                    model=self.model,
                    provider=self._provider_slug(),
                    tool_calls=tool_calls,
                    provider_extensions=LLMProviderExtensions(usage_source="estimate"),
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
