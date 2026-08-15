# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from intergrax.utils import attribute_access

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from anthropic import Anthropic

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
    final_stream_event,
    partial_stream_event,
)
from intergrax.llm_adapters._shared.anthropic_messages import (
    extract_anthropic_text,
    extract_anthropic_tool_calls,
    map_anthropic_messages,
)
from intergrax.llm_adapters._shared.messages import split_system_messages
from intergrax.llm_adapters._shared.tool_schema import openai_tools_to_anthropic
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason, parse_finish_reason
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.context_window import init_adapter_context_window_tokens
from intergrax.llm_adapters.contracts.provider_extensions import LLMProviderExtensions
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.contracts.tool_call import tool_calls_from_openai_dicts


class ClaudeChatAdapter(LLMAdapter):
    """
    Claude (Anthropic) adapter based on the official anthropic Python SDK.

    Contract (aligned with OpenAI adapter pattern):
      - __init__(client: Optional[Anthropic] = None, model: Optional[str] = None, **defaults)
      - generate_messages(...) -> LLMAdapterResponse
      - stream_messages(...)   -> Iterable[LLMStreamEvent]

    Supports native tools (Anthropic tool_use) and streaming.
    """

    _CLAUDE_CONTEXT_WINDOWS: Dict[str, int] = {
        "claude-3-5-sonnet-latest": 200_000,
        "claude-3-5-haiku-latest": 200_000,
    }

    DEFAULT_MODEL = "claude-3-5-sonnet-latest"
    ENV_API_KEY = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        client: Optional[Anthropic] = None,
        model: Optional[str] = None,
        **defaults,
    ):
        super().__init__()
        self._apply_defaults_call_config(defaults)

        api_key = os.getenv(self.ENV_API_KEY)

        resolved_model = model or self.DEFAULT_MODEL

        if client is None:
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not found in environment variables."
                )
            client = Anthropic(api_key=api_key)

        self.client: Anthropic = client
        self.model: str = resolved_model
        self.defaults = defaults
        self.model_name_for_token_estimation: str = self.model
        self._context_window_tokens: int = init_adapter_context_window_tokens(
            provider=LLMProvider.CLAUDE,
            model=self.model,
            constructor_kwargs=defaults,
            legacy_windows=self._CLAUDE_CONTEXT_WINDOWS,
        )
        self.provider = LLMProvider.CLAUDE

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens

    def _estimate_extensions(self) -> LLMProviderExtensions:
        return LLMProviderExtensions(usage_source="estimate")

    def _build_response_from_message(
        self,
        resp: Any,
        *,
        content: str,
        tool_calls_raw: List[Dict[str, Any]] | None = None,
        in_tok: int = 0,
        out_tok: int = 0,
    ) -> LLMAdapterResponse:
        tool_calls = tool_calls_from_openai_dicts(tool_calls_raw or extract_anthropic_tool_calls(resp))
        finish = parse_finish_reason(attribute_access.optional(resp, "stop_reason", None))
        if tool_calls and finish == LLMFinishReason.COMPLETED:
            finish = LLMFinishReason.TOOL_CALLS
        return build_adapter_response(
            content=content,
            finish_reason=finish,
            usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
            model=self.model,
            provider=self._provider_slug(),
            response_id=str(attribute_access.optional(resp, "id", "") or "") or None,
            tool_calls=tool_calls,
            provider_extensions=self._estimate_extensions(),
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
        err_type = None
        in_tok = 0
        out_tok = 0

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))
            system_text, convo = split_system_messages(messages)
            payload_msgs = map_anthropic_messages(convo)
            temp = temperature if temperature is not None else self.defaults.get("temperature", None)
            out_tokens = max_tokens if max_tokens is not None else self.defaults.get("max_tokens", None)
            if out_tokens is None:
                out_tokens = 1024

            resp = self._execute(
                lambda: self.client.messages.create(
                    model=self.model,
                    system=system_text or None,
                    messages=payload_msgs,
                    max_tokens=int(out_tokens),
                    temperature=float(temp) if temp is not None else None,
                )
            )

            text = extract_anthropic_text(resp)
            out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model_name_for_token_estimation))
            if attribute_access.optional(resp, "usage", None):
                in_tok = int(resp.usage.input_tokens or in_tok)
                out_tok = int(resp.usage.output_tokens or out_tok)

            response = self._build_response_from_message(resp, content=text, in_tok=in_tok, out_tok=out_tok)
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
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))
            system_text, convo = split_system_messages(messages)
            payload_msgs = map_anthropic_messages(convo)
            temp = temperature if temperature is not None else self.defaults.get("temperature", None)
            out_tokens = max_tokens if max_tokens is not None else self.defaults.get("max_tokens", None)
            if out_tokens is None:
                out_tokens = 1024

            stream = self._execute(
                lambda: self.client.messages.create(
                    model=self.model,
                    system=system_text or None,
                    messages=payload_msgs,
                    max_tokens=int(out_tokens),
                    temperature=float(temp) if temp is not None else None,
                    stream=True,
                )
            )

            for event in stream:
                if event.type != "content_block_delta":
                    continue
                delta = event.delta
                if not hasattr(delta, "type") or delta.type != "text_delta":
                    continue
                txt = delta.text or ""
                if txt:
                    buf.append(txt)
                    yield partial_stream_event(delta_content=txt)

            out_tok = int(self.estimate_tokens_for_text("".join(buf), model_hint=self.model_name_for_token_estimation))
            success = True
            yield final_stream_event(
                response=build_adapter_response(
                    content="".join(buf),
                    usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                    model=self.model,
                    provider=self._provider_slug(),
                    provider_extensions=self._estimate_extensions(),
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
        in_tok = 0
        out_tok = 0

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))
            system_text, convo = split_system_messages(messages)
            tools = openai_tools_to_anthropic(tools_schema)
            temp = temperature if temperature is not None else self.defaults.get("temperature")
            out_tokens = max_tokens if max_tokens is not None else self.defaults.get("max_tokens", 1024)

            kwargs: Dict[str, Any] = dict(
                model=self.model,
                system=system_text or None,
                messages=map_anthropic_messages(convo),
                max_tokens=int(out_tokens or 1024),
                tools=tools,
            )
            if temp is not None:
                kwargs["temperature"] = float(temp)
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

            resp = self._execute(lambda: self.client.messages.create(**kwargs))
            if attribute_access.optional(resp, "usage", None):
                in_tok = int(resp.usage.input_tokens or in_tok)
                out_tok = int(resp.usage.output_tokens or 0)

            content = extract_anthropic_text(resp)
            response = self._build_response_from_message(resp, content=content, in_tok=in_tok, out_tok=out_tok)
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
        in_tok = 0
        out_tok = 0
        buf: List[str] = []

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))
            system_text, convo = split_system_messages(messages)
            tools = openai_tools_to_anthropic(tools_schema)
            temp = temperature if temperature is not None else self.defaults.get("temperature")
            out_tokens = max_tokens if max_tokens is not None else self.defaults.get("max_tokens", 1024)

            kwargs: Dict[str, Any] = dict(
                model=self.model,
                system=system_text or None,
                messages=map_anthropic_messages(convo),
                max_tokens=int(out_tokens or 1024),
                tools=tools,
                stream=True,
            )
            if temp is not None:
                kwargs["temperature"] = float(temp)
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

            stream = self._execute(lambda: self.client.messages.create(**kwargs))
            for event in stream:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if attribute_access.optional(delta, "type", None) == "text_delta":
                        txt = delta.text or ""
                        if txt:
                            buf.append(txt)
                            yield partial_stream_event(delta_content=txt)

            get_final = attribute_access.optional(stream, "get_final_message", None)
            resp = get_final() if callable(get_final) else None
            if resp is None:
                result = self.generate_with_tools(
                    messages,
                    tools_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tool_choice=tool_choice,
                    run_id=run_id,
                )
                yield final_stream_event(response=result)
                return

            if attribute_access.optional(resp, "usage", None):
                in_tok = int(resp.usage.input_tokens or in_tok)
                out_tok = int(resp.usage.output_tokens or 0)

            content = extract_anthropic_text(resp) or "".join(buf)
            response = self._build_response_from_message(resp, content=content, in_tok=in_tok, out_tok=out_tok)
            success = True
            yield final_stream_event(response=response)
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
        schema = self._model_json_schema(output_model)
        schema_msg = ChatMessage(
            role="user",
            content=(
                "Return ONLY a single JSON object matching this JSON Schema:\n"
                + json.dumps(schema, ensure_ascii=False)
            ),
        )
        extended = list(messages) + [schema_msg]
        adapter_response = self.generate_messages(
            extended,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )
        raw = adapter_response.content
        json_str = self._extract_json_object(raw) or raw.strip()
        parsed = self._validate_with_model(output_model, json_str)
        return LLMStructuredResult(parsed=parsed, response=adapter_response)
