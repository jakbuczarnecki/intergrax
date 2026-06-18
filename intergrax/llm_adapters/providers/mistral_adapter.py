# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from intergrax.utils import attribute_access

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Union

from mistralai import Mistral
from mistralai.models import ChatCompletionResponse

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
from intergrax.llm_adapters.registry.context_window import init_adapter_context_window_tokens


# -----------------------------
# Typed streaming contracts
# -----------------------------
class _MistralDelta(Protocol):
    content: Optional[str]


class _MistralStreamChoice(Protocol):
    delta: _MistralDelta


class _MistralStreamChunk(Protocol):
    choices: List[_MistralStreamChoice]


class MistralChatAdapter(LLMAdapter):
    """
    Mistral adapter based on the official Mistral Python SDK (mistralai).

    - Uses Mistral (official client type).
    - Supports:
        - generate_messages
        - stream_messages
    - Native tools via Mistral Chat Completions API.
    """

    _MISTRAL_CONTEXT_WINDOWS: Dict[str, int] = {
        "mistral-small-latest": 32_000,
        "mistral-medium-latest": 32_000,
        "mistral-large-latest": 32_000,
        "codestral-latest": 32_000,
    }

    DEFAULT_MODEL = "mistral-large-latest"

    ENV_MODEL = "INTERGRAX_DEFAULT_MISTRAL_MODEL"
    ENV_API_KEY = "MISTRAL_API_KEY"

    def __init__(
        self,
        client: Optional[Mistral] = None,
        model: Optional[str] = None,
        **defaults,
    ):
        super().__init__()
        self._apply_defaults_call_config(defaults)

        env_model = os.getenv(self.ENV_MODEL)
        api_key = os.getenv(self.ENV_API_KEY)

        resolved_model = model or env_model or self.DEFAULT_MODEL

        if client is None:

            if not api_key:
                raise RuntimeError(
                    "MISTRAL_API_KEY not found in environment variables."
                )

            client = Mistral(api_key=api_key)

        self.client: Mistral = client
        self.model: str = resolved_model

        self.model_name_for_token_estimation = self.model
        self.defaults = defaults

        self._context_window_tokens: int = init_adapter_context_window_tokens(
            provider=LLMProvider.MISTRAL,
            model=self.model,
            constructor_kwargs=defaults,
            legacy_windows=self._MISTRAL_CONTEXT_WINDOWS,
        )

        self.provider = LLMProvider.MISTRAL

    @property
    def context_window_tokens(self) -> int:
        """
        Cached maximum context window (input + output tokens) for the configured model.
        """
        return self._context_window_tokens

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

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
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            system_text, convo = split_system_messages(messages)

            payload = self._build_chat_params(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            res: ChatCompletionResponse = self._execute(
                lambda: self.client.chat.complete(**payload)
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

            stream: Iterable[_MistralStreamChunk] = self._execute(
                lambda: self.client.chat.complete(**payload)
            )

            buf: List[str] = []

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
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



    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _estimate_mistral_context_window(self, model: str) -> int:
        return self._MISTRAL_CONTEXT_WINDOWS.get(model, 32_000)

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
            res: ChatCompletionResponse = self._execute(
                lambda: self.client.chat.complete(**payload)
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
        in_tok = 0
        out_tok = 0
        buf: List[str] = []
        tool_calls_acc: List[Dict[str, Any]] = []

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
            stream: Iterable[_MistralStreamChunk] = self._execute(
                lambda: self.client.chat.complete(**payload)
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    buf.append(delta.content)
                    yield partial_stream_event(delta_content=delta.content)
                raw_tc = attribute_access.optional(delta, "tool_calls", None)
                if raw_tc:
                    for tc in raw_tc:
                        fn = attribute_access.optional(tc, "function", None)
                        name = attribute_access.optional(fn, "name", None) if fn else None
                        args = attribute_access.optional(fn, "arguments", None) if fn else None
                        if name:
                            tool_calls_acc.append(
                                {
                                    "id": attribute_access.optional(tc, "id", "") or "",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": args if isinstance(args, str) else json.dumps(args or {}),
                                    },
                                }
                            )
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
        schema = self._model_json_schema(output_model)
        schema_msg = ChatMessage(
            role="user",
            content=(
                "Return ONLY a single JSON object matching this JSON Schema:\n"
                + json.dumps(schema, ensure_ascii=False)
            ),
        )
        adapter_response = self.generate_messages(
            list(messages) + [schema_msg],
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )
        raw = adapter_response.content
        json_str = self._extract_json_object(raw) or raw.strip()
        parsed = self._validate_with_model(output_model, json_str)
        return LLMStructuredResult(parsed=parsed, response=adapter_response)

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
    ) -> dict:
        """
        Build a minimal, explicit Mistral Chat payload.

        We force response_format={"type":"text"} to keep extraction deterministic
        (content as a plain string, not a list of chunks).
        """
        temp = temperature if temperature is not None else self.defaults.get("temperature", None)
        out_tokens = max_tokens if max_tokens is not None else self.defaults.get("max_tokens", None)

        mapped = self._map_messages(system_text=system_text, convo=convo)

        payload: dict = {
            "model": self.model,
            "messages": mapped,
            "stream": stream,
        }
        if not tools:
            payload["response_format"] = {"type": "text"}

        if temp is not None:
            payload["temperature"] = float(temp)
        if out_tokens is not None:
            payload["max_tokens"] = int(out_tokens)
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        return payload

    def _map_messages(self, *, system_text: str, convo: Sequence[ChatMessage]) -> List[dict]:
        return map_chat_completion_messages(system_text=system_text, convo=convo)
