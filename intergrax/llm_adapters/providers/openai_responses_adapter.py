# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.


from __future__ import annotations
from intergrax.utils import attribute_access
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from openai import Client
from openai.types.responses import Response

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
    final_stream_event,
    partial_stream_event,
)
from intergrax.llm_adapters._shared.responses_input import messages_to_responses_input
from intergrax.llm_adapters._shared.openai_completion_mapping import adapter_response_from_openai_responses
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.contracts.tool_call import tool_calls_from_openai_dicts
from intergrax.llm_adapters.registry.context_window import init_adapter_context_window_tokens

# OpenAI SDK Client(...) kwargs — must never reach responses.create/stream.
_OPENAI_CLIENT_CONSTRUCTOR_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "organization",
        "project",
        "base_url",
        "timeout",
        "max_retries",
        "default_headers",
        "default_query",
        "http_client",
        "_strict_response_validation",
    }
)

# Adapter/profile identity and LLMCallConfig resilience — not Responses API request kwargs.
_OPENAI_ADAPTER_ONLY_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "client",
        "context_window_tokens",
        "max_tokens",
        "timeout_sec",
        "max_retries",
        "retry_backoff_sec",
        "retry_on_status",
        "calls_per_minute",
        "circuit_breaker_threshold",
        "circuit_breaker_cooldown_sec",
        "use_distributed_rate_limit",
    }
)


def _partition_openai_responses_options(
    defaults: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split profile/constructor kwargs into Client(...) vs responses.create/stream defaults."""
    client_kwargs: dict[str, Any] = {}
    request_defaults: dict[str, Any] = {}
    for key, value in defaults.items():
        if key in _OPENAI_CLIENT_CONSTRUCTOR_KEYS:
            client_kwargs[key] = value
        elif key in _OPENAI_ADAPTER_ONLY_KEYS:
            continue
        else:
            request_defaults[key] = value
    return client_kwargs, request_defaults


def _map_tools_to_responses_api(
    tools_schema: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map canonical Chat-Completions-style tool schema to Responses API shape."""
    mapped: List[Dict[str, Any]] = []
    for index, tool in enumerate(tools_schema):
        if not isinstance(tool, dict):
            raise ValueError(
                "OpenAI Responses adapter: "
                f"tools_schema[{index}] must be a dict, got {type(tool).__name__}"
            )

        if tool.get("type") != "function":
            mapped.append(dict(tool))
            continue

        if "function" not in tool:
            if isinstance(tool.get("name"), str) and tool["name"]:
                mapped.append(dict(tool))
                continue
            raise ValueError(
                "OpenAI Responses adapter: "
                f"tools_schema[{index}] function tool requires nested 'function' "
                "object or top-level 'name'"
            )

        fn = tool.get("function")
        if not isinstance(fn, dict):
            raise ValueError(
                "OpenAI Responses adapter: "
                f"tools_schema[{index}].function must be a dict, "
                f"got {type(fn).__name__ if fn is not None else 'missing'}"
            )

        name = fn.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                "OpenAI Responses adapter: "
                f"tools_schema[{index}].function.name must be a non-empty string"
            )

        out: Dict[str, Any] = {"type": "function", "name": name}
        if "description" in fn:
            out["description"] = fn["description"]
        if "parameters" in fn:
            out["parameters"] = fn["parameters"]
        if "strict" in fn:
            out["strict"] = fn["strict"]
        mapped.append(out)
    return mapped


class OpenAIChatResponsesAdapter(LLMAdapter):
    """
    OpenAI adapter based on the new Responses API.

    Public interface is compatible with the previous Chat Completions adapter:
    - generate_messages
    - stream_messages
    - generate_with_tools
    - stream_with_tools
    - generate_structured
    """

    # Conservative context window estimates for common OpenAI models.
    # For unknown models we fall back to a small, safe default.
    _OPENAI_CONTEXT_WINDOWS: Dict[str, int] = {
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "gpt-4o-2024-08-06": 128_000,
        "gpt-3.5-turbo": 16_385,
        "gpt-3.5-turbo-0301": 16_385,
        "gpt-4.1": 1_000_000,
        "gpt-4.1-mini": 1_000_000,
        "gpt-4.1-nano": 1_000_000,
        "gpt-5": 400_000,
        "gpt-5-mini": 400_000,
    }


    def _estimate_openai_context_window(self, model: str) -> int:
        """
        Best-effort context window estimation for OpenAI models.

        The result is used once at adapter construction time and then cached
        in a private attribute.
        """
        name = (model or "").strip()
        base = name.split(":", 1)[0]  # strip possible snapshot suffixes

        if base in self._OPENAI_CONTEXT_WINDOWS:
            return self._OPENAI_CONTEXT_WINDOWS[base]

        # Conservative fallback for unknown models.
        return 128_000
    
    DEFAULT_MODEL = "gpt-5-mini"
    ENV_API_KEY = "OPENAI_API_KEY"

    def __init__(self, client: Optional[Client] = None, model: Optional[str] = None, **defaults):
        super().__init__()
        defaults.pop("model", None)
        defaults.pop("client", None)
        self._apply_defaults_call_config(defaults)

        resolved_model = model or self.DEFAULT_MODEL

        client_kwargs, self.request_defaults = _partition_openai_responses_options(defaults)

        if client is None:
            api_key = client_kwargs.pop("api_key", None) or os.getenv(self.ENV_API_KEY)
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not found in environment variables."
                )
            client = Client(api_key=api_key, **client_kwargs)

        self.client: Client = client
        self.model: str = resolved_model

        self.model_name_for_token_estimation = self.model
        self._context_window_tokens: int = init_adapter_context_window_tokens(
            provider=LLMProvider.OPENAI,
            model=self.model,
            constructor_kwargs=defaults,
            legacy_windows=self._OPENAI_CONTEXT_WINDOWS,
        )

        self.provider = LLMProvider.OPENAI


    @property
    def context_window_tokens(self) -> int:
        """
        Cached maximum context window (input + output tokens) for the
        configured OpenAI model. Computed once in __init__.
        """
        return self._context_window_tokens

    # ---------------------------------------------------------------------
    # INTERNAL HELPERS (PRIVATE METHODS)
    # ---------------------------------------------------------------------

    def _messages_to_responses_input(self, mapped_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return messages_to_responses_input(mapped_messages)

    def supports_streaming(self) -> bool:
        return True

    def supports_structured_output(self) -> bool:
        return True

    def _collect_output_text(self, response: Response) -> str:
        """
        Extract the assistant's output text from a Responses API result.

        Prefer response.output_text when available, otherwise aggregate
        all text blocks from response.output[*].content[*] where type == "output_text".
        """
        txt = response.output_text
        if txt:
            return txt
        
        chunks: List[str] = []
        for item in response.output or []:
            if item.type == "message":
                for c in item.content or []:
                    if c.type == "output_text":
                        chunks.append(c.text or "")
        return "".join(chunks)
        

    # ---------------------------------------------------------------------
    # PUBLIC: Plain chat
    # ---------------------------------------------------------------------

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        """
        Single-shot completion (non-streaming) using Responses API.
        """
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        response: LLMAdapterResponse | None = None
        success = False
        err_type = None

        try:
            mapped = self._map_messages_to_openai(messages)
            input_items = self._messages_to_responses_input(mapped)

            payload: Dict[str, Any] = dict(
                model=self.model,
                input=input_items,
            )
            if max_tokens is not None:
                payload["max_output_tokens"] = max_tokens

            api_response: Response = self._execute(
                lambda: self.client.responses.create(**payload, **self.request_defaults)
            )

            output_text = self._collect_output_text(api_response)
            response = adapter_response_from_openai_responses(
                api_response,
                model=self.model,
                provider=self._provider_slug(),
                content=output_text,
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
        """
        Streaming completion using Responses API.
        """
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        success = False
        err_type = None
        in_tok = 0
        out_tok = 0

        try:
            in_tok = int(self.count_messages_tokens(messages))
        except Exception:
            in_tok = 0

        buf: List[str] = []

        try:
            mapped = self._map_messages_to_openai(messages)
            input_items = self._messages_to_responses_input(mapped)

            payload: Dict[str, Any] = dict(
                model=self.model,
                input=input_items,
                stream=True,
            )

            if max_tokens is not None:
                payload["max_output_tokens"] = max_tokens

            with self.client.responses.stream(**payload, **self.request_defaults) as stream:
                for ev in stream:
                    if ev.type == "response.output_text.delta":
                        delta = ev.delta
                        if delta:
                            buf.append(delta)
                            yield partial_stream_event(delta_content=delta)

            full_text = "".join(buf)
            out_tok = int(self.estimate_tokens_for_text(full_text))
            success = True
            yield final_stream_event(
                response=build_adapter_response(
                    content=full_text,
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


    # ---------------------------------------------------------------------
    # PUBLIC: Tools
    # ---------------------------------------------------------------------

    def supports_tools(self) -> bool:
        """
        Signal to higher-level orchestration that this adapter supports tools.
        """
        return True

    def supports_vision(self) -> bool:
        return True

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
        """
        Stream assistant text deltas, then yield the final typed response.
        """
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        success = False
        err_type = None
        in_tok = 0
        out_tok = 0
        buf: List[str] = []

        try:
            try:
                in_tok = int(self.count_messages_tokens(messages))
            except Exception:
                in_tok = 0

            mapped = self._map_messages_to_openai(messages)
            input_items = self._messages_to_responses_input(mapped)
            responses_tools = _map_tools_to_responses_api(tools_schema)
            payload: Dict[str, Any] = dict(
                model=self.model,
                input=input_items,
                tools=responses_tools,
                stream=True,
            )
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
            if max_tokens is not None:
                payload["max_output_tokens"] = max_tokens

            with self.client.responses.stream(**payload, **self.request_defaults) as stream:
                for ev in stream:
                    if ev.type == "response.output_text.delta":
                        delta = ev.delta or ""
                        if delta:
                            buf.append(delta)
                            yield partial_stream_event(delta_content=delta)

                get_final = attribute_access.optional(stream, "get_final_response", None)
                resp = get_final() if callable(get_final) else None
                if resp is None:
                    raise RuntimeError("OpenAI responses stream did not return a final response")

                native_tool_calls = self._extract_tool_calls_from_response(resp)
                final_content = self._collect_output_text(resp) or "".join(buf)
                final_response = adapter_response_from_openai_responses(
                    resp,
                    model=self.model,
                    provider=self._provider_slug(),
                    content=final_content,
                    tool_calls=tool_calls_from_openai_dicts(native_tool_calls),
                )
                if resp.usage:
                    in_tok = int(resp.usage.input_tokens or in_tok)
                    out_tok = int(resp.usage.output_tokens or 0)
                success = True
                yield final_stream_event(response=final_response)
                return

        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            if not success and buf:
                try:
                    out_tok = int(self.estimate_tokens_for_text("".join(buf)))
                except Exception:
                    out_tok = 0
            self.usage.end_call(
                call,
                input_tokens=in_tok,
                output_tokens=out_tok,
                success=success,
                error_type=err_type,
            )

    def _extract_tool_calls_from_response(self, response: Response) -> List[Dict[str, Any]]:
        native_tool_calls: List[Dict[str, Any]] = []
        for item in response.output or []:
            if item.type != "function_call":
                continue
            args = item.arguments
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            native_tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": args},
                }
            )
        return native_tool_calls

    # ---------------------------------------------------------------------
    # PUBLIC: Structured JSON output
    # ---------------------------------------------------------------------

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
        err_type = None

        try:
            mapped = self._map_messages_to_openai(messages)
            input_items = self._messages_to_responses_input(mapped)
            schema = self._model_json_schema(output_model)
            payload: Dict[str, Any] = dict(
                model=self.model,
                input=input_items,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": attribute_access.optional(output_model, "__name__", "structured_output"),
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            if max_tokens is not None:
                payload["max_output_tokens"] = max_tokens

            api_response: Response = self._execute(
                lambda: self.client.responses.create(**payload, **self.request_defaults)
            )
            raw = self._collect_output_text(api_response)
            response = adapter_response_from_openai_responses(
                api_response,
                model=self.model,
                provider=self._provider_slug(),
                content=raw,
            )
            json_str = self._extract_json_object(raw) or raw.strip()
            parsed = self._validate_with_model(output_model, json_str)
            success = True
            return LLMStructuredResult(parsed=parsed, response=response)

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
        """
        Generate a response with potential function/tool calls.
        """
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        response: LLMAdapterResponse | None = None
        success = False
        err_type = None

        try:
            mapped = self._map_messages_to_openai(messages)
            input_items = self._messages_to_responses_input(mapped)
            responses_tools = _map_tools_to_responses_api(tools_schema)

            payload: Dict[str, Any] = dict(
                model=self.model,
                input=input_items,
                tools=responses_tools,
            )

            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

            if max_tokens is not None:
                payload["max_output_tokens"] = max_tokens

            api_response: Response = self._execute(
                lambda: self.client.responses.create(**payload, **self.request_defaults)
            )

            content = self._collect_output_text(api_response)
            native_tool_calls = self._extract_tool_calls_from_response(api_response)
            response = adapter_response_from_openai_responses(
                api_response,
                model=self.model,
                provider=self._provider_slug(),
                content=content or "",
                tool_calls=tool_calls_from_openai_dicts(native_tool_calls),
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

    

    def _map_messages_to_openai(self, msgs: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Map internal ChatMessage objects to OpenAI-compatible message dicts.

        Handles:
        - role/content
        - tool messages with tool_call_id/name
        - assistant messages with tool_calls[]
        """
        out: List[Dict[str, Any]] = []
        for m in msgs:
            d: Dict[str, Any] = {"role": m.role, "content": m.content}

            if m.role == "tool":
                if attribute_access.optional(m, "tool_call_id", None) is not None:
                    d["tool_call_id"] = m.tool_call_id
                if attribute_access.optional(m, "name", None) is not None:
                    d["name"] = m.name

            if attribute_access.optional(m, "tool_calls", None):
                d["tool_calls"] = m.tool_calls

            out.append(d)
        return out
