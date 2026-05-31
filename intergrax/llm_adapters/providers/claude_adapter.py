# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from anthropic import Anthropic

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.anthropic_messages import (
    extract_anthropic_text,
    extract_anthropic_tool_calls,
    map_anthropic_messages,
)
from intergrax.llm_adapters._shared.messages import split_system_messages
from intergrax.llm_adapters._shared.tool_results import make_tool_result
from intergrax.llm_adapters._shared.tool_schema import openai_tools_to_anthropic
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider


class ClaudeChatAdapter(LLMAdapter):
    """
    Claude (Anthropic) adapter based on the official anthropic Python SDK.

    Contract (aligned with OpenAI adapter pattern):
      - __init__(client: Optional[Anthropic] = None, model: Optional[str] = None, **defaults)
      - generate_messages(...) -> str
      - stream_messages(...)   -> Iterable[str]

    Supports native tools (Anthropic tool_use) and streaming.
    """

    # Conservative context window estimates (keep safe unless you add real token accounting).
    _CLAUDE_CONTEXT_WINDOWS: Dict[str, int] = {
        "claude-3-5-sonnet-latest": 200_000,
        "claude-3-5-haiku-latest": 200_000,
        # Add exact model ids used in your env as needed.
    }

    DEFAULT_MODEL = "claude-3-5-sonnet-latest"

    ENV_MODEL = "INTERGRAX_DEFAULT_CLAUDE_MODEL"
    ENV_API_KEY = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        client: Optional[Anthropic] = None,
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
                    "ANTHROPIC_API_KEY not found in environment variables."
                )

            client = Anthropic(api_key=api_key)

        self.client: Anthropic = client
        self.model: str = resolved_model

        self.defaults = defaults
        self.model_name_for_token_estimation: str = self.model
        self._context_window_tokens: int = self._CLAUDE_CONTEXT_WINDOWS.get(self.model, 32_000)

        self.provider = LLMProvider.CLAUDE

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
    ) -> str:
        call = self.usage.begin_call(run_id=run_id, adapter=self)

        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            system_text, convo = split_system_messages(messages)
            payload_msgs = map_anthropic_messages(convo)

            temp = temperature if temperature is not None else self.defaults.get("temperature", None)
            out_tokens = max_tokens if max_tokens is not None else self.defaults.get("max_tokens", None)

            # Claude requires max_tokens
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
            success = True
            return text

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



    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[str]:
        call = self.usage.begin_call(run_id=run_id, adapter=self)

        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

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
                    yield txt

            out_tok = int(self.estimate_tokens_for_text("".join(buf), model_hint=self.model_name_for_token_estimation))
            success = True

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
    ) -> Dict[str, Any]:
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

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
            if getattr(resp, "usage", None):
                in_tok = int(resp.usage.input_tokens or in_tok)
                out_tok = int(resp.usage.output_tokens or 0)

            success = True
            return make_tool_result(
                content=extract_anthropic_text(resp),
                tool_calls=extract_anthropic_tool_calls(resp),
                finish_reason=getattr(resp, "stop_reason", None) or "completed",
            )
        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            self.usage.end_call(call, input_tokens=in_tok, output_tokens=out_tok, success=success, error_type=err_type)

    def stream_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[Dict[str, Any]]:
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        in_tok = 0
        out_tok = 0
        success = False
        err_type = None
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
                    if getattr(delta, "type", None) == "text_delta":
                        txt = delta.text or ""
                        if txt:
                            buf.append(txt)
                            yield make_tool_result(content=txt, finish_reason="partial")

            get_final = getattr(stream, "get_final_message", None)
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
                yield result
                return

            if getattr(resp, "usage", None):
                in_tok = int(resp.usage.input_tokens or in_tok)
                out_tok = int(resp.usage.output_tokens or 0)

            success = True
            yield make_tool_result(
                content=extract_anthropic_text(resp) or "".join(buf),
                tool_calls=extract_anthropic_tool_calls(resp),
                finish_reason=getattr(resp, "stop_reason", None) or "completed",
            )
        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            self.usage.end_call(call, input_tokens=in_tok, output_tokens=out_tok, success=success, error_type=err_type)

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ):
        schema = self._model_json_schema(output_model)
        schema_msg = ChatMessage(
            role="user",
            content=(
                "Return ONLY a single JSON object matching this JSON Schema:\n"
                + json.dumps(schema, ensure_ascii=False)
            ),
        )
        extended = list(messages) + [schema_msg]
        raw = self.generate_messages(
            extended,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )
        json_str = self._extract_json_object(raw) or raw.strip()
        return self._validate_with_model(output_model, json_str)
