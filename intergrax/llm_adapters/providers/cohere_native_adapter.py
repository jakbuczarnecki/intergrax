# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Cohere API v2 native adapter (ClientV2 chat / chat_stream)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import cohere

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.messages import split_system_messages
from intergrax.llm_adapters._shared.tool_results import make_tool_result
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider


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
            success = True
            return text
        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            self.usage.end_call(
                call, input_tokens=in_tok, output_tokens=out_tok, success=success, error_type=err_type
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
                        yield txt
            out_tok = int(self.estimate_tokens_for_text("".join(buf), model_hint=self.model))
            success = True
        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            self.usage.end_call(
                call, input_tokens=in_tok, output_tokens=out_tok, success=success, error_type=err_type
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
    ) -> Dict[str, Any]:
        del tool_choice
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        in_tok = 0
        out_tok = 0
        success = False
        err_type = None
        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model))
            tools = [
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
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": self._map_messages(messages),
                "tools": tools,
            }
            if temperature is not None:
                kwargs["temperature"] = float(temperature)
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            resp = self._execute(lambda: self.client.chat(**kwargs))
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            if resp.message and resp.message.content:
                for block in resp.message.content:
                    if getattr(block, "type", None) == "text":
                        text_parts.append(getattr(block, "text", "") or "")
                    if getattr(block, "type", None) == "tool_call":
                        fn = getattr(block, "function", None)
                        tool_calls.append(
                            {
                                "id": getattr(block, "id", "") or "",
                                "type": "function",
                                "function": {
                                    "name": getattr(fn, "name", "") if fn else "",
                                    "arguments": json.dumps(getattr(fn, "arguments", {}) or {}),
                                },
                            }
                        )
            text = "".join(text_parts)
            out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model))
            success = True
            return make_tool_result(content=text, tool_calls=tool_calls, finish_reason="completed")
        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            self.usage.end_call(
                call, input_tokens=in_tok, output_tokens=out_tok, success=success, error_type=err_type
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
    ) -> Iterable[Dict[str, Any]]:
        del tool_choice
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        in_tok = 0
        out_tok = 0
        success = False
        err_type = None
        buf: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

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
                        yield make_tool_result(content=txt, finish_reason="partial")
                if getattr(event, "type", None) == "message-end":
                    msg = getattr(event, "message", None)
                    if msg and getattr(msg, "content", None):
                        for block in msg.content:
                            if getattr(block, "type", None) == "tool_call":
                                fn = getattr(block, "function", None)
                                tool_calls.append(
                                    {
                                        "id": getattr(block, "id", "") or "",
                                        "type": "function",
                                        "function": {
                                            "name": getattr(fn, "name", "") if fn else "",
                                            "arguments": json.dumps(
                                                getattr(fn, "arguments", {}) or {}
                                            ),
                                        },
                                    }
                                )

            if not buf and not tool_calls:
                result = self.generate_with_tools(
                    messages,
                    tools_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    run_id=run_id,
                )
                yield result
                success = True
                return

            out_tok = int(self.estimate_tokens_for_text("".join(buf), model_hint=self.model))
            success = True
            yield make_tool_result(
                content="".join(buf),
                tool_calls=tool_calls,
                finish_reason="completed",
            )
        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            self.usage.end_call(
                call, input_tokens=in_tok, output_tokens=out_tok, success=success, error_type=err_type
            )
