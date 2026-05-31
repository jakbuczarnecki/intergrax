# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from google import genai
from google.genai import types

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.messages import split_system_messages
from intergrax.llm_adapters._shared.tool_results import make_tool_result
from intergrax.llm_adapters._shared.tool_schema import openai_tools_to_gemini
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider


class GeminiChatAdapter(LLMAdapter):
    """
    Gemini adapter based on the official Google Gen AI SDK (google-genai).

    - Uses genai.Client (official client type).
    - Supports:
        - generate_messages
        - stream_messages
    - Native tools via Gemini function calling.
    """

    # Conservative context window estimates (input + output).
    # Keep this small/safe unless you add a real token counter for Gemini.
    _GEMINI_CONTEXT_WINDOWS: Dict[str, int] = {
        "gemini-2.5-pro": 1_000_000,
        "gemini-2.5-flash": 1_000_000,
        "gemini-2.0-pro": 1_000_000,
        "gemini-2.0-flash": 1_000_000,
    }

    DEFAULT_MODEL = "gemini-2.5-flash"

    ENV_MODEL = "INTERGRAX_DEFAULT_GEMINI_MODEL"
    ENV_API_KEY = "GOOGLE_API_KEY"

    def __init__(
        self,
        client: Optional[genai.Client] = None,
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
                    "GOOGLE_API_KEY not found in environment variables."
                )

            client = genai.Client(api_key=api_key)

        self.client: genai.Client = client
        self.model: str = resolved_model

        self.model_name_for_token_estimation = self.model
        self.defaults = defaults

        self._context_window_tokens: int = self._estimate_gemini_context_window(self.model)

        self.provider = LLMProvider.GEMINI

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
    ) -> str:
        call = self.usage.begin_call(run_id=run_id)

        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            system_text, convo = split_system_messages(messages)

            config = self._build_generation_config(
                system_text=system_text,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Typical case: last message is user -> create chat with history and send last user message.
            if convo and convo[-1].role == "user":
                history = self._map_history(convo[:-1])
                prompt = convo[-1].content or ""

                chat_session = self.client.chats.create(
                    model=self.model,
                    history=history,
                    config=config,
                )
                response = self._execute(lambda: chat_session.send_message(prompt))
                text = response.text or ""
            else:
                # Fallback: use generate_content with full contents list (handles odd turn ordering).
                contents = self._map_contents(convo)
                response = self._execute(
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config,
                    )
                )
                text = response.text or ""

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
        call = self.usage.begin_call(run_id=run_id)

        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

        buf: List[str] = []

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            system_text, convo = split_system_messages(messages)

            config = self._build_generation_config(
                system_text=system_text,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if convo and convo[-1].role == "user":
                history = self._map_history(convo[:-1])
                prompt = convo[-1].content or ""

                chat_session = self.client.chats.create(
                    model=self.model,
                    history=history,
                    config=config,
                )

                for chunk in chat_session.send_message_stream(prompt):
                    txt = chunk.text
                    if txt:
                        buf.append(txt)
                        yield txt

                out_tok = int(
                    self.estimate_tokens_for_text("".join(buf), model_hint=self.model_name_for_token_estimation)
                )
                success = True
                return

            # Production-grade fallback: stream via generate_content_stream using full contents list.
            # This mirrors generate_messages() fallback behavior and avoids hard failure on "odd turn ordering".
            contents = self._map_contents(convo)

            # google-genai supports streaming on models via generate_content_stream (SDK-dependent).
            stream_fn = getattr(self.client.models, "generate_content_stream", None)
            if stream_fn is None:
                raise RuntimeError(
                    "GeminiChatAdapter.stream_messages fallback requires google-genai "
                    "Client.models.generate_content_stream, but it is not available in this SDK version."
                )

            for chunk in stream_fn(
                model=self.model,
                contents=contents,
                config=config,
            ):
                txt = getattr(chunk, "text", None)
                if txt:
                    buf.append(txt)
                    yield txt

            out_tok = int(
                self.estimate_tokens_for_text("".join(buf), model_hint=self.model_name_for_token_estimation)
            )
            success = True
            return

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

    def _estimate_gemini_context_window(self, model: str) -> int:
        # Safe fallback (small) if unknown.
        return self._GEMINI_CONTEXT_WINDOWS.get(model, 32_000)

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
        call = self.usage.begin_call(run_id=run_id)
        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))
            system_text, convo = split_system_messages(messages)
            config = self._build_generation_config(
                system_text=system_text,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=openai_tools_to_gemini(tools_schema),
            )
            contents = self._map_contents(convo)
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            text, tool_calls = self._parse_gemini_response(response)
            out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model_name_for_token_estimation))
            success = True
            return make_tool_result(content=text, tool_calls=tool_calls, finish_reason="completed")
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
        call = self.usage.begin_call(run_id=run_id)
        in_tok = 0
        out_tok = 0
        success = False
        err_type = None
        buf: List[str] = []

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))
            system_text, convo = split_system_messages(messages)
            config = self._build_generation_config(
                system_text=system_text,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=openai_tools_to_gemini(tools_schema),
            )
            contents = self._map_contents(convo)
            stream_fn = getattr(self.client.models, "generate_content_stream", None)
            if stream_fn is None:
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

            tool_calls_acc: List[Dict[str, Any]] = []
            for chunk in stream_fn(model=self.model, contents=contents, config=config):
                txt = getattr(chunk, "text", None)
                if txt:
                    buf.append(txt)
                    yield make_tool_result(content=txt, finish_reason="partial")
                _, chunk_tools = self._parse_gemini_response(chunk)
                if chunk_tools:
                    tool_calls_acc = chunk_tools

            success = True
            yield make_tool_result(
                content="".join(buf),
                tool_calls=tool_calls_acc,
                finish_reason="completed",
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
        raw = self.generate_messages(
            list(messages) + [schema_msg],
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )
        json_str = self._extract_json_object(raw) or raw.strip()
        return self._validate_with_model(output_model, json_str)

    def _parse_gemini_response(self, response: Any) -> tuple[str, List[Dict[str, Any]]]:
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "text", None):
                    text_parts.append(part.text or "")
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    name = getattr(fc, "name", "") or ""
                    args = getattr(fc, "args", None) or {}
                    tool_calls.append(
                        {
                            "id": name or "gemini_call",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    )
        return "".join(text_parts), tool_calls

    def _build_generation_config(
        self,
        *,
        system_text: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        tools: Optional[List[Any]] = None,
    ) -> types.GenerateContentConfig:
        # Merge defaults (adapter-level) with call-level overrides.
        # Keep it explicit: only pass supported fields.
        temp = temperature if temperature is not None else self.defaults.get("temperature", None)
        out_tokens = max_tokens if max_tokens is not None else self.defaults.get("max_tokens", None)

        kwargs = {}

        if system_text:
            kwargs["system_instruction"] = system_text
        if temp is not None:
            kwargs["temperature"] = float(temp)
        if out_tokens is not None:
            kwargs["max_output_tokens"] = int(out_tokens)
        if tools:
            kwargs["tools"] = tools

        return types.GenerateContentConfig(**kwargs)

    def _map_history(self, msgs: Sequence[ChatMessage]) -> List[types.Content]:
        """
        Map prior messages (excluding the last user prompt) into chat history.
        Uses official typed Content classes.
        """
        out: List[types.Content] = []
        for m in msgs:
            if not m.content:
                continue
            out.append(self._to_content(m))
        return out

    def _map_contents(self, msgs: Sequence[ChatMessage]) -> List[types.Content]:
        """
        Map full conversation into contents list (for generate_content fallback).
        """
        out: List[types.Content] = []
        for m in msgs:
            if not m.content:
                continue
            out.append(self._to_content(m))
        return out

    def _to_content(self, m: ChatMessage) -> types.Content:
        """ChatMessage -> google.genai Content (user/model roles, tool results)."""
        if m.role == "tool":
            fr = types.Part.from_function_response(
                name=m.name or "tool",
                response={"output": m.content or ""},
            )
            return types.UserContent(parts=[fr])

        if m.role == "assistant" and m.tool_calls:
            parts: List[Any] = []
            if m.content:
                parts.append(types.Part(text=m.content))
            for tc in m.tool_calls:
                fn = tc.get("function") or {}
                args_raw = fn.get("arguments") or "{}"
                try:
                    args_obj = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args_obj = {}
                parts.append(
                    types.Part.from_function_call(
                        name=fn.get("name") or "",
                        args=args_obj if isinstance(args_obj, dict) else {},
                    )
                )
            return types.ModelContent(parts=parts)

        part = types.Part(text=m.content or "")
        if m.role == "user":
            return types.UserContent(parts=[part])
        return types.ModelContent(parts=[part])
