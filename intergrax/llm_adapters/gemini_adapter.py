# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from google import genai
from google.genai import types

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm_adapters.base import ChatMessage, LLMAdapter


class GeminiChatAdapter(LLMAdapter):
    """
    Gemini adapter based on the official Google Gen AI SDK (google-genai).

    - Uses genai.Client (official client type).
    - Supports:
        - generate_messages
        - stream_messages
    - Tools + structured output are intentionally not wired here (yet).
    """

    # Conservative context window estimates (input + output).
    # Keep this small/safe unless you add a real token counter for Gemini.
    _GEMINI_CONTEXT_WINDOWS: Dict[str, int] = {
        "gemini-2.5-pro": 1_000_000,
        "gemini-2.5-flash": 1_000_000,
        "gemini-2.0-pro": 1_000_000,
        "gemini-2.0-flash": 1_000_000,
    }

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model: Optional[str] = None,
        **defaults,
    ):
        super().__init__()

        # If you want framework-wide defaults later, you can route this via GLOBAL_SETTINGS,
        # but this adapter stays self-contained to avoid inventing missing settings fields.
        default_model = GLOBAL_SETTINGS.default_gemini_model

        self.client: genai.Client = client or genai.Client()
        self.model: str = model or default_model
        self.model_name_for_token_estimation: str = self.model
        self.defaults = defaults

        self._context_window_tokens: int = self._estimate_gemini_context_window(self.model)

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
        call = self.begin_call(run_id=run_id)

        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            system_text, convo = self._split_system(messages)

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
                response = chat_session.send_message(prompt)
                text = response.text or ""
            else:
                # Fallback: use generate_content with full contents list (handles odd turn ordering).
                contents = self._map_contents(convo)
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                text = response.text or ""

            out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model_name_for_token_estimation))
            success = True
            return text

        except Exception as e:
            err_type = type(e).__name__
            raise

        finally:
            self.end_call(
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
        call = self.begin_call(run_id=run_id)

        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

        buf: List[str] = []

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            system_text, convo = self._split_system(messages)

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

                out_tok = int(self.estimate_tokens_for_text("".join(buf), model_hint=self.model_name_for_token_estimation))
                success = True
                return

            # Keep behavior explicit and predictable (as you had).
            raise NotImplementedError(
                "GeminiChatAdapter.stream_messages requires the last message to be role='user'."
            )

        except Exception as e:
            err_type = type(e).__name__
            raise

        finally:
            self.end_call(
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

    def _split_system(self, messages: Sequence[ChatMessage]) -> Tuple[str, List[ChatMessage]]:
        system_parts: List[str] = []
        convo: List[ChatMessage] = []

        for m in messages:
            if m.role == "system":
                if m.content:
                    system_parts.append(m.content)
                continue
            convo.append(m)

        return ("\n\n".join(system_parts).strip(), convo)

    def _build_generation_config(
        self,
        *,
        system_text: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
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
        """
        ChatMessage -> google.genai.types.*Content
        """
        part = types.Part(text=m.content)

        if m.role == "user":
            return types.UserContent(parts=[part])

        # Treat assistant/tool as model content (tools are not wired; keep history coherent).
        return types.ModelContent(parts=[part])
