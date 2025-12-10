# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Dict, Iterable, Optional, Sequence

from intergrax.llm_adapters.base import BaseLLMAdapter, ChatMessage


class GeminiChatAdapter(BaseLLMAdapter):
    """
    Minimal Gemini adapter.

    NOTE:
    - This implementation intentionally does not wire tools.
    - It focuses on simple chat usage.
    """

    # Conservative context window estimates for common Gemini models.
    _GEMINI_CONTEXT_WINDOWS: Dict[str, int] = {
        # Gemini 1.5 family (1M tokens)
        "gemini-1.5-pro": 1_000_000,
        "gemini-1.5-flash": 1_000_000,

        # Gemini 2.0 Flash family (1,048,576 tokens = 2^20)
        "gemini-2.0-flash": 1_048_576,
        "gemini-2.0-flash-lite": 1_048_576,
        "gemini-2.0-flash-thinking": 1_048_576,
    }


    def _estimate_gemini_context_window(self, model: str) -> int:
        """
        Best-effort context window estimation for Gemini models.
        Computed once at adapter construction time.
        """
        name = (model or "").strip()
        base = name.split(":", 1)[0]

        if base in self._GEMINI_CONTEXT_WINDOWS:
            return self._GEMINI_CONTEXT_WINDOWS[base]

        return 1_000_000

    def __init__(self, model, **defaults):
        super().__init__()
        self.model = model
        self.defaults = defaults
        self._context_window_tokens: int = self._estimate_gemini_context_window(str(model))


    @property
    def context_window_tokens(self) -> int:
        """
        Cached maximum context window (input + output tokens) for the
        configured Gemini model. Computed once in __init__.
        """
        return self._context_window_tokens
    

    def _split_system(self, messages: Sequence[ChatMessage]):
        """
        Separate system messages from the rest of the conversation, so that
        we can prepend them manually if needed.
        """
        sys_txt = "\n".join(m.content for m in messages if m.role == "system").strip() or None
        convo = [m for m in messages if m.role != "system"]
        return sys_txt, convo

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        sys_txt, convo = self._split_system(messages)
        history = [{"role": m.role, "parts": [m.content]} for m in convo[:-1]]
        chat = self.model.start_chat(history=history)
        user_last = convo[-1].content if convo else ""

        kwargs = {"temperature": temperature, **self.defaults}
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens

        if sys_txt:
            user_last = f"[SYSTEM]\n{sys_txt}\n\n[USER]\n{user_last}"

        res = chat.send_message(user_last, **kwargs)
        return getattr(res, "text", "") or ""

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterable[str]:
        # Simple fallback streaming: single-shot only
        yield self.generate_messages(messages, temperature=temperature, max_tokens=max_tokens)

    # Tools not implemented in this skeleton.
    def supports_tools(self) -> bool:
        return False

    def generate_with_tools(self, *a, **k):
        raise NotImplementedError("Gemini tools are not wired in this adapter.")

    def stream_with_tools(self, *a, **k):
        raise NotImplementedError("Gemini tools are not wired in this adapter.")
    

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Structured output is not implemented for this adapter.
        """
        raise NotImplementedError("Structured output is not implemented for GeminiChatAdapter.")

