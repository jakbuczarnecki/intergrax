# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Iterable, Optional, Sequence

from .base import ChatMessage


class GeminiChatAdapter:
    """
    Minimal Gemini adapter.

    NOTE:
    - This implementation intentionally does not wire tools.
    - It focuses on simple chat usage.
    """

    def __init__(self, model, **defaults):
        self.model = model
        self.defaults = defaults

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
        temperature: float = 0.2,
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
        temperature: float = 0.2,
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
