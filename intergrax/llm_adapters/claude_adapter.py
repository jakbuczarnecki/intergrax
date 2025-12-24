# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from anthropic import Anthropic

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm_adapters.base import ChatMessage, LLMAdapter


class ClaudeChatAdapter(LLMAdapter):
    """
    Claude (Anthropic) adapter based on the official anthropic Python SDK.

    Contract (aligned with OpenAI adapter pattern):
      - __init__(client: Optional[Anthropic] = None, model: Optional[str] = None, **defaults)
      - generate_messages(...) -> str
      - stream_messages(...)   -> Iterable[str]

    Notes:
      - Tools and structured output are not wired here (yet).
    """

    # Conservative context window estimates (keep safe unless you add real token accounting).
    _CLAUDE_CONTEXT_WINDOWS: Dict[str, int] = {
        "claude-3-5-sonnet-latest": 200_000,
        "claude-3-5-haiku-latest": 200_000,
        # Add exact model ids used in your env as needed.
    }

    def __init__(
        self,
        client: Optional[Anthropic] = None,
        model: Optional[str] = None,
        **defaults,
    ):
        super().__init__()
        self.client: Anthropic = client or Anthropic()
        default_model = GLOBAL_SETTINGS.default_claude_model
        self.model: str = model or default_model
        self.defaults = defaults
        self.model_name_for_token_estimation: str = self.model
        self._context_window_tokens: int = self._CLAUDE_CONTEXT_WINDOWS.get(self.model, 32_000)

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        system_text, convo = self._split_system(messages)
        payload_msgs = self._map_messages(convo)

        temp = temperature if temperature is not None else self.defaults.get("temperature", None)
        out_tokens = max_tokens if max_tokens is not None else self.defaults.get("max_tokens", None)

        # Claude requires max_tokens
        if out_tokens is None:
            out_tokens = 1024

        resp = self.client.messages.create(
            model=self.model,
            system=system_text or None,
            messages=payload_msgs,
            max_tokens=int(out_tokens),
            temperature=float(temp) if temp is not None else None,
        )

        # SDK returns content blocks; text is typically in resp.content[*].text
        parts: List[str] = []
        for block in (resp.content or []):            
            if block.type == "text":
                parts.append(block.text or "")
        return "".join(parts)

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterable[str]:
        system_text, convo = self._split_system(messages)
        payload_msgs = self._map_messages(convo)

        temp = temperature if temperature is not None else self.defaults.get("temperature", None)
        out_tokens = max_tokens if max_tokens is not None else self.defaults.get("max_tokens", None)
        if out_tokens is None:
            out_tokens = 1024

        stream = self.client.messages.create(
            model=self.model,
            system=system_text or None,
            messages=payload_msgs,
            max_tokens=int(out_tokens),
            temperature=float(temp) if temp is not None else None,
            stream=True,
        )

        # Anthropic streaming emits SSE events; yield text deltas.
        # Event types vary; the SDK exposes event.type (string).
        for event in stream:
            if event.type == "content_block_delta":
                delta = event.delta
                if getattr(delta, "type", None) == "text_delta":
                    txt = delta.text or ""
                    if txt:
                        yield txt

    # -------------------------
    # Not wired (yet)
    # -------------------------

    def supports_tools(self) -> bool:
        return False

    def generate_with_tools(self, *a, **k):
        raise NotImplementedError("Claude tools are not wired in this adapter.")

    def stream_with_tools(self, *a, **k):
        raise NotImplementedError("Claude tools are not wired in this adapter.")

    def generate_structured(self, *a, **k):
        raise NotImplementedError("Structured output is not implemented for ClaudeChatAdapter.")

    # -------------------------
    # Internals
    # -------------------------

    def _split_system(self, messages: Sequence[ChatMessage]) -> Tuple[str, List[ChatMessage]]:
        sys_parts: List[str] = []
        convo: List[ChatMessage] = []
        for m in messages:
            if m.role == "system":
                if m.content:
                    sys_parts.append(m.content)
            else:
                convo.append(m)
        return ("\n\n".join(sys_parts).strip(), convo)

    def _map_messages(self, msgs: Sequence[ChatMessage]) -> List[Dict[str, str]]:
        """
        Map ChatMessage list to Anthropic Messages API format:
        [{"role": "user"|"assistant", "content": "..."}]

        - System is passed separately via system=...
        - Tool messages are not supported here; treat tool as assistant text if present.
        """
        out: List[Dict[str, str]] = []
        for m in msgs:
            if not m.content:
                continue

            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            else:
                # assistant | tool -> assistant
                out.append({"role": "assistant", "content": m.content})
        return out
