# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import json
from typing import Any, Dict, Iterable, Optional, Sequence

from .base import (
    ChatMessage,
    _extract_json_object,
    _model_json_schema,
    _validate_with_model,
)


class LangChainOllamaAdapter:
    """
    Adapter for Ollama models used via LangChain's ChatModel interface.

    There is no native tools API here, so the agent typically uses a
    "planner" pattern (tool calls are reasoned about in JSON).
    """

    def __init__(self, chat, **defaults):
        self.chat = chat
        self.defaults = defaults

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _to_lc_messages(self, messages: Sequence[ChatMessage]):
        """
        Convert internal ChatMessage list into LangChain message objects.
        """
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        out = []
        for m in messages:
            if m.role == "system":
                out.append(SystemMessage(content=m.content))
            elif m.role == "user":
                out.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                out.append(AIMessage(content=m.content))
            elif m.role == "tool":
                # No native tools: inject tool result as contextual system message
                out.append(SystemMessage(content=f"[TOOL RESULT]\n{m.content}"))
            else:
                out.append(SystemMessage(content=f"[{m.role.upper()}]\n{m.content}"))
        return out

    @staticmethod
    def _with_ollama_options(
        base_kwargs: Dict[str, Any],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Ollama (via langchain_ollama) expects generation parameters inside
        the `options` dictionary.

        Mapping:
            temperature -> options["temperature"]
            max_tokens  -> options["num_predict"]
        """
        kwargs = dict(base_kwargs or {})
        opts = dict(kwargs.get("options") or {})

        if temperature is not None:
            opts["temperature"] = temperature
        if max_tokens is not None:
            opts["num_predict"] = max_tokens

        kwargs["options"] = opts
        return kwargs

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        lc_msgs = self._to_lc_messages(messages)
        kwargs = self._with_ollama_options(self.defaults, temperature=temperature, max_tokens=max_tokens)
        res = self.chat.invoke(lc_msgs, **kwargs)
        return getattr(res, "content", None) or str(res)

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Iterable[str]:
        lc_msgs = self._to_lc_messages(messages)
        kwargs = self._with_ollama_options(self.defaults, temperature=temperature, max_tokens=max_tokens)

        try:
            for chunk in self.chat.stream(lc_msgs, **kwargs):
                c = getattr(chunk, "content", None)
                if c:
                    yield c
        except Exception:
            # Fallback to single-shot call on any streaming error
            yield self.generate_messages(messages, temperature=temperature, max_tokens=max_tokens)

    def supports_tools(self) -> bool:
        """
        There is no native tool-calling support in this adapter.
        The agent should use a planner-style pattern instead.
        """
        return False

    # --- Structured output via prompt + validation ---
    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ):
        """
        Enforce returning a single JSON object conforming to the schema,
        using a strict JSON prompt and post-hoc validation.
        """
        schema = _model_json_schema(output_model)

        from langchain_core.messages import SystemMessage, HumanMessage

        lc_msgs = self._to_lc_messages(messages)

        strict = SystemMessage(
            content=(
                "Return ONLY a single JSON object that strictly conforms to the JSON Schema below. "
                "Do not add any commentary, markdown, or backticks. "
                "If a field is optional and unknown, omit it."
            )
        )
        schema_msg = HumanMessage(content=f"JSON_SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}")
        lc_msgs = [strict, schema_msg] + lc_msgs

        kwargs = self._with_ollama_options(self.defaults, temperature=temperature, max_tokens=max_tokens)
        res = self.chat.invoke(lc_msgs, **kwargs)
        txt = getattr(res, "content", None) or str(res)

        json_str = _extract_json_object(txt) or txt.strip()
        if not json_str:
            raise ValueError("Model did not return JSON content for structured output (Ollama).")

        return _validate_with_model(output_model, json_str)
