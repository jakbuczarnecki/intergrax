# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.


from __future__ import annotations
import json
from typing import Any, Dict, Iterable, Optional, Sequence, List

from .base import (
    ChatMessage,
    _map_messages_to_openai,
    _extract_json_object,
    _model_json_schema,
    _validate_with_model,
)


class OpenAIChatResponsesAdapter:
    """
    OpenAI adapter based on the new Responses API.

    Public interface is compatible with the previous Chat Completions adapter:
    - generate_messages
    - stream_messages
    - generate_with_tools
    - stream_with_tools
    - generate_structured
    """

    def __init__(self, client, model: str, **defaults):
        self.client = client
        self.model = model
        self.defaults = defaults

    # ---------------------------------------------------------------------
    # INTERNAL HELPERS (PRIVATE METHODS)
    # ---------------------------------------------------------------------

    def _messages_to_responses_input(self, mapped_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Chat Completion style messages:
            { "role": "user", "content": "Hello" }

        Into the Responses API "input items":
            { "type": "message", "role": "user", "content": "Hello" }
        """
        items: List[Dict[str, Any]] = []
        for m in mapped_messages:
            items.append(
                {
                    "type": "message",
                    "role": m.get("role", "user"),
                    "content": m.get("content", ""),
                }
            )
        return items

    def _collect_output_text(self, response) -> str:
        """
        Extract the assistant's output text from a Responses API result.

        Prefer response.output_text when available, otherwise aggregate
        all text blocks from response.output[*].content[*] where type == "output_text".
        """
        txt = getattr(response, "output_text", None)
        if txt:
            return txt

        chunks: List[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        chunks.append(getattr(c, "text", "") or "")
        return "".join(chunks)

    # ---------------------------------------------------------------------
    # PUBLIC: Plain chat
    # ---------------------------------------------------------------------

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Single-shot completion (non-streaming) using Responses API.
        """
        mapped = _map_messages_to_openai(messages)
        input_items = self._messages_to_responses_input(mapped)

        payload: Dict[str, Any] = dict(
            model=self.model,
            input=input_items,
            temperature=temperature,
        )

        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        response = self.client.responses.create(**payload, **self.defaults)
        return self._collect_output_text(response)

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Iterable[str]:
        """
        Streaming completion using Responses API.

        Yields incremental text deltas taken from the streaming events.
        """
        mapped = _map_messages_to_openai(messages)
        input_items = self._messages_to_responses_input(mapped)

        payload: Dict[str, Any] = dict(
            model=self.model,
            input=input_items,
            temperature=temperature,
            stream=True,
        )

        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        stream = self.client.responses.create(**payload, **self.defaults)
        for ev in stream:
            # We are interested in "response.output_text.delta" events
            if getattr(ev, "type", None) == "response.output_text.delta":
                delta = getattr(ev, "delta", None)
                if delta:
                    yield delta

    # ---------------------------------------------------------------------
    # PUBLIC: Tools
    # ---------------------------------------------------------------------

    def supports_tools(self) -> bool:
        """
        Signal to higher-level orchestration that this adapter supports tools.
        """
        return True

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema,
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        tool_choice=None,
    ) -> Dict[str, Any]:
        """
        Generate a response with potential function/tool calls.

        Returns:
            {
              "content": str,
              "tool_calls": [...],
              "finish_reason": str
            }
        """
        mapped = _map_messages_to_openai(messages)
        input_items = self._messages_to_responses_input(mapped)

        payload: Dict[str, Any] = dict(
            model=self.model,
            input=input_items,
            temperature=temperature,
            tools=tools_schema,
        )

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        response = self.client.responses.create(**payload, **self.defaults)

        # Assistant text (if present)
        content = self._collect_output_text(response)

        # Extract tool calls in a format compatible with Chat Completions
        native_tool_calls: List[Dict[str, Any]] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                args = getattr(item, "arguments", "{}")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)

                native_tool_calls.append(
                    {
                        "id": getattr(item, "call_id", None),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", None),
                            "arguments": args,
                        },
                    }
                )

        finish_reason = getattr(response, "status", None) or "completed"

        return {
            "content": content or "",
            "tool_calls": native_tool_calls,
            "finish_reason": finish_reason,
        }

    def stream_with_tools(self, *args, **kwargs):
        """
        Currently we do not stream tool calls.
        Fallback to a single non-streaming call for simplicity.
        """
        yield self.generate_with_tools(*args, **kwargs)

    # ---------------------------------------------------------------------
    # PUBLIC: Structured JSON output
    # ---------------------------------------------------------------------

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ):
        """
        Use Responses API + JSON Schema to produce a validated structured output.

        Returns an instance of `output_model` validated from the JSON payload.
        """
        schema = _model_json_schema(output_model)

        sys_extra = {
            "role": "system",
            "content": (
                "Return ONLY a single JSON object that strictly conforms to this JSON Schema. "
                "No prose, no markdown, no backticks. If a field is optional and unknown, omit it.\n"
                f"JSON_SCHEMA: {json.dumps(schema, ensure_ascii=False)}"
            ),
        }

        mapped = _map_messages_to_openai(messages)
        mapped = [sys_extra] + mapped
        input_items = self._messages_to_responses_input(mapped)

        payload: Dict[str, Any] = dict(
            model=self.model,
            input=input_items,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(output_model, "__name__", "OutputModel"),
                    "schema": schema,
                    "strict": True,
                },
            },
        )

        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        response = self.client.responses.create(**payload, **self.defaults)

        txt = self._collect_output_text(response)
        json_str = _extract_json_object(txt) or txt.strip()
        if not json_str:
            raise ValueError("Model did not return valid JSON content for structured output.")

        return _validate_with_model(output_model, json_str)
