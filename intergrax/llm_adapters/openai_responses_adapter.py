# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.


from __future__ import annotations
import json
from typing import Any, Dict, Iterable, Optional, Sequence, List
from mistralai import Union
from openai import Client
from openai.types.responses import Response, ResponseUsage

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm_adapters.llm_adapter import (    
    ChatMessage,
    LLMAdapter,
)


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

    def __init__(self, client: Optional[Client] = None, model: Optional[str] = None, **defaults):
        super().__init__()
        self.client = client or Client()
        self.model = model or GLOBAL_SETTINGS.default_openai_model
        self.model_name_for_token_estimation = self.model
        self.defaults = defaults
        self._context_window_tokens: int = self._estimate_openai_context_window(self.model)


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
    ) -> str:
        """
        Single-shot completion (non-streaming) using Responses API.
        """        
        call = self.usage.begin_call(run_id=run_id)

        in_tok = 0
        out_tok = 0
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

            response: Response = self.client.responses.create(**payload, **self.defaults)

            usage = response.usage
            if usage is not None:
                in_tok = int(usage.input_tokens or 0)
                out_tok = int(usage.output_tokens or 0)      

            output_text = self._collect_output_text(response)
            
            success = True

            return output_text

        except Exception as e:
            err_type = type(e).__name__
            raise
        finally:
            self.usage.end_call(call, input_tokens=in_tok, output_tokens=out_tok, success=success, error_type=err_type)


    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[str]:
        """
        Streaming completion using Responses API.

        Yields incremental text deltas taken from the streaming events.
        """
        call = self.usage.begin_call(run_id=run_id)

        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

        # Fallback input tokens (streaming may not provide usage reliably)
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

            with self.client.responses.stream(**payload, **self.defaults) as stream:
                for ev in stream:
                    # We are interested in "response.output_text.delta" events
                    if ev.type == "response.output_text.delta":                                                                        
                        delta = ev.delta
                        if delta:
                            # accumulate for token estimation on finish
                            buf.append(delta)
                            yield delta

            # If we reached here, the stream finished naturally
            # Estimate output tokens from the assembled text
            try:
                full_text = "".join(buf)
                out_tok = int(self.estimate_tokens_for_text(full_text))
            except Exception:
                out_tok = 0

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
        tools_schema: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
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
        mapped = self._map_messages_to_openai(messages)
        input_items = self._messages_to_responses_input(mapped)

        payload: Dict[str, Any] = dict(
            model=self.model,
            input=input_items,
            # temperature=temperature, - not applicable in openai responses
            tools=tools_schema,
        )

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        response: Response = self.client.responses.create(**payload, **self.defaults)

        # Assistant text (if present)
        content = self._collect_output_text(response)

        # Extract tool calls in a format compatible with Chat Completions
        native_tool_calls: List[Dict[str, Any]] = []
        for item in response.output or []:
            if item.type == "function_call":
                args = item.arguments
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)

                native_tool_calls.append(
                    {
                        "id": item.call_id,
                        "type": "function",
                        "function": {
                            "name": item.name,
                            "arguments": args,
                        },
                    }
                )

        finish_reason = response.status or "completed"

        return {
            "content": content or "",
            "tool_calls": native_tool_calls,
            "finish_reason": finish_reason,
        }

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
        """
        Currently we do not stream tool calls.
        Fallback to a single non-streaming call for simplicity.
        """
        yield self.generate_with_tools(
            messages,
            tools_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            run_id=run_id,
        )

    # ---------------------------------------------------------------------
    # PUBLIC: Structured JSON output
    # ---------------------------------------------------------------------

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
        """
        Generate a response with potential function/tool calls.

        Returns:
            {
            "content": str,
            "tool_calls": [...],
            "finish_reason": str
            }
        """
        call = self.usage.begin_call(run_id=run_id)

        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

        try:
            mapped = self._map_messages_to_openai(messages)
            input_items = self._messages_to_responses_input(mapped)

            payload: Dict[str, Any] = dict(
                model=self.model,
                input=input_items,
                tools=tools_schema,
            )

            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

            if max_tokens is not None:
                payload["max_output_tokens"] = max_tokens

            response: Response = self.client.responses.create(**payload, **self.defaults)

            usage = response.usage
            if usage is not None:
                in_tok = int(usage.input_tokens or 0)
                out_tok = int(usage.output_tokens or 0)

            # Assistant text (if present)
            content = self._collect_output_text(response)

            # Extract tool calls in a format compatible with Chat Completions
            native_tool_calls: List[Dict[str, Any]] = []
            for item in response.output or []:
                if item.type == "function_call":
                    args = item.arguments
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)

                    native_tool_calls.append(
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": item.name,
                                "arguments": args,
                            },
                        }
                    )

            finish_reason = response.status or "completed"

            success = True
            return {
                "content": content or "",
                "tool_calls": native_tool_calls,
                "finish_reason": finish_reason,
            }

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
                if getattr(m, "tool_call_id", None) is not None:
                    d["tool_call_id"] = m.tool_call_id
                if getattr(m, "name", None) is not None:
                    d["name"] = m.name

            if getattr(m, "tool_calls", None):
                d["tool_calls"] = m.tool_calls

            out.append(d)
        return out
