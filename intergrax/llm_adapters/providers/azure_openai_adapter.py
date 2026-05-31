# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from openai import AzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.messages import map_chat_completion_messages, split_system_messages
from intergrax.llm_adapters._shared.tool_results import make_tool_result
from intergrax.llm_adapters._shared.tool_schema import extract_openai_tool_calls
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider


class AzureOpenAIChatAdapter(LLMAdapter):
    """
    Azure OpenAI adapter based on the official OpenAI Python SDK (AzureOpenAI).

    Contract (aligned with other adapters):
      - __init__(client: Optional[AzureOpenAI] = None, deployment: Optional[str] = None, **defaults)
      - generate_messages(...) -> str
      - stream_messages(...)   -> Iterable[str]

    Notes:
      - On Azure, the 'model' field in chat completions must be set to the DEPLOYMENT NAME,
        not the underlying model id. :contentReference[oaicite:2]{index=2}
      - Native tools via Chat Completions API.
    """

    # Conservative context window estimates (input + output).
    # Keep safe unless you add a token counter per deployment.
    _AZURE_CONTEXT_WINDOWS: Dict[str, int] = {
        # deployments are tenant-specific; keep fallback conservative
    }

    def __init__(
        self,
        client: Optional[AzureOpenAI] = None,
        deployment: Optional[str] = None,
        **defaults,
    ):
        super().__init__()
        self._apply_defaults_call_config(defaults)

        # Framework-wide defaults should be routed via GLOBAL_SETTINGS.
        # Keep these names consistent with your settings pattern.        

        endpoint = (os.getenv("INTERGRAX_DEFAULT_AZURE_OPENAI_ENDPOINT", "") or "").strip()
        api_version = (os.getenv("INTERGRAX_DEFAULT_AZURE_OPENAI_API_VERSION", "") or "").strip()
        default_deployment = (os.getenv("INTERGRAX_DEFAULT_AZURE_OPENAI_DEPLOYMENT", "") or "").strip()

        self.deployment: str = (deployment or default_deployment).strip()
        if not endpoint:
            raise RuntimeError(
                "INTERGRAX_DEFAULT_AZURE_OPENAI_ENDPOINT must be configured for Azure OpenAI adapter."
            )
        if not api_version:
            raise RuntimeError(
                "INTERGRAX_DEFAULT_AZURE_OPENAI_API_VERSION must be configured for Azure OpenAI adapter."
            )
        if not self.deployment:
            raise RuntimeError(
                "INTERGRAX_DEFAULT_AZURE_OPENAI_DEPLOYMENT or deployment= must be set for Azure OpenAI adapter."
            )

        self.client: AzureOpenAI = client or AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.model_name_for_token_estimation: str = self.deployment

        self.defaults = defaults
        self._context_window_tokens: int = self._estimate_context_window(self.deployment)

        self.provider = LLMProvider.AZURE_OPENAI
        self.model = self.deployment

    @property
    def context_window_tokens(self) -> int:
        """
        Cached maximum context window (input + output tokens) for the configured deployment.
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
            system_text, convo = split_system_messages(messages)

            payload = self._build_chat_params(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            res: ChatCompletion = self._execute(
                lambda: self.client.chat.completions.create(**payload)
            )

            usage = res.usage
            if usage is not None:
                in_tok = int(usage.prompt_tokens or 0)
                out_tok = int(usage.completion_tokens or 0)

            if not res.choices:
                success = True
                return ""

            msg = res.choices[0].message
            text = msg.content or ""
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
            # Streaming usually does not provide usage in chunks -> estimate input
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            system_text, convo = split_system_messages(messages)

            payload = self._build_chat_params(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            stream = self._execute(lambda: self.client.chat.completions.create(**payload))

            for chunk in stream:
                c: ChatCompletionChunk = chunk
                if not c.choices:
                    continue

                delta = c.choices[0].delta
                if delta is None:
                    continue

                if delta.content:
                    buf.append(delta.content)
                    yield delta.content

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



    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _estimate_context_window(self, deployment: str) -> int:
        # Deployments are user-defined names; keep fallback conservative.
        return self._AZURE_CONTEXT_WINDOWS.get(deployment, 32_000)

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
            system_text, convo = split_system_messages(messages)
            payload = self._build_chat_params(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                tools=tools_schema,
                tool_choice=tool_choice,
            )
            res: ChatCompletion = self._execute(
                lambda: self.client.chat.completions.create(**payload)
            )
            if res.usage:
                in_tok = int(res.usage.prompt_tokens or 0)
                out_tok = int(res.usage.completion_tokens or 0)
            if not res.choices:
                success = True
                return make_tool_result()
            msg = res.choices[0].message
            success = True
            return make_tool_result(
                content=msg.content or "",
                tool_calls=extract_openai_tool_calls(msg),
                finish_reason=res.choices[0].finish_reason or "completed",
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
        call = self.usage.begin_call(run_id=run_id)
        in_tok = 0
        out_tok = 0
        success = False
        err_type = None
        buf: List[str] = []
        tool_calls_acc: List[Dict[str, Any]] = []

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))
            system_text, convo = split_system_messages(messages)
            payload = self._build_chat_params(
                system_text=system_text,
                convo=convo,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                tools=tools_schema,
                tool_choice=tool_choice,
            )
            stream = self._execute(lambda: self.client.chat.completions.create(**payload))
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta and delta.content:
                    buf.append(delta.content)
                    yield make_tool_result(content=delta.content, finish_reason="partial")
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index or 0
                        while len(tool_calls_acc) <= idx:
                            tool_calls_acc.append(
                                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                            )
                        acc = tool_calls_acc[idx]
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                acc["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                acc["function"]["arguments"] += tc.function.arguments

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
        system_text, convo = split_system_messages(messages)
        schema = self._model_json_schema(output_model)
        payload = self._build_chat_params(
            system_text=system_text,
            convo=convo,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(output_model, "__name__", "structured_output"),
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        res = self._execute(lambda: self.client.chat.completions.create(**payload))
        raw = (res.choices[0].message.content or "") if res.choices else ""
        json_str = self._extract_json_object(raw) or raw.strip()
        return self._validate_with_model(output_model, json_str)

    def _build_chat_params(
        self,
        *,
        system_text: str,
        convo: Sequence[ChatMessage],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Build a minimal, explicit Azure Chat Completions payload.

        Azure expects:
          - model: deployment name (not model id) :contentReference[oaicite:3]{index=3}
          - messages: list of {role, content}
        """
        temp = temperature if temperature is not None else self.defaults.get("temperature", None)
        out_tokens = max_tokens if max_tokens is not None else self.defaults.get("max_tokens", None)

        mapped = self._map_messages(system_text=system_text, convo=convo)

        payload: dict = {
            "model": self.deployment,
            "messages": mapped,
            "stream": stream,
        }

        if temp is not None:
            payload["temperature"] = float(temp)
        if out_tokens is not None:
            payload["max_tokens"] = int(out_tokens)
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format

        return payload

    def _map_messages(self, *, system_text: str, convo: Sequence[ChatMessage]) -> List[dict]:
        return map_chat_completion_messages(system_text=system_text, convo=convo)
