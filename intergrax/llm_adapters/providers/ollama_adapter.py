# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from langchain_ollama import ChatOllama

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
    final_stream_event,
    partial_stream_event,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.provider_extensions import LLMProviderExtensions
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.contracts.tool_call import (
    LLMToolCall,
    tool_calls_from_langchain_message,
)
from intergrax.llm_adapters.providers._ollama_schema import prepare_ollama_generation_schema
from intergrax.llm_adapters.providers.ollama_capabilities import (
    OllamaModelCapabilities,
    OllamaModelCapabilityResolver,
)
from intergrax.llm_adapters.registry.context_window import init_adapter_context_window_tokens
from intergrax.utils import attribute_access


class LangChainOllamaAdapter(LLMAdapter):
    """
    Adapter for Ollama models used via LangChain's ChatOllama interface.

    Native non-streaming tool calling is supported through ``ChatOllama.bind_tools()``.
    Structured output remains supported separately via ``with_structured_output()``.
    Tool streaming is not implemented. Model-level tool capability may vary by model.
    """


    # Conservative context window estimates for common Ollama models.
    # Extend this mapping with the models you actually use.
    _OLLAMA_CONTEXT_WINDOWS: Dict[str, int] = {
        # LLaMA 3 / LLaMA 3.1
        "llama3:8b": 8_192,
        "llama3:70b": 8_192,
        "llama3.1:8b": 16_384,
        "llama3.1:70b": 16_384,

        # Qwen 2 / Qwen 2.5
        "qwen2:7b": 32_768,
        "qwen2:72b": 32_768,
        "qwen2.5:1.5b": 32_768,
        "qwen2.5:7b": 32_768,
        "qwen2.5:14b": 32_768,

        # Phi-3 / Phi-3.5
        "phi3:mini": 4_096,
        "phi3:medium": 8_192,
        "phi3.5:mini": 128_000,
        "phi3.5:moe": 128_000,

        # Mistral / Mixtral
        "mistral:7b": 8_192,
        "mixtral:8x7b": 32_768,

        # StarCoder2
        "starcoder2:7b": 8_192,
        "starcoder2:15b": 8_192,

        # Gemma 2
        "gemma2:2b": 8_192,
        "gemma2:9b": 8_192,
        "gemma2:27b": 8_192,

        # openAI
        "gpt-oss:20b": 128_000,
        "gpt-oss:120b": 128_000,
    }

    DEFAULT_MODEL = "llama3.1:latest"


    def _estimate_ollama_context_window_from_model(self, model: str) -> int:
        """
        Best-effort context window estimation for an Ollama model based on
        its name. This is computed once in the adapter __init__.
        """
        name = (model or "").strip()
        base = name.split(":", 1)[0]  # remove possible tags like ":latest"

        if base in self._OLLAMA_CONTEXT_WINDOWS:
            return self._OLLAMA_CONTEXT_WINDOWS[base]

        # Conservative fallback if the model is unknown.
        return 8_192

    ENV_MODEL = "INTERGRAX_DEFAULT_OLLAMA_MODEL"

    def __init__(
        self,
        chat: Optional[ChatOllama] = None,
        model: Optional[str] = None,
        context_window_tokens: Optional[int] = None,
        *,
        capability_resolver: OllamaModelCapabilityResolver | None = None,
        base_url: Optional[str] = None,
        **defaults,
    ):
        super().__init__()
        self._apply_defaults_call_config(defaults)

        resolved_model = model or os.getenv(self.ENV_MODEL) or self.DEFAULT_MODEL
        chat_kwargs: Dict[str, Any] = {"model": resolved_model}
        if base_url is not None:
            chat_kwargs["base_url"] = base_url
        self.chat = chat or ChatOllama(**chat_kwargs)
        self.defaults = defaults
        self._base_url = base_url
        self._capability_resolver = capability_resolver or OllamaModelCapabilityResolver(
            base_url=base_url,
        )
        self._model_capabilities: OllamaModelCapabilities | None = None

        if context_window_tokens is not None and int(context_window_tokens) > 0:
            defaults["context_window_tokens"] = int(context_window_tokens)

        self._context_window_tokens = int(
            init_adapter_context_window_tokens(
                provider=LLMProvider.OLLAMA,
                model=resolved_model,
                constructor_kwargs=defaults,
                legacy_windows=self._OLLAMA_CONTEXT_WINDOWS,
            )
        )

        self.provider = LLMProvider.OLLAMA
        self.model = self.chat.model


    @property
    def context_window_tokens(self) -> int:
        """
        Cached maximum context window (input + output tokens) for the
        configured Ollama model. Computed once in __init__.
        """
        return self._context_window_tokens
    

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    @staticmethod
    def _internal_tool_call_to_langchain(tool_call: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(tool_call, dict):
            raise ValueError("tool call must be a dictionary")

        if "name" in tool_call and "args" in tool_call:
            name = tool_call.get("name")
            if not name or not str(name).strip():
                raise ValueError("tool call requires name")
            args = tool_call.get("args")
            if not isinstance(args, dict):
                raise ValueError("tool call args must be a dictionary")
            return {
                "name": str(name),
                "args": args,
                "id": str(tool_call.get("id") or ""),
                "type": "tool_call",
            }

        fn = tool_call.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if not name or not str(name).strip():
                raise ValueError("tool call requires name")
            raw_args = fn.get("arguments")
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args or "{}")
                except json.JSONDecodeError:
                    raise ValueError("tool call arguments must be valid JSON") from None
                if not isinstance(parsed, dict):
                    raise ValueError("tool call arguments must decode to an object")
                args = parsed
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            return {
                "name": str(name),
                "args": args,
                "id": str(tool_call.get("id") or ""),
                "type": "tool_call",
            }

        raise ValueError("tool call requires name")

    @staticmethod
    def _coerce_ai_message_content(message: Any) -> str:
        content = attribute_access.optional(message, "content", "")
        if isinstance(content, str):
            return content
        if not content:
            return ""
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    @staticmethod
    def _validate_ollama_tool_choice(
        tool_choice: Optional[Union[str, Dict[str, Any]]],
    ) -> None:
        if tool_choice is None:
            return
        if isinstance(tool_choice, str) and tool_choice in {"auto", "required"}:
            return
        raise ValueError(
            "Ollama native tool calling supports only tool_choice=None, 'auto', or 'required'"
        )

    @staticmethod
    def _estimate_tool_output_text(content: str, tool_calls: tuple[LLMToolCall, ...]) -> str:
        parts = [content]
        for tool_call in tool_calls:
            parts.append(tool_call.name)
            parts.append(tool_call.arguments_json)
        return "".join(parts)

    def _to_lc_messages(self, messages: Sequence[ChatMessage]):
        """
        Convert internal ChatMessage list into LangChain message objects.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        out = []
        for m in messages:
            if m.role == "system":
                out.append(SystemMessage(content=m.content))
            elif m.role == "user":
                out.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                if m.tool_calls:
                    lc_tool_calls = [
                        self._internal_tool_call_to_langchain(tc) for tc in m.tool_calls
                    ]
                    out.append(AIMessage(content=m.content, tool_calls=lc_tool_calls))
                else:
                    out.append(AIMessage(content=m.content))
            elif m.role == "tool":
                if not m.tool_call_id or not str(m.tool_call_id).strip():
                    raise ValueError("tool message requires tool_call_id")
                out.append(
                    ToolMessage(content=m.content, tool_call_id=str(m.tool_call_id))
                )
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
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        response: LLMAdapterResponse | None = None
        success = False
        err_type = None
        in_tok = 0
        out_tok = 0

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))
            lc_msgs = self._to_lc_messages(messages)
            kwargs = self._with_ollama_options(self.defaults, temperature=temperature, max_tokens=max_tokens)
            res = self.chat.invoke(lc_msgs, **kwargs)
            text = res.content or str(res)
            out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model_name_for_token_estimation))
            response = build_adapter_response(
                content=text,
                usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                model=self.model,
                provider=self._provider_slug(),
                provider_extensions=LLMProviderExtensions(usage_source="estimate"),
            )
            success = True
            return response

        except Exception as e:
            err_type = type(e).__name__
            raise

        finally:
            usage = response.usage if (success and response and response.usage) else LLMTokenUsage()
            self.usage.end_call(
                call,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
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
    ) -> Iterable[LLMStreamEvent]:
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        success = False
        err_type = None
        in_tok = 0
        out_tok = 0

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))
            lc_msgs = self._to_lc_messages(messages)
            kwargs = self._with_ollama_options(self.defaults, temperature=temperature, max_tokens=max_tokens)
            buf: List[str] = []

            try:
                for chunk in self.chat.stream(lc_msgs, **kwargs):
                    c = chunk.content
                    if c:
                        buf.append(c)
                        yield partial_stream_event(delta_content=c)

                out_tok = int(self.estimate_tokens_for_text("".join(buf), model_hint=self.model_name_for_token_estimation))
                success = True
                yield final_stream_event(
                    response=build_adapter_response(
                        content="".join(buf),
                        usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                        model=self.model,
                        provider=self._provider_slug(),
                        provider_extensions=LLMProviderExtensions(usage_source="estimate"),
                    )
                )
                return

            except Exception:
                res = self.chat.invoke(lc_msgs, **kwargs)
                text = res.content or str(res)
                if text:
                    yield partial_stream_event(delta_content=text)
                out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model_name_for_token_estimation))
                success = True
                yield final_stream_event(
                    response=build_adapter_response(
                        content=text,
                        usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                        model=self.model,
                        provider=self._provider_slug(),
                        provider_extensions=LLMProviderExtensions(usage_source="estimate"),
                    )
                )
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



    @property
    def model_capabilities(self) -> OllamaModelCapabilities:
        if self._model_capabilities is None:
            self._model_capabilities = self._capability_resolver.resolve(self.model)
        return self._model_capabilities

    def refresh_model_capabilities(self) -> OllamaModelCapabilities:
        self._model_capabilities = self._capability_resolver.resolve(self.model)
        return self._model_capabilities

    def supports_tools(self) -> bool:
        """Native non-streaming tool calling when the installed model declares tools."""
        return self.model_capabilities.supports_tools

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
    ) -> LLMAdapterResponse:
        if not self.supports_tools():
            raise ValueError(
                f"Ollama model does not declare native tool support: {self.model}"
            )
        self._validate_ollama_tool_choice(tool_choice)
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        response: LLMAdapterResponse | None = None
        success = False
        err_type = None
        in_tok = 0
        out_tok = 0

        try:
            in_tok = int(
                self.estimate_tokens_for_messages(
                    messages,
                    model_hint=self.model_name_for_token_estimation,
                )
            )
            lc_msgs = self._to_lc_messages(messages)
            kwargs = self._with_ollama_options(
                self.defaults,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            bind_kwargs: Dict[str, Any] = {}
            if tool_choice is not None:
                bind_kwargs["tool_choice"] = tool_choice
            bound_chat = self.chat.bind_tools(tools_schema, **bind_kwargs)
            result = self._execute(lambda: bound_chat.invoke(lc_msgs, **kwargs))

            invalid_tool_calls = attribute_access.optional(result, "invalid_tool_calls", None) or []
            if invalid_tool_calls:
                raise ValueError("Ollama returned invalid native tool calls")

            tool_calls = tool_calls_from_langchain_message(result)
            content = self._coerce_ai_message_content(result)
            finish = (
                LLMFinishReason.TOOL_CALLS if tool_calls else LLMFinishReason.COMPLETED
            )
            out_text = self._estimate_tool_output_text(content, tool_calls)
            out_tok = int(
                self.estimate_tokens_for_text(
                    out_text,
                    model_hint=self.model_name_for_token_estimation,
                )
            )
            response = build_adapter_response(
                content=content,
                finish_reason=finish,
                usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                model=self.model,
                provider=self._provider_slug(),
                tool_calls=tool_calls,
                provider_extensions=LLMProviderExtensions(usage_source="estimate"),
            )
            success = True
            return response

        except Exception as e:
            err_type = type(e).__name__
            raise

        finally:
            usage = response.usage if (success and response and response.usage) else LLMTokenUsage()
            self.usage.end_call(
                call,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                success=success,
                error_type=err_type,
            )


    @staticmethod
    def _coerce_parsed_structured_output(output_model: type, parsed: Any) -> Any:
        if isinstance(parsed, output_model):
            return parsed
        if hasattr(output_model, "model_validate"):
            return output_model.model_validate(parsed)  # type: ignore[attr-defined]
        raise TypeError(f"Cannot validate structured output with {output_model}")

    @staticmethod
    def _structured_raw_text(raw: Any, validated: Any) -> str:
        content = attribute_access.optional(raw, "content", None)
        if isinstance(content, str) and content:
            return content
        if hasattr(validated, "model_dump_json"):
            return validated.model_dump_json()  # type: ignore[attr-defined]
        return str(validated)

    # --- Structured output via native Ollama JSON Schema ---
    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMStructuredResult[Any]:
        call = self.usage.begin_call(run_id=run_id, adapter=self)
        response: LLMAdapterResponse | None = None
        success = False
        err_type = None
        in_tok = 0
        out_tok = 0

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))
            lc_msgs = self._to_lc_messages(messages)

            generation_schema = prepare_ollama_generation_schema(output_model)
            structured_chat = self.chat.with_structured_output(
                generation_schema,
                method="json_schema",
                include_raw=True,
            )

            kwargs = self._with_ollama_options(self.defaults, temperature=temperature, max_tokens=max_tokens)
            result = structured_chat.invoke(lc_msgs, **kwargs)

            if not isinstance(result, dict):
                raise ValueError("Ollama structured output returned unexpected result type")

            parsing_error = result.get("parsing_error")
            if parsing_error is not None:
                raise parsing_error

            parsed = result.get("parsed")
            if parsed is None:
                raise ValueError("Ollama structured output returned no parsed result")

            validated = self._coerce_parsed_structured_output(output_model, parsed)
            txt = self._structured_raw_text(result.get("raw"), validated)
            out_tok = int(self.estimate_tokens_for_text(txt, model_hint=self.model_name_for_token_estimation))

            response = build_adapter_response(
                content=txt,
                usage=LLMTokenUsage.from_counts(input_tokens=in_tok, output_tokens=out_tok),
                model=self.model,
                provider=self._provider_slug(),
                provider_extensions=LLMProviderExtensions(usage_source="estimate"),
            )
            success = True
            return LLMStructuredResult(parsed=validated, response=response)

        except Exception as e:
            err_type = type(e).__name__
            raise

        finally:
            usage = response.usage if (success and response and response.usage) else LLMTokenUsage()
            self.usage.end_call(
                call,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                success=success,
                error_type=err_type,
            )

    

