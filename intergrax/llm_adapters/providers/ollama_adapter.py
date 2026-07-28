# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence
from langchain_ollama import ChatOllama

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
    final_stream_event,
    partial_stream_event,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.provider_extensions import LLMProviderExtensions
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.providers._ollama_schema import prepare_ollama_generation_schema
from intergrax.llm_adapters.registry.context_window import init_adapter_context_window_tokens


class LangChainOllamaAdapter(LLMAdapter):
    """
    Adapter for Ollama models used via LangChain's ChatModel interface.

    There is no native tools API here, so the agent typically uses a
    "planner" pattern (tool calls are reasoned about in JSON).
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
        **defaults,
    ):
        super().__init__()
        self._apply_defaults_call_config(defaults)

        resolved_model = model or os.getenv(self.ENV_MODEL) or self.DEFAULT_MODEL
        self.chat = chat or ChatOllama(model=resolved_model)
        self.defaults = defaults

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



    def supports_tools(self) -> bool:
        """Ollama uses JSON planner pattern in ToolsAgent, not native tool API."""
        return False

    def supports_structured_output(self) -> bool:
        return True


    @staticmethod
    def _coerce_parsed_structured_output(output_model: type, parsed: Any) -> Any:
        if isinstance(parsed, output_model):
            return parsed
        if hasattr(output_model, "model_validate"):
            return output_model.model_validate(parsed)  # type: ignore[attr-defined]
        raise TypeError(f"Cannot validate structured output with {output_model}")

    @staticmethod
    def _structured_raw_text(raw: Any, validated: Any) -> str:
        content = getattr(raw, "content", None)
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

    

