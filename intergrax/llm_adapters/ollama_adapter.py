# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence
from langchain_ollama import ChatOllama

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm_adapters.base import (    
    ChatMessage,
    LLMAdapter,
)


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

    def __init__(self, chat : Optional[ChatOllama] = None, context_window_tokens: int = None, **defaults):
        super().__init__()

        self.chat = chat or ChatOllama(
            model=GLOBAL_SETTINGS.default_ollama_model
        )
        self.defaults = defaults

        if context_window_tokens is not None and context_window_tokens > 0:
            # User-provided value = authoritative.
            self._context_window_tokens = int(context_window_tokens)
        else:
            # Otherwise estimate from the model name (fallback path).
            self._context_window_tokens = int(
                self._estimate_ollama_context_window_from_model(self.chat.model)
            )


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
    ) -> str:
        call = self.usage.begin_call(run_id=run_id)

        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            lc_msgs = self._to_lc_messages(messages)
            kwargs = self._with_ollama_options(self.defaults, temperature=temperature, max_tokens=max_tokens)
            res = self.chat.invoke(lc_msgs, **kwargs)

            text = res.content or str(res)
            out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model_name_for_token_estimation))

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
                        yield c

                # normal stream end
                out_tok = int(self.estimate_tokens_for_text("".join(buf), model_hint=self.model_name_for_token_estimation))
                success = True
                return

            except Exception:
                # Streaming failed -> fallback to single-shot (but DO NOT call self.generate_messages here)
                res = self.chat.invoke(lc_msgs, **kwargs)
                text = res.content or str(res)

                # emit fallback full text as one chunk
                if text:
                    yield text

                out_tok = int(self.estimate_tokens_for_text(text, model_hint=self.model_name_for_token_estimation))
                success = True
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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ):
        call = self.usage.begin_call(run_id=run_id)

        in_tok = 0
        out_tok = 0
        success = False
        err_type = None

        try:
            in_tok = int(self.estimate_tokens_for_messages(messages, model_hint=self.model_name_for_token_estimation))

            schema = self._model_json_schema(output_model)

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
            txt = res.content or str(res)

            out_tok = int(self.estimate_tokens_for_text(txt, model_hint=self.model_name_for_token_estimation))

            json_str = self._extract_json_object(txt) or txt.strip()
            if not json_str:
                raise ValueError("Model did not return JSON content for structured output (Ollama).")

            obj = self._validate_with_model(output_model, json_str)
            success = True
            return obj

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

    

