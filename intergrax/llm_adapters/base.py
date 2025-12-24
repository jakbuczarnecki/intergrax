# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass, field
import time
from enum import Enum
from abc import ABC, abstractmethod
from typing import Sequence, Iterable, Optional, Any, Dict, Union, List
import json
import re
from typing import Callable
import tiktoken

# if TYPE_CHECKING:
#     from ..llm.conversational_memory import ChatMessage
# else:
#     ChatMessage = Any  # runtime stub
from intergrax.memory.conversational_memory import ChatMessage


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    MISTRAL = "mistral"
    CLAUDE = "claude"
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"


@dataclass
class LLMCallStats:
    run_id: str
    t0: float = field(default_factory=time.perf_counter)

    # filled on end
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0

    success: bool = True
    error_type: Optional[str] = None


@dataclass
class LLMRunStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    errors: int = 0

# ============================================================
# Universal interface (ABC)
# ============================================================

class LLMAdapter(ABC):
    """
    Universal runtime interface for LLM adapters.

    This used to be a Protocol. It is now an ABC to provide:
      - strong runtime guarantees (abstract methods)
      - shared base implementation (token counting)

    Required:
      - generate_messages(...)
      - context_window_tokens

    Optional (default to NotImplemented/False):
      - streaming
      - tools
      - structured output
    """

    # Hint used by the generic token estimator (e.g. OpenAI model name).
    model_name_for_token_estimation: Optional[str] = None

    def __init__(self) -> None:
        self._run_stats: Dict[str, LLMRunStats] = {}
    

    def begin_call(self, run_id: Optional[str] = None) -> LLMCallStats:
        """
        Begin one LLM call (not the whole runtime.run()).

        Returns a per-call context object, safe for nested/parallel use
        because it is local to the caller.
        """
        rid = run_id or "general"
        if rid not in self._run_stats:
            self._run_stats[rid] = LLMRunStats()
        return LLMCallStats(run_id=rid)


    def end_call(
        self,
        call: LLMCallStats,
        *,
        input_tokens: int,
        output_tokens: int,
        success: bool = True,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Finish one LLM call and aggregate into per-run stats.
        """
        dt_ms = int((time.perf_counter() - call.t0) * 1000)

        call.input_tokens = int(input_tokens or 0)
        call.output_tokens = int(output_tokens or 0)
        call.total_tokens = call.input_tokens + call.output_tokens
        call.duration_ms = dt_ms

        call.success = bool(success)
        call.error_type = error_type

        st = self._run_stats.get(call.run_id)
        if st is None:
            st = LLMRunStats()
            self._run_stats[call.run_id] = st

        st.calls += 1
        st.input_tokens += call.input_tokens
        st.output_tokens += call.output_tokens
        st.total_tokens += call.total_tokens
        st.duration_ms += call.duration_ms

        if not call.success:
            st.errors += 1
    
    def get_run_stats(self, run_id: Optional[str] = None) -> Optional[LLMRunStats]:
        """
        Get aggregated stats for a given run_id.
        Returns None if no stats exist for that run_id.
        """
        rid = run_id or "general"
        return self._run_stats.get(rid)


    def get_all_run_stats(self) -> Dict[str, LLMRunStats]:
        """
        Get a shallow copy of all aggregated run stats.
        """
        return dict(self._run_stats)


    def reset_run_stats(self, run_id: Optional[str] = None) -> None:
        """
        Reset stats for a specific run_id (or 'general' if None).
        """
        rid = run_id or "general"
        if rid in self._run_stats:
            del self._run_stats[rid]


    def reset_all_run_stats(self) -> None:
        """
        Reset all stored stats.
        """
        self._run_stats.clear()


    def export_run_stats_dict(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export aggregated stats to a JSON-serializable dict.
        Helpful for debug_trace / logging.
        """
        rid = run_id or "general"
        st = self._run_stats.get(rid)
        if st is None:
            return {
                "run_id": rid,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "duration_ms": 0,
                "errors": 0,
            }

        return {
            "run_id": rid,
            "calls": int(st.calls),
            "input_tokens": int(st.input_tokens),
            "output_tokens": int(st.output_tokens),
            "total_tokens": int(st.total_tokens),
            "duration_ms": int(st.duration_ms),
            "errors": int(st.errors),
        }



    @abstractmethod
    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> str:
        raise NotImplementedError


    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[str]:
        raise NotImplementedError("Streaming is not supported by this adapter.")


    # ---- Tools (optional) ----
    def supports_tools(self) -> bool:
        return False


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
        raise NotImplementedError("Tools are not supported by this adapter.")

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
        raise NotImplementedError("Tools streaming is not supported by this adapter.")

    # ---- Structured output (optional) ----
    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ):
        raise NotImplementedError("Structured output is not supported by this adapter.")

    # ---- Token counting (base impl; moved from the removed LLMAdapter) ----
    def count_messages_tokens(self, messages: Sequence[ChatMessage]) -> int:
        return self.estimate_tokens_for_messages(
            messages,
            model_hint=self.model_name_for_token_estimation,
        )

    @property
    @abstractmethod
    def context_window_tokens(self) -> int:
        raise NotImplementedError
    
    
    def _strip_code_fences(self, text: str) -> str:
        """
        Remove wrappers like ```json ... ``` or ``` ... ``` if present.
        Useful when the model wraps a JSON object in Markdown fences.
        """
        if not text:
            return text
        fence_re = r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$"
        m = re.match(fence_re, text, flags=re.DOTALL)
        return m.group(1) if m else text


    def _extract_json_object(self, text: str) -> str:
        """
        Try to extract the first {...} block that looks like a JSON object.
        Returns an empty string if none is found.

        This is tolerant to extra text around the JSON.
        """
        if not text:
            return ""
        text = self._strip_code_fences(text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return ""
        return text[start : end + 1]


    def _model_json_schema(self, model_cls: type) -> Dict[str, Any]:
        """
        Return JSON Schema for the model class (Pydantic v2/v1).
        If unavailable, return a minimal object schema.
        """
        # pydantic v2
        if hasattr(model_cls, "model_json_schema"):
            try:
                return model_cls.model_json_schema()  # type: ignore[attr-defined]
            except Exception:
                pass

        # pydantic v1
        if hasattr(model_cls, "schema"):
            try:
                return model_cls.schema()  # type: ignore[attr-defined]
            except Exception:
                pass

        # fallback
        return {"type": "object"}


    def _validate_with_model(self, model_cls: type, json_str: str):
        """
        Validate and create a model instance from JSON string.

        Supports:
        - Pydantic v2 (model_validate_json/model_validate)
        - Pydantic v1 (parse_raw/parse_obj)
        - Plain dataclasses/classes via **data
        """
        if not json_str or not json_str.strip():
            raise ValueError("Empty JSON content for structured output.")

        data = json.loads(json_str)

        # pydantic v2
        if hasattr(model_cls, "model_validate_json"):
            try:
                return model_cls.model_validate_json(json_str)  # type: ignore[attr-defined]
            except Exception:
                pass

        if hasattr(model_cls, "model_validate"):
            try:
                return model_cls.model_validate(data)  # type: ignore[attr-defined]
            except Exception:
                pass

        # pydantic v1
        if hasattr(model_cls, "parse_raw"):
            try:
                return model_cls.parse_raw(json_str)  # type: ignore[attr-defined]
            except Exception:
                pass

        if hasattr(model_cls, "parse_obj"):
            try:
                return model_cls.parse_obj(data)  # type: ignore[attr-defined]
            except Exception:
                pass

        # fallback (plain class/dataclass with compatible __init__)
        try:
            return model_cls(**data)
        except Exception as e:
            raise ValueError(f"Cannot validate structured output with {model_cls}: {e}")


    def estimate_tokens_for_messages(
        self,
        messages: Sequence[ChatMessage],
        model_hint: Optional[str] = None,
    ) -> int:
        """
        Estimate token count for a list of ChatMessage objects.

        Strategy:
        - If tiktoken is available:
            * use encoding_for_model(model_hint) when model_hint is provided,
            * otherwise fall back to a generic encoding (e.g. cl100k_base).
        - If tiktoken is not available:
            * use a simple character-based heuristic (approx. 4 chars/token).

        This is a generic, model-agnostic estimator designed to be "good enough"
        for budgeting and trimming, not for billing accuracy.
        """
        # Aggregate all message contents into a single string.
        parts: List[str] = []
        for m in messages:
            content = m.content
            if not isinstance(content, str):
                content = str(content)
            parts.append(content)
        joined = "\n".join(parts)
        if not joined:
            return 0

        if model_hint:
            enc = tiktoken.encoding_for_model(model_hint)
        else:
            # Reasonable default for many chat models.
            enc = tiktoken.get_encoding("cl100k_base")

        return len(enc.encode(joined))


    def estimate_tokens_for_text(
        self,
        text: str,
        model_hint: Optional[str] = None,
    ) -> int:
        """
        Estimate token count for a plain text string.

        Uses the same strategy as estimate_tokens_for_messages:
        - tiktoken encoding_for_model(model_hint) if available
        - else cl100k_base
        - fallback heuristic if needed
        """
        if not text:
            return 0

        mh = model_hint or self.model_name_for_token_estimation
        try:
            if mh:
                enc = tiktoken.encoding_for_model(mh)
            else:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)


# ============================================================
# Adapter registry
# ============================================================
class LLMAdapterRegistry:
    _factories: Dict[str, Any] = {}

    @staticmethod
    def _normalize_provider(provider: Union[str, LLMProvider]) -> str:
        if isinstance(provider, LLMProvider):
            return provider.value
        p = str(provider).strip().lower()
        if not p:
            raise ValueError("provider must not be empty")
        return p

    @classmethod
    def register(cls, provider: Union[str, LLMProvider], factory: Callable[..., LLMAdapter]) -> None:
        key = cls._normalize_provider(provider)
        cls._factories[key] = factory

    @classmethod
    def create(cls, provider: Union[str, LLMProvider], **kwargs) -> LLMAdapter:
        key = cls._normalize_provider(provider)
        if key not in cls._factories:
            raise ValueError(f"LLM adapter not registered for provider='{key}'")
        return cls._factories[key](**kwargs)
