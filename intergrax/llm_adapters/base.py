# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Protocol, Sequence, Iterable, Optional, Any, Dict, Union, List, TYPE_CHECKING
import json
import re
import tiktoken

# if TYPE_CHECKING:
#     from ..llm.conversational_memory import ChatMessage
# else:
#     ChatMessage = Any  # runtime stub
from intergrax.memory.conversational_memory import ChatMessage

__all__ = [
    "LLMAdapter",
    "LLMAdapterRegistry",
    "BaseModel",
    "BaseLLMAdapter",
    "estimate_tokens_for_messages",
    "_extract_json_object",
    "_model_json_schema",
    "_validate_with_model",
    "_map_messages_to_openai",
]

# ============================================================
# Pydantic compatibility (v2/v1/fallback) – for structured output only
# ============================================================
try:
    from pydantic import BaseModel  # type: ignore
    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover - pydantic not installed
    class BaseModel: ...
    _HAS_PYDANTIC = False


def _strip_code_fences(text: str) -> str:
    """
    Remove wrappers like ```json ... ``` or ``` ... ``` if present.
    Useful when the model wraps a JSON object in Markdown fences.
    """
    if not text:
        return text
    fence_re = r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$"
    m = re.match(fence_re, text, flags=re.DOTALL)
    return m.group(1) if m else text


def _extract_json_object(text: str) -> str:
    """
    Try to extract the first {...} block that looks like a JSON object.
    Returns an empty string if none is found.

    This is tolerant to extra text around the JSON.
    """
    if not text:
        return ""
    text = _strip_code_fences(text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def _model_json_schema(model_cls: type) -> Dict[str, Any]:
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


def _validate_with_model(model_cls: type, json_str: str):
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


class BaseLLMAdapter:
    """
    Optional base class providing shared utilities such as token counting.

    Adapters may subclass this to inherit the default implementation of
    `count_messages_tokens`. If a specific provider exposes a more accurate
    or native token counter, the adapter can override this method.
    """

    # Hint used by the generic token estimator (e.g. OpenAI model name).
    model_name_for_token_estimation: Optional[str] = None

    def count_messages_tokens(self, messages: Sequence[ChatMessage]) -> int:
        """
        Default token counting implementation.

        Uses `estimate_tokens_for_messages` with an optional model hint.
        Concrete adapters are free to override this method if they can
        provide a provider-specific implementation.
        """
        return estimate_tokens_for_messages(
            messages,
            model_hint=self.model_name_for_token_estimation,
        )



# ============================================================
# Universal interface (protocol)
# ============================================================
class LLMAdapter(Protocol):
    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        ...

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterable[str]:
        ...

    # ---- Tools (optional) ----
    def supports_tools(self) -> bool:
        ...

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        ...

    def stream_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Iterable[Dict[str, Any]]:
        ...

    # ---- Structured output (optional) ----
    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        ...


    def count_messages_tokens(self, messages: Sequence[ChatMessage]) -> int:
        """
        Return approximate token count for the given list of messages.
        A best-effort implementation is acceptable.
        """
        ...


    # ---- Context window (required) ----
    @property
    def context_window_tokens(self) -> int:
        """
        Maximum number of tokens the underlying model can accept in a single
        request (input + output), as defined by the model provider.

        Implementations should compute this once (e.g. in __init__) based on
        the configured model name and cache it in a private attribute.
        """
        ...


# ============================================================
# Helper – convert ChatMessage → OpenAI schema
# ============================================================
def _map_messages_to_openai(msgs: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
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


# ============================================================
# Adapter registry
# ============================================================
class LLMAdapterRegistry:
    """
    Simple string-keyed registry for LLM adapter factories.

    Usage:
        LLMAdapterRegistry.register("openai", lambda **kw: OpenAIChatResponsesAdapter(**kw))
        adapter = LLMAdapterRegistry.create("openai", client=..., model=...)
    """

    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, factory) -> None:
        cls._registry[name.lower()] = factory

    @classmethod
    def create(cls, name: str, **kwargs) -> LLMAdapter:
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(f"Unknown adapter: {name}")
        return cls._registry[key](**kwargs)
