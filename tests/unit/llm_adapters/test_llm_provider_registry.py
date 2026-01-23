# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unit tests for LLMAdapterRegistry.

These tests define the behavioral contract for adapter registration and creation:
- provider normalization is deterministic (strip + lowercase; enum values supported),
- invalid providers fail fast,
- register() overwrites existing factories explicitly,
- create() raises a clear error for unknown providers,
- create() forwards kwargs to the underlying factory,
- factories must return a valid LLMAdapter instance.

Why this matters:
LLMAdapterRegistry is a central wiring mechanism. Regressions here can break
adapter resolution across the system in subtle, hard-to-debug ways.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, Sequence, Optional

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.llm_adapter import LLMAdapter
from intergrax.llm_adapters.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry


pytestmark = pytest.mark.unit


_Factory = Callable[..., LLMAdapter]


# ---------------------------------------------------------------------------
# Minimal test adapter
# ---------------------------------------------------------------------------

class _TestAdapter(LLMAdapter):
    """
    Minimal concrete LLMAdapter used for registry contract tests.
    """

    provider = "unit-test"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = dict(kwargs)

    @property
    def context_window_tokens(self) -> int:
        return 1

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> str:
        return "ok"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _restore_registry_state() -> Iterator[Dict[str, _Factory]]:
    """
    Snapshot and restore the global registry state.

    LLMAdapterRegistry uses class-level global state. Unit tests must isolate changes
    to avoid cross-test coupling and flakiness.
    """
    snapshot: Dict[str, _Factory] = dict(LLMAdapterRegistry._factories)
    try:
        yield snapshot
    finally:
        LLMAdapterRegistry._factories = snapshot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_normalize_provider_accepts_enum_values(_restore_registry_state: Dict[str, Any]) -> None:
    """
    Enum providers must normalize to their canonical string values.
    """
    key = LLMAdapterRegistry._normalize_provider(LLMProvider.OPENAI)
    assert key == LLMProvider.OPENAI.value


def test_normalize_provider_strips_and_lowercases(_restore_registry_state: Dict[str, Any]) -> None:
    """
    String providers must be stripped and lowercased to ensure stable lookup keys.
    """
    key = LLMAdapterRegistry._normalize_provider("  OpEnAI  ")
    assert key == "openai"


def test_normalize_provider_rejects_empty(_restore_registry_state: Dict[str, Any]) -> None:
    """
    Empty or whitespace-only providers must fail fast.
    """
    with pytest.raises(ValueError) as exc:
        LLMAdapterRegistry._normalize_provider("   ")

    assert "provider must not be empty" in str(exc.value)


def test_register_overwrites_existing_factory(_restore_registry_state: Dict[str, Any]) -> None:
    """
    register() must NOT silently overwrite existing factories.
    Overwrite is only allowed when override=True is explicitly provided.
    """
    provider = "unit-test-provider"

    def factory_v1(**kwargs: Any) -> LLMAdapter:
        return _TestAdapter(version="v1")

    def factory_v2(**kwargs: Any) -> LLMAdapter:
        return _TestAdapter(version="v2")

    # First registration works
    LLMAdapterRegistry.register(provider, factory_v1)
    out1 = LLMAdapterRegistry.create(provider)
    assert isinstance(out1, _TestAdapter)
    assert out1.kwargs["version"] == "v1"

    # Second registration WITHOUT override must fail
    with pytest.raises(ValueError) as exc:
        LLMAdapterRegistry.register(provider, factory_v2)

    assert "already registered" in str(exc.value)

    # Explicit override must succeed
    LLMAdapterRegistry.register(provider, factory_v2, override=True)
    out2 = LLMAdapterRegistry.create(provider)
    assert isinstance(out2, _TestAdapter)
    assert out2.kwargs["version"] == "v2"


def test_create_raises_for_unregistered_provider(_restore_registry_state: Dict[str, Any]) -> None:
    """
    create() must raise a clear error when provider is not registered.
    """
    with pytest.raises(ValueError) as exc:
        LLMAdapterRegistry.create("missing-provider")

    msg = str(exc.value)
    assert "LLM adapter not registered" in msg
    assert "missing-provider" in msg


def test_create_forwards_kwargs_to_factory(_restore_registry_state: Dict[str, Any]) -> None:
    """
    create() must forward kwargs to the registered factory.
    """
    provider = "unit-test-kwargs"

    def factory(**kwargs: Any) -> LLMAdapter:
        return _TestAdapter(**kwargs)

    LLMAdapterRegistry.register(provider, factory)

    out = LLMAdapterRegistry.create(provider, x=1, y="a")
    assert isinstance(out, _TestAdapter)
    assert out.kwargs == {"x": 1, "y": "a"}


def test_normalize_provider_rejects_non_string_and_non_enum(_restore_registry_state: Dict[str, Any]) -> None:
    """
    Non-string and non-enum provider values must be rejected explicitly.
    """
    with pytest.raises((TypeError, ValueError)):
        LLMAdapterRegistry._normalize_provider(None)  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError)):
        LLMAdapterRegistry._normalize_provider(42)  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError)):
        LLMAdapterRegistry._normalize_provider(object())  # type: ignore[arg-type]
