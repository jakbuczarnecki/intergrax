# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.context_window import resolve_context_window_tokens
from intergrax.llm_adapters.registry.model_catalog import (
    ModelCatalog,
    ModelRecord,
    PrefixRule,
    get_model_catalog,
    reset_model_catalog_cache,
)


@pytest.fixture(autouse=True)
def _clear_catalog_cache() -> None:
    reset_model_catalog_cache()
    yield
    reset_model_catalog_cache()


def test_bundled_catalog_loads_at_least_fifty_models() -> None:
    catalog = get_model_catalog()
    assert len(catalog.models) >= 50


def test_resolve_override_wins_over_catalog() -> None:
    tokens = resolve_context_window_tokens(
        LLMProvider.CLAUDE,
        "claude-3-5-sonnet-latest",
        profile_options={"context_window_tokens": 999_999},
    )
    assert tokens == 999_999


def test_resolve_exact_match() -> None:
    tokens = resolve_context_window_tokens(
        LLMProvider.OPENAI,
        "gpt-4o",
    )
    assert tokens == 128_000


def test_resolve_prefix_rule() -> None:
    tokens = resolve_context_window_tokens(
        LLMProvider.CLAUDE,
        "claude-unknown-future-model",
    )
    assert tokens == 200_000


def test_resolve_provider_default_for_unknown_openrouter_model() -> None:
    tokens = resolve_context_window_tokens(
        LLMProvider.OPENROUTER,
        "vendor/obscure-model-v9",
    )
    assert tokens == 128_000


def test_catalog_overlay_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        """
models:
  - model_id: custom-local-model
    context_window_tokens: 65536
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("INTERGRAX_LLM_MODEL_CATALOG_PATH", str(overlay))
    reset_model_catalog_cache()
    catalog = get_model_catalog()
    record = catalog.lookup_exact("custom-local-model")
    assert record is not None
    assert record.context_window_tokens == 65_536


def test_model_record_immutable() -> None:
    record = ModelRecord(model_id="x", context_window_tokens=1000)
    with pytest.raises(Exception):
        record.context_window_tokens = 2000  # type: ignore[misc]


def test_prefix_longest_match() -> None:
    catalog = ModelCatalog.from_mapping(
        {
            "models": [],
            "prefix_rules": [
                {"prefix": "gemini-", "context_window_tokens": 128000},
                {"prefix": "gemini-1.5", "context_window_tokens": 1000000},
            ],
            "provider_defaults": {},
            "fallback_default": 32000,
        }
    )
    assert catalog.lookup_prefix("gemini-1.5-pro") == 1_000_000
    assert catalog.lookup_prefix("gemini-2.0-flash") == 128_000
