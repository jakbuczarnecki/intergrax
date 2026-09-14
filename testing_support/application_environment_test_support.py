# © Artur Czarnecki. All rights reserved.

"""Tier-3 application environment wiring stubs aligned with production import paths."""

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from testing_support.builder import FakeLLMAdapter


def stub_environment_llm_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    adapter: LLMAdapter | None = None,
) -> LLMAdapter:
    """Patch LLM resolution at symbols consumed by harness and environment wiring."""
    resolved = adapter or FakeLLMAdapter()

    def _resolve(*_args: object, **_kwargs: object) -> LLMAdapter:
        return resolved

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_environment_llm_adapter",
        _resolve,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_optional_environment_llm_adapter",
        _resolve,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.nexus_factory.resolve_environment_llm_adapter",
        _resolve,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.environment_wiring.resolve_optional_environment_llm_adapter",
        _resolve,
    )
    return resolved
