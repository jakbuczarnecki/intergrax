# © Artur Czarnecki. All rights reserved.

"""Model-aware capability resolution for Ollama adapters (TOKEN-9)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast
from dataclasses import dataclass
from enum import StrEnum

from intergrax.utils import attribute_access


class OllamaCapabilityResolutionSource(StrEnum):
    API_SHOW = "api_show"
    EXPLICIT_TEST_OVERRIDE = "explicit_test_override"
    UNAVAILABLE = "unavailable"


def _normalize_capability_sequence(raw: object) -> tuple[frozenset[str], bool, str | None]:
    """Return normalized capabilities, resolved flag, and optional error_type."""
    if raw is None:
        return frozenset(), False, "MissingCapabilities"
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return frozenset(), False, "InvalidCapabilities"
    normalized: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            return frozenset(), False, "InvalidCapabilities"
        name = item.strip().lower()
        if name:
            normalized.add(name)
    return frozenset(normalized), True, None


@dataclass(frozen=True, slots=True)
class OllamaModelCapabilities:
    model: str
    capabilities: frozenset[str]
    resolved: bool
    source: OllamaCapabilityResolutionSource
    error_type: str | None = None

    def __post_init__(self) -> None:
        if self.resolved and (not self.model or not self.model.strip()):
            raise ValueError("resolved=True requires non-empty model")
        if not self.resolved and self.capabilities:
            raise ValueError("resolved=False must not claim capabilities")

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.capabilities


def _default_show_model(base_url: str | None) -> Callable[[str], object]:
    client: object | None = None

    def show(model: str) -> object:
        nonlocal client
        if client is None:
            try:
                from ollama import Client
            except ModuleNotFoundError as exc:
                if exc.name != "ollama":
                    raise
                from intergrax.llm_adapters.llm_provider_registry import (
                    LLMAdapterDependencyError,
                )

                raise LLMAdapterDependencyError(
                    "LLM provider 'ollama' requires dependency 'ollama'. "
                    "Install it with 'Intergrax-ai[llm-ollama]' before selecting "
                    "this provider."
                ) from exc

            client = Client(host=base_url) if base_url else Client()

        show_client = cast(Callable[..., object], client)
        return show_client.show(model=model)

    return show


class OllamaModelCapabilityResolver:
    """Resolve installed Ollama model capabilities via ``/api/show``."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        show_model: Callable[[str], object] | None = None,
    ) -> None:
        self._base_url = base_url
        if show_model is not None:
            self._show_model = show_model
            self._source = OllamaCapabilityResolutionSource.EXPLICIT_TEST_OVERRIDE
        else:
            self._show_model = _default_show_model(base_url)
            self._source = OllamaCapabilityResolutionSource.API_SHOW

    def resolve(self, model: str) -> OllamaModelCapabilities:
        trimmed = (model or "").strip()
        if not trimmed:
            return OllamaModelCapabilities(
                model="",
                capabilities=frozenset(),
                resolved=False,
                source=OllamaCapabilityResolutionSource.UNAVAILABLE,
                error_type="ValueError",
            )
        try:
            payload = self._show_model(trimmed)
            raw_caps = attribute_access.optional(payload, "capabilities", None)
            capabilities, resolved, error_type = _normalize_capability_sequence(raw_caps)
            if not resolved:
                return OllamaModelCapabilities(
                    model=trimmed,
                    capabilities=frozenset(),
                    resolved=False,
                    source=OllamaCapabilityResolutionSource.UNAVAILABLE,
                    error_type=error_type,
                )
            return OllamaModelCapabilities(
                model=trimmed,
                capabilities=capabilities,
                resolved=True,
                source=self._source,
            )
        except Exception as exc:
            return OllamaModelCapabilities(
                model=trimmed,
                capabilities=frozenset(),
                resolved=False,
                source=OllamaCapabilityResolutionSource.UNAVAILABLE,
                error_type=type(exc).__name__,
            )
