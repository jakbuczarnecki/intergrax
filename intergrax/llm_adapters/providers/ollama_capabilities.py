# © Artur Czarnecki. All rights reserved.

"""Model-aware capability resolution for Ollama adapters (TOKEN-9)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from intergrax.utils import attribute_access


class OllamaCapabilityResolutionSource(StrEnum):
    API_SHOW = "api_show"
    EXPLICIT_TEST_OVERRIDE = "explicit_test_override"
    UNAVAILABLE = "unavailable"


def _normalize_capabilities(raw: object) -> frozenset[str]:
    if raw is None:
        return frozenset()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return frozenset()
    normalized: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        name = item.strip().lower()
        if name:
            normalized.add(name)
    return frozenset(normalized)


@dataclass(frozen=True, slots=True)
class OllamaModelCapabilities:
    model: str
    capabilities: frozenset[str]
    resolved: bool
    source: OllamaCapabilityResolutionSource
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not self.model or not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.resolved and self.capabilities:
            raise ValueError("resolved=False must not claim capabilities")

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.capabilities


def _default_show_model(base_url: str | None) -> Callable[[str], object]:
    from ollama import Client

    client = Client(host=base_url) if base_url else Client()

    def show(model: str) -> object:
        return client.show(model=model)

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
            raise ValueError("model must be non-empty")
        try:
            payload = self._show_model(trimmed)
            raw_caps = attribute_access.optional(payload, "capabilities", None)
            capabilities = _normalize_capabilities(raw_caps)
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
