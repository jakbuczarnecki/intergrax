# © Artur Czarnecki. All rights reserved.

"""Qualification environment preflight helpers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

from testing_support.decision_e2e.bindings import ProviderBindingEvidence


@dataclass(frozen=True, slots=True)
class OllamaModelInventory:
    host: str
    installed_models: frozenset[str]


def _ollama_base_url() -> str:
    return (os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip().rstrip("/")


def list_ollama_installed_models(
    *,
    base_url: str | None = None,
    timeout_sec: float = 5.0,
) -> OllamaModelInventory | None:
    """Return installed Ollama model names or ``None`` when unreachable."""
    raw = (base_url or _ollama_base_url()).strip().rstrip("/")
    tags_url = f"{raw}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None
    models_raw = payload.get("models")
    if not isinstance(models_raw, list):
        return None
    names: set[str] = set()
    for item in models_raw:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return OllamaModelInventory(host=raw, installed_models=frozenset(names))


def _model_available(installed: frozenset[str], required_model: str) -> bool:
    if required_model in installed:
        return True
    base = required_model.split(":", 1)[0]
    return any(name == base or name.startswith(f"{base}:") for name in installed)


def verify_required_ollama_models(
    required_models: frozenset[str],
    *,
    base_url: str | None = None,
) -> tuple[bool, str | None]:
    inventory = list_ollama_installed_models(base_url=base_url)
    if inventory is None:
        return False, "Ollama API unreachable for model inventory preflight"
    missing = tuple(
        model
        for model in sorted(required_models)
        if not _model_available(inventory.installed_models, model)
    )
    if missing:
        installed = ", ".join(sorted(inventory.installed_models)) or "none"
        required = ", ".join(sorted(required_models))
        return (
            False,
            f"Missing required Ollama models: {', '.join(missing)} "
            f"(required={required}; installed={installed}; host={inventory.host})",
        )
    return True, None


def required_models_for_bindings(
    bindings: tuple[ProviderBindingEvidence, ...],
) -> frozenset[str]:
    models: set[str] = set()
    for binding in bindings:
        if binding.model:
            models.add(binding.model)
    return frozenset(models)


def preflight_qualification_bindings(
    bindings: tuple[ProviderBindingEvidence, ...],
) -> tuple[bool, str | None]:
    """Verify required models are available when Ollama is in use."""
    uses_ollama = any(binding.provider == LLMProvider.OLLAMA.value for binding in bindings)
    if not uses_ollama:
        return True, None
    required = required_models_for_bindings(bindings)
    if not required:
        return False, "Ollama qualification requires INTERGRAX_LLM_MODEL"
    return verify_required_ollama_models(required)
