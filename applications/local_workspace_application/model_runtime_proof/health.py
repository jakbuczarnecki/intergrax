# © Artur Czarnecki. All rights reserved.

"""Provider health and identity probes for model runtime portability proof."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.catalog_capabilities import (
    unwrap_catalog_capability_adapter,
)
from intergrax.llm_adapters.providers.ollama_capabilities import (
    OllamaModelCapabilityResolver,
)
from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
from intergrax.llm_adapters.providers.openai_compat_providers import VllmChatAdapter

from local_workspace_application.model_runtime_proof.config import (
    ModelRuntimeProofConfig,
    classify_endpoint,
    materialize_provider_env,
)
from local_workspace_application.model_runtime_proof.contracts import ProofFailureCode
from local_workspace_application.model_runtime_proof.safety import (
    normalize_provider_error,
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_PREFIX_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_DIGEST_LEN = 128


@dataclass(frozen=True, slots=True)
class ProviderHealthSnapshot:
    provider: Literal["ollama", "vllm"]
    configured_model: str
    resolved_model: str
    server_model: str
    adapter_class: str
    server_version: str | None
    base_url_classification: str
    server_model_digest: str | None = None
    capability_metadata: tuple[str, ...] = ()


def _http_json(url: str, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        body = json.loads(raw) if raw else {}
        if not isinstance(body, dict):
            raise ValueError("health_response_not_object")
        return body


def normalize_model_digest(digest: str) -> str | None:
    value = digest.strip()
    if not value or len(value) > _MAX_DIGEST_LEN:
        return None
    if _SHA256_PREFIX_RE.fullmatch(value):
        return value
    if _SHA256_HEX_RE.fullmatch(value):
        return f"sha256:{value}"
    return None


def _resolve_ollama_model_digest(
    models: list[Any],
    *,
    model_name: str,
) -> str | None:
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name != model_name:
            continue
        raw_digest = item.get("digest")
        if not isinstance(raw_digest, str):
            return None
        return normalize_model_digest(raw_digest)
    return None


def probe_ollama_health(
    config: ModelRuntimeProofConfig,
) -> tuple[ProviderHealthSnapshot | None, ProofFailureCode | None, str | None]:
    base = config.ollama_base_url.rstrip("/")
    try:
        version_body = _http_json(f"{base}/api/version", timeout=config.timeout_seconds)
        tags_body = _http_json(f"{base}/api/tags", timeout=config.timeout_seconds)
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return (
            None,
            ProofFailureCode.PROVIDER_UNREACHABLE,
            normalize_provider_error(exc)[1],
        )

    models = tags_body.get("models", [])
    installed = {
        str(item.get("name", "")).strip()
        for item in models
        if isinstance(item, dict) and str(item.get("name", "")).strip()
    }
    if config.ollama_model not in installed:
        return (
            None,
            ProofFailureCode.PROVIDER_MODEL_MISSING,
            f"model={config.ollama_model}",
        )

    server_model_digest = _resolve_ollama_model_digest(
        models,
        model_name=config.ollama_model,
    )
    if server_model_digest is None:
        return (
            None,
            ProofFailureCode.PROVIDER_IDENTITY_MISMATCH,
            "model_digest_missing",
        )

    env = materialize_provider_env(provider="ollama", config=config)
    adapter = unwrap_catalog_capability_adapter(
        LLMAdapterRegistry.create(
            LLMProvider.OLLAMA,
            model=env["INTERGRAX_LLM_MODEL"],
            base_url=env["OLLAMA_HOST"],
        )
    )
    if not isinstance(adapter, LangChainOllamaAdapter):
        return (
            None,
            ProofFailureCode.PROVIDER_IDENTITY_MISMATCH,
            adapter.__class__.__name__,
        )

    resolved_model = str(getattr(adapter, "model", "") or config.ollama_model)
    if resolved_model != config.ollama_model:
        return None, ProofFailureCode.PROVIDER_MODEL_MISMATCH, resolved_model

    resolver = OllamaModelCapabilityResolver(base_url=env["OLLAMA_HOST"])
    caps = resolver.resolve(config.ollama_model)
    metadata = tuple(sorted(caps.capabilities)) if caps.resolved else ()

    return (
        ProviderHealthSnapshot(
            provider="ollama",
            configured_model=config.ollama_model,
            resolved_model=resolved_model,
            server_model=config.ollama_model,
            adapter_class=adapter.__class__.__name__,
            server_version=str(version_body.get("version", "")) or None,
            base_url_classification=classify_endpoint(base),
            server_model_digest=server_model_digest,
            capability_metadata=metadata,
        ),
        None,
        None,
    )


def probe_vllm_health(
    config: ModelRuntimeProofConfig,
) -> tuple[ProviderHealthSnapshot | None, ProofFailureCode | None, str | None]:
    base = config.vllm_base_url.rstrip("/")
    models_url = f"{base}/models"
    try:
        body = _http_json(models_url, timeout=config.timeout_seconds)
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return (
            None,
            ProofFailureCode.PROVIDER_UNREACHABLE,
            normalize_provider_error(exc)[1],
        )

    data = body.get("data", [])
    served = {
        str(item.get("id", "")).strip()
        for item in data
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    if config.vllm_model not in served:
        return (
            None,
            ProofFailureCode.PROVIDER_MODEL_MISSING,
            f"model={config.vllm_model}",
        )

    env = materialize_provider_env(provider="vllm", config=config)
    adapter = unwrap_catalog_capability_adapter(
        LLMAdapterRegistry.create(
            LLMProvider.VLLM,
            model=env["INTERGRAX_LLM_MODEL"],
            base_url=env["INTERGRAX_DEFAULT_VLLM_BASE_URL"],
        )
    )
    if not isinstance(adapter, VllmChatAdapter):
        return (
            None,
            ProofFailureCode.PROVIDER_IDENTITY_MISMATCH,
            adapter.__class__.__name__,
        )

    resolved_model = str(getattr(adapter, "model", "") or config.vllm_model)
    if resolved_model != config.vllm_model:
        return None, ProofFailureCode.PROVIDER_MODEL_MISMATCH, resolved_model

    server_version = None
    try:
        version_body = _http_json(f"{base.rsplit('/v1', 1)[0]}/version", timeout=5.0)
        server_version = str(version_body.get("version", "")) or None
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ):
        server_version = None

    return (
        ProviderHealthSnapshot(
            provider="vllm",
            configured_model=config.vllm_model,
            resolved_model=resolved_model,
            server_model=config.vllm_model,
            adapter_class=adapter.__class__.__name__,
            server_version=server_version,
            base_url_classification=classify_endpoint(base),
            capability_metadata=(),
        ),
        None,
        None,
    )


def probe_provider_health(
    provider: Literal["ollama", "vllm"],
    config: ModelRuntimeProofConfig,
) -> tuple[ProviderHealthSnapshot | None, ProofFailureCode | None, str | None]:
    if provider == "ollama":
        return probe_ollama_health(config)
    return probe_vllm_health(config)
