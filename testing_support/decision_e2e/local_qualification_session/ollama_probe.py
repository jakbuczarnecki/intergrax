# © Artur Czarnecki. All rights reserved.

"""Ollama runtime identity probe for qualification preconditions."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationRuntimeIdentity,
)
from testing_support.decision_e2e.local_qualification_session.versioning import (
    parse_provider_runtime_version,
)


@dataclass(frozen=True, slots=True)
class OllamaProbeConfig:
    base_url: str
    timeout_sec: float = 5.0


def _normalize_model_digest(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if text.startswith("sha256:"):
        return text
    if len(text) >= 64 and all(ch in "0123456789abcdef" for ch in text.lower()):
        return f"sha256:{text.lower()}"
    return text


def _digest_from_tags_list(
    base: str,
    model_name: str,
    *,
    timeout_sec: float,
) -> str | None:
    tags_payload = _http_json(f"{base}/api/tags", timeout_sec=timeout_sec)
    if tags_payload is None:
        return None
    models = tags_payload.get("models")
    if not isinstance(models, list):
        return None
    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if name != model_name and entry.get("model") != model_name:
            continue
        digest = entry.get("digest")
        if isinstance(digest, str):
            normalized = _normalize_model_digest(digest)
            if normalized is not None:
                return normalized
    return None


def _http_json(url: str, *, timeout_sec: float) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def probe_ollama_runtime_identity(
    config: OllamaProbeConfig,
    *,
    model_name: str | None = None,
) -> QualificationRuntimeIdentity | None:
    base = config.base_url.strip().rstrip("/")
    version_payload = _http_json(f"{base}/api/version", timeout_sec=config.timeout_sec)
    if version_payload is None:
        return None
    version_raw = version_payload.get("version")
    runtime_version = (
        parse_provider_runtime_version(str(version_raw))
        if isinstance(version_raw, str)
        else None
    )

    model_digest: str | None = None
    quantization: str | None = None
    resolved_model = model_name
    if model_name:
        show_payload = _ollama_show_post(base, model_name, timeout_sec=config.timeout_sec)
        if show_payload is not None:
            digest = show_payload.get("digest")
            if isinstance(digest, str) and digest:
                model_digest = _normalize_model_digest(digest)
            details = show_payload.get("details")
            if isinstance(details, dict):
                quant = details.get("quantization_level")
                if isinstance(quant, str) and quant:
                    quantization = quant
            model_info_name = show_payload.get("model")
            if isinstance(model_info_name, str) and model_info_name:
                resolved_model = model_info_name
        if model_digest is None:
            model_digest = _digest_from_tags_list(
                base,
                model_name,
                timeout_sec=config.timeout_sec,
            )

    return QualificationRuntimeIdentity(
        provider_kind="ollama",
        runtime_version=runtime_version,
        endpoint_host=base,
        model_name=resolved_model,
        model_digest=model_digest,
        quantization=quantization,
    )


def _ollama_show_post(
    base: str,
    model_name: str,
    *,
    timeout_sec: float,
) -> dict[str, object] | None:
    body = json.dumps({"name": model_name}).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/api/show",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload
