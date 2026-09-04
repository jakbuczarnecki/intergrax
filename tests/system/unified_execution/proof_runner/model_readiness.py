# © Artur Czarnecki. All rights reserved.

"""Provider-neutral certification model readiness bootstrap."""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from tests.system.unified_execution.proof_runner.contracts import ProofConfig
from tests.system.unified_execution.proof_runner.lkw_client import LkwClientError
from tests.system.unified_execution.proof_runner.provider_evidence import probe_ollama_model

_CERTIFICATION_EMBEDDING_PROBE_TEXT = "intergrax certification embedding readiness"
_CERTIFICATION_GENERATION_PROBE_TEXT = "ok"


class ModelCapability(StrEnum):
    EMBEDDING = "EMBEDDING"
    GENERATION = "GENERATION"


class ModelRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model_id: str
    capability: ModelCapability


class ModelReadinessProbeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts: int = 10
    request_timeout_seconds: float = 30.0
    backoff_seconds: float = 3.0


class ModelReadinessResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    provider: str
    capability: ModelCapability
    present: bool
    ready: bool
    attempts: int
    elapsed_seconds: float
    last_error_code: str | None = None


class ModelReadinessReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirements: list[ModelRequirement] = Field(default_factory=list)
    results: list[ModelReadinessResult] = Field(default_factory=list)


class ModelReadinessError(LkwClientError):
    """Bootstrap failure when a required model does not reach inference readiness."""


class OllamaHttpClient:
    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def get_json(self, path: str) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("ollama_response_not_object")
        return parsed

    def post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("ollama_response_not_object")
        return parsed

    def post_raw(self, path: str, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            _ = response.read()


class ProviderReadinessAdapter(Protocol):
    def ensure_present(self, model_id: str) -> bool: ...

    def probe_capability(self, model_id: str, capability: ModelCapability) -> tuple[bool, str | None]: ...


class OllamaReadinessAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        client: OllamaHttpClient,
        pull_timeout_seconds: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._pull_timeout_seconds = pull_timeout_seconds

    def ensure_present(self, model_id: str) -> bool:
        tags_payload = self._client.get_json("/api/tags")
        evidence = probe_ollama_model(tags_payload=tags_payload, model_name=model_id)
        if evidence.listed_after_run:
            return True
        pull_client = OllamaHttpClient(
            base_url=self._base_url,
            timeout_seconds=self._pull_timeout_seconds,
        )
        pull_client.post_raw("/api/pull", {"name": model_id})
        tags_after = self._client.get_json("/api/tags")
        return probe_ollama_model(tags_payload=tags_after, model_name=model_id).listed_after_run

    def probe_capability(
        self,
        model_id: str,
        capability: ModelCapability,
    ) -> tuple[bool, str | None]:
        if capability is ModelCapability.EMBEDDING:
            return _probe_ollama_embedding(self._client, model_id=model_id)
        if capability is ModelCapability.GENERATION:
            return _probe_ollama_generation(self._client, model_id=model_id)
        return False, "unsupported_capability"


def c1_model_requirements(config: ProofConfig) -> list[ModelRequirement]:
    return [
        ModelRequirement(
            provider=config.embedding_provider,
            model_id=config.embedding_model,
            capability=ModelCapability.EMBEDDING,
        ),
    ]


def _sorted_requirements(requirements: list[ModelRequirement]) -> list[ModelRequirement]:
    return sorted(
        requirements,
        key=lambda item: (item.provider, item.capability, item.model_id),
    )


def _adapter_for_provider(
    provider: str,
    config: ProofConfig,
    client: OllamaHttpClient,
) -> OllamaReadinessAdapter:
    if provider != "ollama":
        raise ModelReadinessError(f"unsupported_provider:{provider}")
    return OllamaReadinessAdapter(
        base_url=config.ollama_base_url,
        client=client,
        pull_timeout_seconds=config.readiness_timeout_seconds,
    )


def ensure_model_readiness(
    config: ProofConfig,
    requirements: list[ModelRequirement],
    *,
    probe_config: ModelReadinessProbeConfig | None = None,
) -> ModelReadinessReport:
    probe_settings = probe_config or ModelReadinessProbeConfig()
    sorted_requirements = _sorted_requirements(requirements)
    client = OllamaHttpClient(
        base_url=config.ollama_base_url,
        timeout_seconds=probe_settings.request_timeout_seconds,
    )
    results: list[ModelReadinessResult] = []
    for requirement in sorted_requirements:
        result = _ensure_requirement_ready(
            config=config,
            requirement=requirement,
            client=client,
            probe_settings=probe_settings,
        )
        results.append(result)
        if not result.ready:
            raise ModelReadinessError(
                _format_model_not_ready_failure(result),
            )
    return ModelReadinessReport(requirements=sorted_requirements, results=results)


def _ensure_requirement_ready(
    *,
    config: ProofConfig,
    requirement: ModelRequirement,
    client: OllamaHttpClient,
    probe_settings: ModelReadinessProbeConfig,
) -> ModelReadinessResult:
    started = time.monotonic()
    adapter = _adapter_for_provider(requirement.provider, config, client)
    present = False
    last_error: str | None = "not_started"
    attempts = 0
    ready = False
    try:
        present = adapter.ensure_present(requirement.model_id)
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        last_error = _bounded_error_code(exc)
        present = False

    if not present:
        elapsed = time.monotonic() - started
        return ModelReadinessResult(
            model_id=requirement.model_id,
            provider=requirement.provider,
            capability=requirement.capability,
            present=False,
            ready=False,
            attempts=attempts,
            elapsed_seconds=elapsed,
            last_error_code=last_error,
        )

    while attempts < probe_settings.max_attempts:
        attempts += 1
        try:
            probe_ready, probe_error = adapter.probe_capability(
                requirement.model_id,
                requirement.capability,
            )
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            probe_ready = False
            probe_error = _bounded_error_code(exc)
        if probe_ready:
            ready = True
            last_error = None
            break
        last_error = probe_error or "probe_not_ready"
        if attempts < probe_settings.max_attempts:
            time.sleep(probe_settings.backoff_seconds)

    elapsed = time.monotonic() - started
    return ModelReadinessResult(
        model_id=requirement.model_id,
        provider=requirement.provider,
        capability=requirement.capability,
        present=True,
        ready=ready,
        attempts=attempts,
        elapsed_seconds=elapsed,
        last_error_code=last_error,
    )


def _probe_ollama_embedding(
    client: OllamaHttpClient,
    *,
    model_id: str,
) -> tuple[bool, str | None]:
    payload = client.post_json(
        "/api/embeddings",
        {
            "model": model_id,
            "prompt": _CERTIFICATION_EMBEDDING_PROBE_TEXT,
        },
    )
    embedding = payload.get("embedding")
    if _valid_embedding_vector(embedding):
        return True, None
    return False, "embedding_vector_invalid"


def _probe_ollama_generation(
    client: OllamaHttpClient,
    *,
    model_id: str,
) -> tuple[bool, str | None]:
    payload = client.post_json(
        "/api/generate",
        {
            "model": model_id,
            "prompt": _CERTIFICATION_GENERATION_PROBE_TEXT,
            "stream": False,
            "options": {"num_predict": 1},
        },
    )
    response_text = payload.get("response")
    if isinstance(response_text, str) and response_text.strip():
        return True, None
    return False, "generation_response_invalid"


def _valid_embedding_vector(raw: object) -> bool:
    if not isinstance(raw, list):
        return False
    if len(raw) == 0:
        return False
    for item in raw:
        if not isinstance(item, (int, float)):
            return False
        if not math.isfinite(float(item)):
            return False
    return True


def _bounded_error_code(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_error:{exc.code}"
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, BaseException):
            return f"url_error:{type(reason).__name__}"
        return f"url_error:{reason}"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, json.JSONDecodeError):
        return "json_decode_error"
    if isinstance(exc, ValueError):
        return str(exc)
    if isinstance(exc, OSError):
        return f"os_error:{type(exc).__name__}"
    return type(exc).__name__


def _format_model_not_ready_failure(result: ModelReadinessResult) -> str:
    return (
        "MODEL_NOT_READY"
        f":provider={result.provider}"
        f":model={result.model_id}"
        f":capability={result.capability}"
        f":attempts={result.attempts}"
        f":error={result.last_error_code or 'unknown'}"
    )
