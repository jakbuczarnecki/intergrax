# © Artur Czarnecki. All rights reserved.

"""Proof configuration loading and canonical provider env materialization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from local_workspace_application.model_runtime_proof.contracts import ProofFailureCode

VllmProvisioningClassification = Literal[
    "committed_compose_sufficient",
    "external_runtime",
    "unverified",
]

_SUPPORTED_VLLM_PROVISIONING: frozenset[str] = frozenset(
    {"committed_compose_sufficient", "external_runtime", "unverified"}
)


@dataclass(frozen=True, slots=True)
class ModelRuntimeProofConfig:
    ollama_model: str
    vllm_model: str
    ollama_base_url: str
    vllm_base_url: str
    tenant_id: str
    data_home: str
    timeout_seconds: float
    vector_store: Literal["qdrant", "inmemory"] = "qdrant"
    require_live_providers: bool = True
    vllm_provisioning_classification: VllmProvisioningClassification = "unverified"

    def validate(self) -> list[ProofFailureCode]:
        errors: list[ProofFailureCode] = []
        if not self.ollama_model.strip():
            errors.append(ProofFailureCode.CONFIG_INVALID)
        if not self.vllm_model.strip():
            errors.append(ProofFailureCode.CONFIG_INVALID)
        if not self.ollama_base_url.strip():
            errors.append(ProofFailureCode.CONFIG_INVALID)
        if not self.vllm_base_url.strip():
            errors.append(ProofFailureCode.CONFIG_INVALID)
        if not self.tenant_id.strip():
            errors.append(ProofFailureCode.CONFIG_INVALID)
        if self.timeout_seconds <= 0:
            errors.append(ProofFailureCode.CONFIG_INVALID)
        if self.vllm_provisioning_classification not in _SUPPORTED_VLLM_PROVISIONING:
            errors.append(ProofFailureCode.CONFIG_INVALID)
        return errors


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def load_vllm_provisioning_classification_from_env() -> str:
    return _env(
        "LKW_MODEL_RUNTIME_PROOF_VLLM_PROVISIONING_CLASSIFICATION",
        "unverified",
    )


def load_proof_config_from_env() -> ModelRuntimeProofConfig:
    vector_raw = _env("LKW_MODEL_RUNTIME_PROOF_VECTOR_STORE", "qdrant").lower()
    vector_store: Literal["qdrant", "inmemory"] = (
        "inmemory" if vector_raw == "inmemory" else "qdrant"
    )
    return ModelRuntimeProofConfig(
        ollama_model=_env(
            "LKW_MODEL_RUNTIME_PROOF_OLLAMA_MODEL", _env("INTERGRAX_LLM_MODEL")
        ),
        vllm_model=_env(
            "LKW_MODEL_RUNTIME_PROOF_VLLM_MODEL", _env("INTERGRAX_LLM_MODEL")
        ),
        ollama_base_url=_env(
            "LKW_MODEL_RUNTIME_PROOF_OLLAMA_BASE_URL",
            _env("OLLAMA_HOST", "http://127.0.0.1:11434"),
        ),
        vllm_base_url=_env(
            "LKW_MODEL_RUNTIME_PROOF_VLLM_BASE_URL",
            _env("INTERGRAX_DEFAULT_VLLM_BASE_URL", "http://127.0.0.1:8100/v1"),
        ),
        tenant_id=_env("LKW_MODEL_RUNTIME_PROOF_TENANT_ID", "lkw-model-runtime-proof"),
        data_home=_env("LKW_MODEL_RUNTIME_PROOF_DATA_HOME", ""),
        timeout_seconds=float(
            _env("LKW_MODEL_RUNTIME_PROOF_TIMEOUT_SECONDS", "300") or "300"
        ),
        vector_store=vector_store,
        require_live_providers=_env("INTERGRAX_LKW_MODEL_RUNTIME_PROOF", "0") == "1",
        vllm_provisioning_classification=load_vllm_provisioning_classification_from_env(),  # type: ignore[assignment]
    )


def materialize_provider_env(
    *,
    provider: Literal["ollama", "vllm"],
    config: ModelRuntimeProofConfig,
    target: dict[str, str] | None = None,
) -> dict[str, str]:
    """Materialize canonical adapter env vars for one conversation provider."""
    env = dict(target or os.environ)
    if provider == "ollama":
        env["INTERGRAX_LLM_PROVIDER"] = "ollama"
        env["INTERGRAX_LLM_MODEL"] = config.ollama_model
        env["OLLAMA_HOST"] = config.ollama_base_url.rstrip("/")
    else:
        env.pop("OLLAMA_HOST", None)
        env["INTERGRAX_LLM_PROVIDER"] = "vllm"
        env["INTERGRAX_LLM_MODEL"] = config.vllm_model
        env["INTERGRAX_DEFAULT_VLLM_BASE_URL"] = config.vllm_base_url.rstrip("/")
    return env


def apply_env(env: dict[str, str]) -> None:
    for key, value in env.items():
        os.environ[key] = value


def classify_endpoint(url: str) -> str:
    lowered = url.lower()
    if "127.0.0.1" in lowered or "localhost" in lowered:
        return "loopback"
    if lowered.startswith("http://"):
        return "private_http"
    if lowered.startswith("https://"):
        return "remote_https"
    return "unknown"
