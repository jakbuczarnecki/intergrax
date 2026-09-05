# © Artur Czarnecki. All rights reserved.

"""Environment resolution for DS-E2E qualification."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile, llm_profile_from_env


_QUALIFICATION_FLAG = "INTERGRAX_DECISION_E2E_QUALIFICATION"
_REQUIRED_FLAG = "INTERGRAX_DECISION_E2E_REQUIRED"


def qualification_required() -> bool:
    raw = os.environ.get(_QUALIFICATION_FLAG, "").strip().lower()
    return raw in {"1", "true", "yes"}


def qualification_strict_required() -> bool:
    raw = os.environ.get(_REQUIRED_FLAG, "").strip().lower()
    return raw in {"1", "true", "yes"}


def docker_cli_available() -> bool:
    return shutil.which("docker") is not None


def docker_daemon_available() -> bool:
    if not docker_cli_available():
        return False
    completed = subprocess.run(
        ["docker", "info"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


@dataclass(frozen=True, slots=True)
class ProviderBindingEvidence:
    """Safe provider identity without secrets."""

    profile_id: str
    provider: str
    model: str | None
    host: str | None = None


@dataclass(frozen=True, slots=True)
class QualificationEnvironment:
    """Resolved real-provider qualification environment."""

    producer_profile: LLMProfile
    producer_adapter: LLMAdapter
    verifier_profile: LLMProfile
    verifier_adapter: LLMAdapter
    council_profile_b: LLMProfile
    council_adapter_b: LLMAdapter
    council_profile_c: LLMProfile
    council_adapter_c: LLMAdapter
    producer_evidence: ProviderBindingEvidence
    verifier_evidence: ProviderBindingEvidence
    council_b_evidence: ProviderBindingEvidence
    council_c_evidence: ProviderBindingEvidence
    independence_level: str

    def provider_evidence_tuple(self) -> tuple[ProviderBindingEvidence, ...]:
        return (
            self.producer_evidence,
            self.verifier_evidence,
            self.council_b_evidence,
            self.council_c_evidence,
        )


def _profile_from_env_prefix(prefix: str, fallback: LLMProfile) -> LLMProfile:
    provider_raw = os.getenv(f"{prefix}_PROVIDER", "").strip()
    if not provider_raw:
        return fallback
    model = os.getenv(f"{prefix}_MODEL")
    return LLMProfile(provider=provider_raw, model=model or fallback.model)


def _binding_evidence(profile_id: str, profile: LLMProfile) -> ProviderBindingEvidence:
    provider = (
        profile.provider.value
        if isinstance(profile.provider, LLMProvider)
        else str(profile.provider)
    )
    host = os.environ.get("OLLAMA_HOST") if provider == LLMProvider.OLLAMA.value else None
    return ProviderBindingEvidence(
        profile_id=profile_id,
        provider=provider,
        model=profile.model,
        host=host,
    )


def _adapter_supports_structured(adapter: LLMAdapter) -> bool:
    return adapter.supports_structured_output()


def resolve_qualification_environment() -> tuple[QualificationEnvironment | None, str | None]:
    """Resolve provider bindings or return ``(None, block_reason)``."""
    if not qualification_required():
        return None, f"{_QUALIFICATION_FLAG} is not enabled"

    producer_profile = llm_profile_from_env()
    if producer_profile.model is None and producer_profile.provider == LLMProvider.OLLAMA:
        return None, "INTERGRAX_LLM_MODEL is required for Ollama qualification"

    try:
        producer_adapter = producer_profile.create_adapter()
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"producer adapter unavailable: {type(exc).__name__}"

    if not _adapter_supports_structured(producer_adapter):
        return None, "producer adapter does not support structured output"

    verifier_profile = _profile_from_env_prefix(
        "INTERGRAX_DECISION_E2E_VERIFIER_LLM",
        producer_profile,
    )
    council_b_profile = _profile_from_env_prefix(
        "INTERGRAX_DECISION_E2E_COUNCIL_LLM_B",
        producer_profile,
    )
    council_c_profile = _profile_from_env_prefix(
        "INTERGRAX_DECISION_E2E_COUNCIL_LLM_C",
        council_b_profile,
    )

    try:
        verifier_adapter = verifier_profile.create_adapter()
        council_adapter_b = council_b_profile.create_adapter()
        council_adapter_c = council_c_profile.create_adapter()
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"secondary adapter unavailable: {type(exc).__name__}"

    for label, adapter in (
        ("verifier", verifier_adapter),
        ("council-b", council_adapter_b),
        ("council-c", council_adapter_c),
    ):
        if not _adapter_supports_structured(adapter):
            return None, f"{label} adapter does not support structured output"

    independence_bits: list[str] = []
    if verifier_profile.model != producer_profile.model:
        independence_bits.append("distinct_verifier_model")
    if verifier_profile.provider != producer_profile.provider:
        independence_bits.append("distinct_verifier_provider")
    if council_b_profile.model != producer_profile.model:
        independence_bits.append("distinct_council_b_model")
    if council_c_profile.model != council_b_profile.model:
        independence_bits.append("distinct_council_c_model")
    independence_level = (
        "+".join(independence_bits) if independence_bits else "profile_id_only"
    )

    return QualificationEnvironment(
        producer_profile=producer_profile,
        producer_adapter=producer_adapter,
        verifier_profile=verifier_profile,
        verifier_adapter=verifier_adapter,
        council_profile_b=council_b_profile,
        council_adapter_b=council_adapter_b,
        council_profile_c=council_c_profile,
        council_adapter_c=council_adapter_c,
        producer_evidence=_binding_evidence("profile-producer", producer_profile),
        verifier_evidence=_binding_evidence("profile-verifier", verifier_profile),
        council_b_evidence=_binding_evidence("profile-b", council_b_profile),
        council_c_evidence=_binding_evidence("profile-c", council_c_profile),
        independence_level=independence_level,
    ), None
