# © Artur Czarnecki. All rights reserved.

"""Canonical qualification environment bootstrap (process env > nearest .env > defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.secrets import resolve_api_key
from intergrax.llm_adapters.registry.profile import llm_profile_from_env


@dataclass(frozen=True, slots=True)
class QualificationEnvBootstrapReport:
    dotenv_discovered: str | None
    dotenv_loaded: bool
    provider: str | None
    model: str | None
    qualification_enabled: bool
    credential_available: bool


def _discover_nearest_dotenv(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        env_path = candidate / ".env"
        if env_path.is_file():
            return env_path
    return None


def bootstrap_qualification_environment(
    *,
    start_path: Path | None = None,
    qualification_flag: str = "INTERGRAX_DECISION_E2E_QUALIFICATION",
    llm_prefix: str = "INTERGRAX_LLM",
) -> QualificationEnvBootstrapReport:
    """Resolve env without pre-failing on missing credentials before bootstrap."""
    dotenv_path = _discover_nearest_dotenv(start_path)
    dotenv_loaded = False
    if dotenv_path is not None:
        try:
            from dotenv import load_dotenv
        except ImportError:
            dotenv_loaded = False
        else:
            dotenv_loaded = bool(load_dotenv(dotenv_path, override=False))

    qualification_enabled = os.environ.get(qualification_flag, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    provider: str | None = None
    model: str | None = None
    credential_available = False
    try:
        profile = llm_profile_from_env(prefix=llm_prefix)
        provider = (
            profile.provider.value
            if isinstance(profile.provider, LLMProvider)
            else str(profile.provider)
        )
        model = profile.model
        credential_available = resolve_api_key(profile.provider) is not None
    except (OSError, RuntimeError, TypeError, ValueError):
        provider = os.environ.get(f"{llm_prefix}_PROVIDER")
        model = os.environ.get(f"{llm_prefix}_MODEL")
        raw_provider = (provider or "").strip().lower()
        if raw_provider == LLMProvider.OPENAI.value:
            credential_available = bool(os.environ.get("OPENAI_API_KEY", "").strip())

    return QualificationEnvBootstrapReport(
        dotenv_discovered=str(dotenv_path) if dotenv_path is not None else None,
        dotenv_loaded=dotenv_loaded,
        provider=provider,
        model=model,
        qualification_enabled=qualification_enabled,
        credential_available=credential_available,
    )
