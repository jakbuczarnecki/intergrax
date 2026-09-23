# © Artur Czarnecki. All rights reserved.

"""LLMProfile runtime materialization — pure re-export plus explicit factories."""

from __future__ import annotations

import os
from typing import Mapping, Optional

from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.llm_adapters._shared.call_config import parse_call_config
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile
from intergrax.llm_adapters.registry.secrets import (
    load_api_key_from_secrets_store,
    merge_secrets_into_options,
)


def create_adapter(
    profile: LLMProfile,
    *,
    secrets: Optional[Mapping[str, str]] = None,
    **overrides: object,
) -> LLMAdapter:
    from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

    kwargs = merge_secrets_into_options(
        profile.provider,
        {**profile.options, **overrides},
        secrets,
    )
    if profile.model:
        kwargs.setdefault("model", profile.model)
    return LLMAdapterRegistry.create(profile.provider, **kwargs)


def create_adapter_with_failover(
    profile: LLMProfile,
    *,
    secrets: Optional[Mapping[str, str]] = None,
    policy_route_hint: str | None = None,
    **overrides: object,
) -> LLMAdapter:
    from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
    from intergrax.llm_adapters.registry.model_router import ModelRouter

    hint = policy_route_hint or profile.routing_policy_hint
    router = ModelRouter.from_profiles(
        profile,
        fallbacks=profile.fallback_profiles,
        policy_route_hint=hint,
    )
    ordered_profiles = router.ordered_profiles()
    adapters = [
        create_adapter(candidate, secrets=secrets, **overrides)
        for candidate in ordered_profiles
    ]
    if len(adapters) == 1:
        return adapters[0]
    adapter_failover_retry_configs = [
        parse_call_config(
            merge_secrets_into_options(
                candidate.provider,
                {**candidate.options, **overrides},
                secrets,
            )
        )
        for candidate in ordered_profiles
    ]
    return FailoverLLMAdapter(
        adapters,
        profile_ids=router.ordered_profile_ids(),
        failover_retry_config=adapter_failover_retry_configs[0],
        adapter_failover_retry_configs=adapter_failover_retry_configs,
    )


def validate_runtime(
    profile: LLMProfile,
    *,
    secrets: Optional[Mapping[str, str]] = None,
) -> list[str]:
    from intergrax.llm_adapters.registry.context_window import resolve_context_window_tokens

    warnings: list[str] = []
    model_id = (profile.model or "").strip()
    if not model_id:
        warnings.append("LLMProfile.model is unset")
    else:
        tokens = resolve_context_window_tokens(
            profile.provider,
            model_id,
            profile_options=profile.options,
        )
        if tokens <= 0:
            warnings.append(
                f"context_window_tokens resolved to {tokens} for model={model_id!r}"
            )

    merged = merge_secrets_into_options(
        profile.provider,
        dict(profile.options),
        secrets,
    )
    if not merged.get("api_key"):
        slug = LLMProfile._provider_slug(profile.provider)
        if slug not in {"ollama", "vllm", "llama_cpp"}:
            warnings.append(
                f"no api_key in profile options or secrets for provider={slug}"
            )
    return warnings


def create_adapter_from_secrets_store(
    profile: LLMProfile,
    store: SecretsStore,
    *,
    secret_path: str | None = None,
    **overrides: object,
) -> LLMAdapter:
    key = load_api_key_from_secrets_store(store, profile.provider, path=secret_path)
    return create_adapter(profile, secrets={"api_key": key}, **overrides)


def llm_profile_from_env(*, prefix: str = "INTERGRAX_LLM") -> LLMProfile | None:
    provider_raw = os.getenv(f"{prefix}_PROVIDER")
    if provider_raw is None or not provider_raw.strip():
        return None
    model = os.getenv(f"{prefix}_MODEL")
    return LLMProfile(provider=provider_raw.strip(), model=model or None)


__all__ = [
    "LLMProfile",
    "create_adapter",
    "create_adapter_from_secrets_store",
    "create_adapter_with_failover",
    "llm_profile_from_env",
    "validate_runtime",
]
