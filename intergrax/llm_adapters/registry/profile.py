# © Artur Czarnecki. All rights reserved.

"""LLMProfile runtime materialization — re-exports public contract."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Mapping, Optional

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile
from intergrax.llm_adapters.registry.secrets import (
    load_api_key_from_secrets_store,
    merge_secrets_into_options,
)

if TYPE_CHECKING:
    from intergrax.integrations.contracts.secrets_store import SecretsStore


def _create_adapter(
    self: LLMProfile,
    *,
    secrets: Optional[Mapping[str, str]] = None,
    **overrides: Any,
) -> LLMAdapter:
    from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

    kwargs = merge_secrets_into_options(
        self.provider,
        {**self.options, **overrides},
        secrets,
    )
    if self.model:
        kwargs.setdefault("model", self.model)
    return LLMAdapterRegistry.create(self.provider, **kwargs)


def _create_adapter_with_failover(
    self: LLMProfile,
    *,
    secrets: Optional[Mapping[str, str]] = None,
    policy_route_hint: str | None = None,
    **overrides: Any,
) -> LLMAdapter:
    from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
    from intergrax.llm_adapters.registry.model_router import ModelRouter

    hint = policy_route_hint or self.routing_policy_hint
    router = ModelRouter.from_profiles(
        self,
        fallbacks=self.fallback_profiles,
        policy_route_hint=hint,
    )
    ordered_profiles = router.ordered_profiles()
    adapters = [
        profile.create_adapter(secrets=secrets, **overrides)
        for profile in ordered_profiles
    ]
    if len(adapters) == 1:
        return adapters[0]
    return FailoverLLMAdapter(
        adapters,
        profile_ids=router.ordered_profile_ids(),
    )


def _validate_runtime(
    self: LLMProfile,
    *,
    secrets: Optional[Mapping[str, str]] = None,
) -> list[str]:
    from intergrax.llm_adapters.registry.context_window import resolve_context_window_tokens

    warnings: list[str] = []
    model_id = (self.model or "").strip()
    if not model_id:
        warnings.append("LLMProfile.model is unset")
    else:
        tokens = resolve_context_window_tokens(
            self.provider,
            model_id,
            profile_options=self.options,
        )
        if tokens <= 0:
            warnings.append(
                f"context_window_tokens resolved to {tokens} for model={model_id!r}"
            )

    merged = merge_secrets_into_options(
        self.provider,
        dict(self.options),
        secrets,
    )
    if not merged.get("api_key"):
        slug = LLMProfile._provider_slug(self.provider)
        if slug not in {"ollama", "vllm", "llama_cpp"}:
            warnings.append(
                f"no api_key in profile options or secrets for provider={slug}"
            )
    return warnings


def _create_adapter_from_secrets_store(
    self: LLMProfile,
    store: "SecretsStore",
    *,
    secret_path: str | None = None,
    **overrides: Any,
) -> LLMAdapter:
    key = load_api_key_from_secrets_store(store, self.provider, path=secret_path)
    return self.create_adapter(secrets={"api_key": key}, **overrides)


setattr(LLMProfile, "create_adapter", _create_adapter)
setattr(LLMProfile, "create_adapter_with_failover", _create_adapter_with_failover)
setattr(LLMProfile, "validate_runtime", _validate_runtime)
setattr(LLMProfile, "create_adapter_from_secrets_store", _create_adapter_from_secrets_store)


def llm_profile_from_env(*, prefix: str = "INTERGRAX_LLM") -> LLMProfile | None:
    provider_raw = os.getenv(f"{prefix}_PROVIDER")
    if provider_raw is None or not provider_raw.strip():
        return None
    model = os.getenv(f"{prefix}_MODEL")
    return LLMProfile(provider=provider_raw.strip(), model=model or None)


__all__ = ["LLMProfile", "llm_profile_from_env"]
