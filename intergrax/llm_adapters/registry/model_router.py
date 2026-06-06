# © Artur Czarnecki. All rights reserved.

"""Policy-driven LLM model routing (FAUDIT-LLM.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.llm_adapters.registry.profile import LLMProfile


@dataclass(frozen=True, slots=True)
class ModelRoutingDecision:
    """Resolved model route for a single inference call."""

    profile_id: str
    provider: str
    model: str
    fallback_profile_id: str | None = None
    routing_reason: str = "environment_default"


class ModelRouter:
    """Select LLM profile using explicit policy hints and environment defaults."""

    def __init__(
        self,
        *,
        primary: LLMProfile,
        fallback: LLMProfile | None = None,
        policy_route_hint: str | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._policy_route_hint = (policy_route_hint or "").strip()

    def _profile_id(self, profile: LLMProfile) -> str:
        model = profile.model or "default"
        return f"{profile.provider.value}:{model}"

    def resolve(self) -> ModelRoutingDecision:
        if self._policy_route_hint == "balanced" and self._fallback is not None:
            return ModelRoutingDecision(
                profile_id=self._profile_id(self._fallback),
                provider=str(self._fallback.provider.value),
                model=self._fallback.model or "",
                fallback_profile_id=self._profile_id(self._primary),
                routing_reason="policy_hint_balanced",
            )
        return ModelRoutingDecision(
            profile_id=self._profile_id(self._primary),
            provider=str(self._primary.provider.value),
            model=self._primary.model or "",
            fallback_profile_id=self._profile_id(self._fallback) if self._fallback else None,
            routing_reason="primary_profile",
        )

    @classmethod
    def from_profiles(
        cls,
        primary: LLMProfile,
        *,
        fallback: LLMProfile | None = None,
        policy_route_hint: str | None = None,
    ) -> ModelRouter:
        return cls(primary=primary, fallback=fallback, policy_route_hint=policy_route_hint)
