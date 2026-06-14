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


_ROUTING_HINTS = frozenset({"balanced", "cheapest", "fastest", "quality"})


class ModelRouter:
    """Select LLM profile using explicit policy hints and environment defaults."""

    def __init__(
        self,
        *,
        primary: LLMProfile,
        fallback: LLMProfile | None = None,
        fallbacks: tuple[LLMProfile, ...] = (),
        policy_route_hint: str | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        extra = tuple(fallbacks)
        if fallback is not None and fallback not in extra:
            extra = (fallback, *extra)
        self._fallbacks = extra
        self._policy_route_hint = (policy_route_hint or "").strip().lower()

    def _profile_id(self, profile: LLMProfile) -> str:
        model = profile.model or "default"
        return f"{profile.provider.value}:{model}"

    def ordered_profiles(self) -> tuple[LLMProfile, ...]:
        """Return profiles in routing order for adapter creation."""
        hint = self._policy_route_hint
        if hint == "balanced" and self._fallbacks:
            return (self._fallbacks[0], self._primary, *self._fallbacks[1:])
        if hint == "cheapest" and self._fallbacks:
            return (*self._fallbacks, self._primary)
        if hint == "fastest" and self._fallbacks:
            return (self._primary, *self._fallbacks)
        if hint == "quality" and self._fallbacks:
            return (self._primary, *reversed(self._fallbacks))
        if self._fallbacks:
            return (self._primary, *self._fallbacks)
        return (self._primary,)

    def resolve(self) -> ModelRoutingDecision:
        ordered = self.ordered_profiles()
        selected = ordered[0]
        fallback_id = self._profile_id(ordered[1]) if len(ordered) > 1 else None
        reason = "primary_profile"
        if self._policy_route_hint in _ROUTING_HINTS:
            reason = f"policy_hint_{self._policy_route_hint}"
        elif self._fallbacks:
            reason = "primary_with_fallbacks"
        return ModelRoutingDecision(
            profile_id=self._profile_id(selected),
            provider=str(selected.provider.value),
            model=selected.model or "",
            fallback_profile_id=fallback_id,
            routing_reason=reason,
        )

    @classmethod
    def from_profiles(
        cls,
        primary: LLMProfile,
        *,
        fallback: LLMProfile | None = None,
        fallbacks: tuple[LLMProfile, ...] = (),
        policy_route_hint: str | None = None,
    ) -> ModelRouter:
        return cls(
            primary=primary,
            fallback=fallback,
            fallbacks=fallbacks,
            policy_route_hint=policy_route_hint,
        )
