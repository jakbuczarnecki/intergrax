# © Artur Czarnecki. All rights reserved.

"""Gate AdaptiveProfile modes via Tier-0 feature-flag backends (Phase M.6 P4 follow-up)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import AdaptiveMode, AdaptiveProfile
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.feature_flag import FeatureFlagBackend
from intergrax.integrations.registry.profile import IntegrationProfile


def resolve_effective_adaptive_profile(
    profile: AdaptiveProfile,
    *,
    integration_profile: IntegrationProfile,
    tenant_id: str,
    user_id: str = "",
) -> AdaptiveProfile:
    """
    Apply feature-flag rollout guardrails before wiring adaptive stores.

    When ``rollout_flag_key`` is set and configured mode is beyond ``observe``,
    downgrade to ``observe`` unless the flag backend reports enabled for the tenant.
    """
    if not profile.enabled or profile.mode == "observe":
        return profile
    flag_key = profile.rollout_flag_key.strip()
    if not flag_key:
        return profile

    slug = profile.feature_flag_slug or integration_profile.slug_for_category(
        IntegrationCategory.FEATURE_FLAG
    )
    if not slug:
        return profile.model_copy(update={"mode": "observe"})

    backend = integration_profile.resolve(IntegrationCategory.FEATURE_FLAG)
    if not isinstance(backend, FeatureFlagBackend):
        return profile.model_copy(update={"mode": "observe"})
    if backend.is_enabled(flag_key, tenant_id=tenant_id, user_id=user_id):
        return profile
    return profile.model_copy(update={"mode": "observe"})


def effective_adaptive_mode(
    profile: AdaptiveProfile,
    *,
    integration_profile: IntegrationProfile,
    tenant_id: str,
    user_id: str = "",
) -> AdaptiveMode:
    """Return the post-gate adaptive mode for logging and runtime config."""
    return resolve_effective_adaptive_profile(
        profile,
        integration_profile=integration_profile,
        tenant_id=tenant_id,
        user_id=user_id,
    ).mode
