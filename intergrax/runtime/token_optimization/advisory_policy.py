# © Artur Czarnecki. All rights reserved.

"""Advisory policy presets and resolver (TOKEN-7D)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.runtime.token_optimization.advisory_integration import (
    TokenOptimizationAdvisoryIntegrationMode,
    TokenOptimizationAdvisoryIntegrationPolicy,
)

_ALLOWED_OVERRIDE_NAMES: frozenset[str] = frozenset(
    {
        "allow_strategy_enable",
        "allow_strategy_disable",
        "require_review_for_risky_recommendations",
    }
)

_OVERRIDE_FIELD_ORDER: tuple[str, ...] = (
    "allow_strategy_enable",
    "allow_strategy_disable",
    "require_review_for_risky_recommendations",
)


class TokenOptimizationAdvisoryPolicyPreset(StrEnum):
    """Named advisory policy presets (deterministic; no global config)."""

    DISABLED = "disabled"
    REPORT_ONLY = "report_only"
    DRY_RUN_SAFE = "dry_run_safe"
    REVIEW_FIRST = "review_first"
    ADVISORY_ALLOWED_SAFE = "advisory_allowed_safe"


@dataclass(frozen=True, slots=True)
class TokenOptimizationAdvisoryPolicyOverrides:
    """Safe overrides for advisory policy presets (safety switches only)."""

    allow_strategy_enable: bool | None = None
    allow_strategy_disable: bool | None = None
    require_review_for_risky_recommendations: bool | None = None

    def __post_init__(self) -> None:
        for field_name in _OVERRIDE_FIELD_ORDER:
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field_name} override must be bool or None")


@dataclass(frozen=True, slots=True)
class TokenOptimizationAdvisoryPolicyResolution:
    """Resolved advisory policy from a preset (non-auto-apply; redaction-safe)."""

    preset: TokenOptimizationAdvisoryPolicyPreset
    policy: TokenOptimizationAdvisoryIntegrationPolicy
    overrides_applied: tuple[str, ...] = ()
    auto_apply_allowed: bool = False
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        if self.auto_apply_allowed:
            raise ValueError("auto_apply_allowed must remain False")
        if self.raw_content_included:
            raise ValueError("raw_content_included must remain False")
        if self.policy.allow_auto_apply:
            raise ValueError("policy.allow_auto_apply must remain False")
        for name in self.overrides_applied:
            if not isinstance(name, str):
                raise ValueError("overrides_applied values must be strings")
            if name not in _ALLOWED_OVERRIDE_NAMES:
                raise ValueError(f"overrides_applied contains unsupported override: {name}")
        if self.preset is TokenOptimizationAdvisoryPolicyPreset.DISABLED:
            if self.policy.enabled:
                raise ValueError("DISABLED preset must resolve to enabled=False")
            if self.policy.mode is not TokenOptimizationAdvisoryIntegrationMode.DISABLED:
                raise ValueError("DISABLED preset must resolve to mode=DISABLED")
        else:
            if not self.policy.enabled:
                raise ValueError("non-DISABLED preset must resolve to enabled=True")
            if self.policy.mode is TokenOptimizationAdvisoryIntegrationMode.DISABLED:
                raise ValueError("non-DISABLED preset must not resolve to mode=DISABLED")


def _base_policy_for_preset(
    preset: TokenOptimizationAdvisoryPolicyPreset,
) -> TokenOptimizationAdvisoryIntegrationPolicy:
    if preset is TokenOptimizationAdvisoryPolicyPreset.DISABLED:
        return TokenOptimizationAdvisoryIntegrationPolicy.disabled()
    if preset is TokenOptimizationAdvisoryPolicyPreset.REPORT_ONLY:
        return TokenOptimizationAdvisoryIntegrationPolicy(
            enabled=True,
            mode=TokenOptimizationAdvisoryIntegrationMode.REPORT_ONLY,
            allow_strategy_enable=False,
            allow_strategy_disable=True,
            require_review_for_risky_recommendations=True,
            allow_auto_apply=False,
        )
    if preset is TokenOptimizationAdvisoryPolicyPreset.DRY_RUN_SAFE:
        return TokenOptimizationAdvisoryIntegrationPolicy(
            enabled=True,
            mode=TokenOptimizationAdvisoryIntegrationMode.DRY_RUN,
            allow_strategy_enable=False,
            allow_strategy_disable=True,
            require_review_for_risky_recommendations=True,
            allow_auto_apply=False,
        )
    if preset is TokenOptimizationAdvisoryPolicyPreset.REVIEW_FIRST:
        return TokenOptimizationAdvisoryIntegrationPolicy(
            enabled=True,
            mode=TokenOptimizationAdvisoryIntegrationMode.REVIEW_ONLY,
            allow_strategy_enable=False,
            allow_strategy_disable=True,
            require_review_for_risky_recommendations=True,
            allow_auto_apply=False,
        )
    if preset is TokenOptimizationAdvisoryPolicyPreset.ADVISORY_ALLOWED_SAFE:
        return TokenOptimizationAdvisoryIntegrationPolicy(
            enabled=True,
            mode=TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED,
            allow_strategy_enable=False,
            allow_strategy_disable=True,
            require_review_for_risky_recommendations=True,
            allow_auto_apply=False,
        )
    raise ValueError(f"unsupported advisory policy preset: {preset}")


def _apply_overrides(
    base: TokenOptimizationAdvisoryIntegrationPolicy,
    overrides: TokenOptimizationAdvisoryPolicyOverrides,
) -> tuple[TokenOptimizationAdvisoryIntegrationPolicy, tuple[str, ...]]:
    applied: list[str] = []
    allow_strategy_enable = base.allow_strategy_enable
    allow_strategy_disable = base.allow_strategy_disable
    require_review = base.require_review_for_risky_recommendations

    if overrides.allow_strategy_enable is not None:
        allow_strategy_enable = overrides.allow_strategy_enable
        applied.append("allow_strategy_enable")
    if overrides.allow_strategy_disable is not None:
        allow_strategy_disable = overrides.allow_strategy_disable
        applied.append("allow_strategy_disable")
    if overrides.require_review_for_risky_recommendations is not None:
        require_review = overrides.require_review_for_risky_recommendations
        applied.append("require_review_for_risky_recommendations")

    policy = TokenOptimizationAdvisoryIntegrationPolicy(
        enabled=base.enabled,
        mode=base.mode,
        allow_strategy_enable=allow_strategy_enable,
        allow_strategy_disable=allow_strategy_disable,
        require_review_for_risky_recommendations=require_review,
        allow_auto_apply=False,
    )
    return policy, tuple(applied)


def resolve_token_optimization_advisory_policy(
    preset: TokenOptimizationAdvisoryPolicyPreset,
    overrides: TokenOptimizationAdvisoryPolicyOverrides | None = None,
) -> TokenOptimizationAdvisoryPolicyResolution:
    """Resolve a preset (and optional overrides) to an explicit integration policy."""
    if preset is TokenOptimizationAdvisoryPolicyPreset.DISABLED and overrides is not None:
        raise ValueError("DISABLED preset does not accept overrides")

    base = _base_policy_for_preset(preset)
    overrides_applied: tuple[str, ...] = ()
    policy = base

    if overrides is not None:
        policy, overrides_applied = _apply_overrides(base, overrides)

    return TokenOptimizationAdvisoryPolicyResolution(
        preset=preset,
        policy=policy,
        overrides_applied=overrides_applied,
        auto_apply_allowed=False,
        raw_content_included=False,
    )


def _enum_value(value: StrEnum) -> str:
    return value.value


def token_optimization_advisory_policy_resolution_to_dict(
    resolution: TokenOptimizationAdvisoryPolicyResolution,
) -> dict[str, object]:
    """Serialize a policy resolution for JSON output (redaction-safe scalars only)."""
    policy = resolution.policy
    return {
        "preset": _enum_value(resolution.preset),
        "overrides_applied": list(resolution.overrides_applied),
        "auto_apply_allowed": resolution.auto_apply_allowed,
        "raw_content_included": resolution.raw_content_included,
        "policy_enabled": policy.enabled,
        "policy_mode": _enum_value(policy.mode),
        "policy_allow_strategy_enable": policy.allow_strategy_enable,
        "policy_allow_strategy_disable": policy.allow_strategy_disable,
        "policy_require_review_for_risky_recommendations": (
            policy.require_review_for_risky_recommendations
        ),
        "policy_allow_auto_apply": policy.allow_auto_apply,
    }


def format_token_optimization_advisory_policy_resolution(
    resolution: TokenOptimizationAdvisoryPolicyResolution,
) -> str:
    """Human-readable policy resolution (deterministic, redaction-safe)."""
    policy = resolution.policy
    overrides = ",".join(resolution.overrides_applied)
    lines = [
        f"preset={_enum_value(resolution.preset)}",
        f"overrides_applied={overrides}",
        f"auto_apply_allowed={resolution.auto_apply_allowed}",
        f"raw_content_included={resolution.raw_content_included}",
        f"policy_enabled={policy.enabled}",
        f"policy_mode={_enum_value(policy.mode)}",
        f"policy_allow_strategy_enable={policy.allow_strategy_enable}",
        f"policy_allow_strategy_disable={policy.allow_strategy_disable}",
        (
            "policy_require_review_for_risky_recommendations="
            f"{policy.require_review_for_risky_recommendations}"
        ),
        f"policy_allow_auto_apply={policy.allow_auto_apply}",
    ]
    return "\n".join(lines)
