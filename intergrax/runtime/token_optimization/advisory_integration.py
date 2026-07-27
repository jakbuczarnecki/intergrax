# © Artur Czarnecki. All rights reserved.

"""Policy-gated advisory integration surface (TOKEN-7C)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from intergrax.runtime.token_optimization.advisory import recommend_token_optimization_action
from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationAdvisoryRecommendation,
    TokenOptimizationAdvisorySignal,
    TokenOptimizationRecommendationAction,
    TokenOptimizationSourceType,
)

_FORBIDDEN_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "prompt",
        "content",
        "context",
        "evidence",
        "tool_output",
        "raw_payload",
        "document_text",
        "source_text",
        "recommendation",
        "signal",
    }
)

_RISKY_RECOMMENDATION_ACTIONS: frozenset[TokenOptimizationRecommendationAction] = frozenset(
    {
        TokenOptimizationRecommendationAction.ESCALATE_TO_FULL_CONTEXT,
        TokenOptimizationRecommendationAction.REQUIRE_MANUAL_REVIEW,
        TokenOptimizationRecommendationAction.DISABLE_STRATEGY,
    }
)


def _validate_non_empty_stripped(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be non-empty after stripping")
    return stripped


def _validate_optional_request_id(request_id: str | None) -> str | None:
    if request_id is None:
        return None
    return _validate_non_empty_stripped(request_id, "request_id")


def _validate_string_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    validated: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("metadata keys and values must be strings")
        if key in _FORBIDDEN_METADATA_KEYS:
            raise ValueError(f"metadata must not contain raw-content-like key: {key}")
        validated[key] = value
    return validated


def _enum_value(value: StrEnum | None) -> str | None:
    if value is None:
        return None
    return value.value


class TokenOptimizationAdvisoryIntegrationMode(StrEnum):
    """Advisory integration operating mode (explicit policy; no global config)."""

    DISABLED = "disabled"
    REPORT_ONLY = "report_only"
    DRY_RUN = "dry_run"
    REVIEW_ONLY = "review_only"
    ADVISORY_ALLOWED = "advisory_allowed"


class TokenOptimizationAdvisoryIntegrationStatus(StrEnum):
    """Outcome status for a policy-gated advisory integration request."""

    BLOCKED_BY_POLICY = "blocked_by_policy"
    REPORT_ONLY = "report_only"
    DRY_RUN = "dry_run"
    REVIEW_REQUIRED = "review_required"
    RECOMMENDATION_READY = "recommendation_ready"
    INSUFFICIENT_SIGNALS = "insufficient_signals"


class TokenOptimizationAdvisoryGateReason(StrEnum):
    """Why the advisory integration gate reached its outcome."""

    POLICY_DISABLED = "policy_disabled"
    MODE_DISABLED = "mode_disabled"
    REPORT_ONLY_MODE = "report_only_mode"
    DRY_RUN_MODE = "dry_run_mode"
    REVIEW_ONLY_MODE = "review_only_mode"
    ADVISORY_ALLOWED = "advisory_allowed"
    INSUFFICIENT_SIGNALS = "insufficient_signals"
    STRATEGY_ENABLE_NOT_ALLOWED = "strategy_enable_not_allowed"
    STRATEGY_DISABLE_NOT_ALLOWED = "strategy_disable_not_allowed"
    RISK_REQUIRES_REVIEW = "risk_requires_review"
    MANUAL_REVIEW_RECOMMENDED = "manual_review_recommended"
    AUTO_APPLY_FORBIDDEN = "auto_apply_forbidden"


@dataclass(frozen=True, slots=True)
class TokenOptimizationAdvisoryIntegrationPolicy:
    """Explicit advisory integration policy passed per request (no env/YAML resolver)."""

    enabled: bool = True
    mode: TokenOptimizationAdvisoryIntegrationMode = (
        TokenOptimizationAdvisoryIntegrationMode.DRY_RUN
    )
    allow_strategy_enable: bool = False
    allow_strategy_disable: bool = True
    require_review_for_risky_recommendations: bool = True
    allow_auto_apply: bool = False

    def __post_init__(self) -> None:
        if self.allow_auto_apply:
            raise ValueError("allow_auto_apply must remain False")
        if not self.enabled and self.mode is not TokenOptimizationAdvisoryIntegrationMode.DISABLED:
            raise ValueError("when enabled is False, mode must be DISABLED")
        if self.enabled and self.mode is TokenOptimizationAdvisoryIntegrationMode.DISABLED:
            raise ValueError("when enabled is True, mode must not be DISABLED")

    @classmethod
    def disabled(cls) -> TokenOptimizationAdvisoryIntegrationPolicy:
        return cls(
            enabled=False,
            mode=TokenOptimizationAdvisoryIntegrationMode.DISABLED,
        )

    @classmethod
    def dry_run(cls) -> TokenOptimizationAdvisoryIntegrationPolicy:
        return cls(mode=TokenOptimizationAdvisoryIntegrationMode.DRY_RUN)

    @classmethod
    def report_only(cls) -> TokenOptimizationAdvisoryIntegrationPolicy:
        return cls(mode=TokenOptimizationAdvisoryIntegrationMode.REPORT_ONLY)

    @classmethod
    def review_only(cls) -> TokenOptimizationAdvisoryIntegrationPolicy:
        return cls(mode=TokenOptimizationAdvisoryIntegrationMode.REVIEW_ONLY)


@dataclass(frozen=True, slots=True)
class TokenOptimizationAdvisoryIntegrationRequest:
    """Policy-gated advisory integration request with redaction-safe metadata only."""

    signal: TokenOptimizationAdvisorySignal
    policy: TokenOptimizationAdvisoryIntegrationPolicy = field(
        default_factory=TokenOptimizationAdvisoryIntegrationPolicy
    )
    request_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _validate_optional_request_id(self.request_id),
        )
        object.__setattr__(
            self,
            "metadata",
            _validate_string_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class TokenOptimizationAdvisoryIntegrationResult:
    """Policy-gated advisory integration outcome (non-auto-apply; redaction-safe)."""

    status: TokenOptimizationAdvisoryIntegrationStatus
    reason: TokenOptimizationAdvisoryGateReason
    source_type: TokenOptimizationSourceType
    recommendation: TokenOptimizationAdvisoryRecommendation | None = None
    request_id: str | None = None
    auto_apply_allowed: bool = False
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _validate_optional_request_id(self.request_id),
        )
        if self.auto_apply_allowed:
            raise ValueError("auto_apply_allowed must remain False")
        if self.raw_content_included:
            raise ValueError("raw_content_included must remain False")
        if (
            self.status is TokenOptimizationAdvisoryIntegrationStatus.BLOCKED_BY_POLICY
            and self.recommendation is not None
        ):
            raise ValueError("blocked results must not include a recommendation")
        if (
            self.status is TokenOptimizationAdvisoryIntegrationStatus.INSUFFICIENT_SIGNALS
            and self.recommendation is not None
            and self.recommendation.action
            is not TokenOptimizationRecommendationAction.INSUFFICIENT_DATA
        ):
            raise ValueError(
                "insufficient_signals results may include recommendations only for "
                "INSUFFICIENT_DATA actions"
            )
        if self.recommendation is not None:
            if self.recommendation.auto_apply_allowed:
                raise ValueError("recommendation.auto_apply_allowed must remain False")
            if self.recommendation.raw_content_included:
                raise ValueError("recommendation.raw_content_included must remain False")


def _blocked_result(
    *,
    reason: TokenOptimizationAdvisoryGateReason,
    source_type: TokenOptimizationSourceType,
    request_id: str | None,
) -> TokenOptimizationAdvisoryIntegrationResult:
    return TokenOptimizationAdvisoryIntegrationResult(
        status=TokenOptimizationAdvisoryIntegrationStatus.BLOCKED_BY_POLICY,
        reason=reason,
        source_type=source_type,
        recommendation=None,
        request_id=request_id,
        auto_apply_allowed=False,
        raw_content_included=False,
    )


def _result_with_recommendation(
    *,
    status: TokenOptimizationAdvisoryIntegrationStatus,
    reason: TokenOptimizationAdvisoryGateReason,
    recommendation: TokenOptimizationAdvisoryRecommendation,
    request_id: str | None,
) -> TokenOptimizationAdvisoryIntegrationResult:
    return TokenOptimizationAdvisoryIntegrationResult(
        status=status,
        reason=reason,
        source_type=recommendation.source_type,
        recommendation=recommendation,
        request_id=request_id,
        auto_apply_allowed=False,
        raw_content_included=False,
    )


def evaluate_policy_gated_advisory_request(
    request: TokenOptimizationAdvisoryIntegrationRequest,
) -> TokenOptimizationAdvisoryIntegrationResult:
    """Evaluate advisory integration under explicit policy (deterministic; no auto-apply)."""
    request_id = request.request_id
    source_type = request.signal.source_type

    if not request.policy.enabled:
        return _blocked_result(
            reason=TokenOptimizationAdvisoryGateReason.POLICY_DISABLED,
            source_type=source_type,
            request_id=request_id,
        )

    if request.policy.mode is TokenOptimizationAdvisoryIntegrationMode.DISABLED:
        return _blocked_result(
            reason=TokenOptimizationAdvisoryGateReason.MODE_DISABLED,
            source_type=source_type,
            request_id=request_id,
        )

    recommendation = recommend_token_optimization_action(request.signal)
    if recommendation.auto_apply_allowed:
        raise ValueError("recommendation.auto_apply_allowed must remain False")
    if recommendation.raw_content_included:
        raise ValueError("recommendation.raw_content_included must remain False")

    if recommendation.action is TokenOptimizationRecommendationAction.INSUFFICIENT_DATA:
        return _result_with_recommendation(
            status=TokenOptimizationAdvisoryIntegrationStatus.INSUFFICIENT_SIGNALS,
            reason=TokenOptimizationAdvisoryGateReason.INSUFFICIENT_SIGNALS,
            recommendation=recommendation,
            request_id=request_id,
        )

    if (
        recommendation.action is TokenOptimizationRecommendationAction.ENABLE_STRATEGY
        and not request.policy.allow_strategy_enable
    ):
        return _blocked_result(
            reason=TokenOptimizationAdvisoryGateReason.STRATEGY_ENABLE_NOT_ALLOWED,
            source_type=source_type,
            request_id=request_id,
        )

    if (
        recommendation.action is TokenOptimizationRecommendationAction.DISABLE_STRATEGY
        and not request.policy.allow_strategy_disable
    ):
        return _blocked_result(
            reason=TokenOptimizationAdvisoryGateReason.STRATEGY_DISABLE_NOT_ALLOWED,
            source_type=source_type,
            request_id=request_id,
        )

    if (
        request.policy.require_review_for_risky_recommendations
        and recommendation.action in _RISKY_RECOMMENDATION_ACTIONS
    ):
        return _result_with_recommendation(
            status=TokenOptimizationAdvisoryIntegrationStatus.REVIEW_REQUIRED,
            reason=TokenOptimizationAdvisoryGateReason.RISK_REQUIRES_REVIEW,
            recommendation=recommendation,
            request_id=request_id,
        )

    if recommendation.action is TokenOptimizationRecommendationAction.REQUIRE_MANUAL_REVIEW:
        return _result_with_recommendation(
            status=TokenOptimizationAdvisoryIntegrationStatus.REVIEW_REQUIRED,
            reason=TokenOptimizationAdvisoryGateReason.MANUAL_REVIEW_RECOMMENDED,
            recommendation=recommendation,
            request_id=request_id,
        )

    if request.policy.mode is TokenOptimizationAdvisoryIntegrationMode.REPORT_ONLY:
        return _result_with_recommendation(
            status=TokenOptimizationAdvisoryIntegrationStatus.REPORT_ONLY,
            reason=TokenOptimizationAdvisoryGateReason.REPORT_ONLY_MODE,
            recommendation=recommendation,
            request_id=request_id,
        )

    if request.policy.mode is TokenOptimizationAdvisoryIntegrationMode.DRY_RUN:
        return _result_with_recommendation(
            status=TokenOptimizationAdvisoryIntegrationStatus.DRY_RUN,
            reason=TokenOptimizationAdvisoryGateReason.DRY_RUN_MODE,
            recommendation=recommendation,
            request_id=request_id,
        )

    if request.policy.mode is TokenOptimizationAdvisoryIntegrationMode.REVIEW_ONLY:
        return _result_with_recommendation(
            status=TokenOptimizationAdvisoryIntegrationStatus.REVIEW_REQUIRED,
            reason=TokenOptimizationAdvisoryGateReason.REVIEW_ONLY_MODE,
            recommendation=recommendation,
            request_id=request_id,
        )

    if request.policy.mode is TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED:
        return _result_with_recommendation(
            status=TokenOptimizationAdvisoryIntegrationStatus.RECOMMENDATION_READY,
            reason=TokenOptimizationAdvisoryGateReason.ADVISORY_ALLOWED,
            recommendation=recommendation,
            request_id=request_id,
        )

    return _blocked_result(
        reason=TokenOptimizationAdvisoryGateReason.AUTO_APPLY_FORBIDDEN,
        source_type=source_type,
        request_id=request_id,
    )


def token_optimization_advisory_integration_result_to_dict(
    result: TokenOptimizationAdvisoryIntegrationResult,
) -> dict[str, object]:
    """Serialize an integration result for JSON output (redaction-safe scalars only)."""
    payload: dict[str, object] = {
        "status": _enum_value(result.status),
        "reason": _enum_value(result.reason),
        "source_type": _enum_value(result.source_type),
        "request_id": result.request_id,
        "auto_apply_allowed": result.auto_apply_allowed,
        "raw_content_included": result.raw_content_included,
    }
    if result.recommendation is not None:
        recommendation = result.recommendation
        payload["recommendation_action"] = _enum_value(recommendation.action)
        payload["recommendation_reason"] = _enum_value(recommendation.reason)
        payload["recommendation_confidence"] = _enum_value(recommendation.confidence)
        payload["recommendation_strategy_kind"] = _enum_value(recommendation.strategy_kind)
    return payload


def format_token_optimization_advisory_integration_result(
    result: TokenOptimizationAdvisoryIntegrationResult,
) -> str:
    """Human-readable integration result (deterministic, redaction-safe)."""
    lines = [
        f"status={_enum_value(result.status)}",
        f"reason={_enum_value(result.reason)}",
        f"source_type={_enum_value(result.source_type)}",
        f"request_id={result.request_id}",
        f"auto_apply_allowed={result.auto_apply_allowed}",
        f"raw_content_included={result.raw_content_included}",
    ]
    if result.recommendation is not None:
        recommendation = result.recommendation
        lines.extend(
            [
                f"recommendation_action={_enum_value(recommendation.action)}",
                f"recommendation_reason={_enum_value(recommendation.reason)}",
                f"recommendation_confidence={_enum_value(recommendation.confidence)}",
                (
                    "recommendation_strategy_kind="
                    f"{_enum_value(recommendation.strategy_kind)}"
                ),
            ]
        )
    return "\n".join(lines)
