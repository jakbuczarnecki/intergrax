# © Artur Czarnecki. All rights reserved.

"""Synthetic corpus for advisory policy presets and resolver (TOKEN-7D)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.token_optimization.advisory_integration import (
    TokenOptimizationAdvisoryIntegrationMode,
)
from intergrax.runtime.token_optimization.advisory_policy import (
    TokenOptimizationAdvisoryPolicyOverrides,
    TokenOptimizationAdvisoryPolicyPreset,
)

ADVISORY_POLICY_SYNTHETIC_CORPUS_MARKER = "SYNTHETIC_ADVISORY_POLICY_CORPUS_V1"


@dataclass(frozen=True, slots=True)
class AdvisoryPolicyCorpusCase:
    case_id: str
    title: str
    preset: TokenOptimizationAdvisoryPolicyPreset
    overrides: TokenOptimizationAdvisoryPolicyOverrides | None
    expected_enabled: bool
    expected_mode: TokenOptimizationAdvisoryIntegrationMode
    expected_allow_strategy_enable: bool
    expected_allow_strategy_disable: bool
    expected_require_review_for_risky_recommendations: bool
    expected_overrides_applied: tuple[str, ...] = ()
    synthetic_marker: str = ADVISORY_POLICY_SYNTHETIC_CORPUS_MARKER


ADVISORY_POLICY_CORPUS: tuple[AdvisoryPolicyCorpusCase, ...] = (
    AdvisoryPolicyCorpusCase(
        case_id="advisory_policy.disabled",
        title="Disabled preset resolves to disabled policy",
        preset=TokenOptimizationAdvisoryPolicyPreset.DISABLED,
        overrides=None,
        expected_enabled=False,
        expected_mode=TokenOptimizationAdvisoryIntegrationMode.DISABLED,
        expected_allow_strategy_enable=False,
        expected_allow_strategy_disable=True,
        expected_require_review_for_risky_recommendations=True,
    ),
    AdvisoryPolicyCorpusCase(
        case_id="advisory_policy.report_only",
        title="Report-only preset resolves to report-only policy",
        preset=TokenOptimizationAdvisoryPolicyPreset.REPORT_ONLY,
        overrides=None,
        expected_enabled=True,
        expected_mode=TokenOptimizationAdvisoryIntegrationMode.REPORT_ONLY,
        expected_allow_strategy_enable=False,
        expected_allow_strategy_disable=True,
        expected_require_review_for_risky_recommendations=True,
    ),
    AdvisoryPolicyCorpusCase(
        case_id="advisory_policy.dry_run_safe",
        title="Dry-run-safe preset resolves to dry-run policy",
        preset=TokenOptimizationAdvisoryPolicyPreset.DRY_RUN_SAFE,
        overrides=None,
        expected_enabled=True,
        expected_mode=TokenOptimizationAdvisoryIntegrationMode.DRY_RUN,
        expected_allow_strategy_enable=False,
        expected_allow_strategy_disable=True,
        expected_require_review_for_risky_recommendations=True,
    ),
    AdvisoryPolicyCorpusCase(
        case_id="advisory_policy.review_first",
        title="Review-first preset resolves to review-only policy",
        preset=TokenOptimizationAdvisoryPolicyPreset.REVIEW_FIRST,
        overrides=None,
        expected_enabled=True,
        expected_mode=TokenOptimizationAdvisoryIntegrationMode.REVIEW_ONLY,
        expected_allow_strategy_enable=False,
        expected_allow_strategy_disable=True,
        expected_require_review_for_risky_recommendations=True,
    ),
    AdvisoryPolicyCorpusCase(
        case_id="advisory_policy.advisory_allowed_safe",
        title="Advisory-allowed-safe preset resolves to advisory-allowed policy",
        preset=TokenOptimizationAdvisoryPolicyPreset.ADVISORY_ALLOWED_SAFE,
        overrides=None,
        expected_enabled=True,
        expected_mode=TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED,
        expected_allow_strategy_enable=False,
        expected_allow_strategy_disable=True,
        expected_require_review_for_risky_recommendations=True,
    ),
    AdvisoryPolicyCorpusCase(
        case_id="advisory_policy.override_allows_strategy_enable",
        title="Override allows strategy enable on advisory-allowed-safe preset",
        preset=TokenOptimizationAdvisoryPolicyPreset.ADVISORY_ALLOWED_SAFE,
        overrides=TokenOptimizationAdvisoryPolicyOverrides(allow_strategy_enable=True),
        expected_enabled=True,
        expected_mode=TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED,
        expected_allow_strategy_enable=True,
        expected_allow_strategy_disable=True,
        expected_require_review_for_risky_recommendations=True,
        expected_overrides_applied=("allow_strategy_enable",),
    ),
    AdvisoryPolicyCorpusCase(
        case_id="advisory_policy.override_blocks_strategy_disable",
        title="Override blocks strategy disable on dry-run-safe preset",
        preset=TokenOptimizationAdvisoryPolicyPreset.DRY_RUN_SAFE,
        overrides=TokenOptimizationAdvisoryPolicyOverrides(allow_strategy_disable=False),
        expected_enabled=True,
        expected_mode=TokenOptimizationAdvisoryIntegrationMode.DRY_RUN,
        expected_allow_strategy_enable=False,
        expected_allow_strategy_disable=False,
        expected_require_review_for_risky_recommendations=True,
        expected_overrides_applied=("allow_strategy_disable",),
    ),
    AdvisoryPolicyCorpusCase(
        case_id="advisory_policy.override_disables_risky_review",
        title="Override disables risky review requirement on review-first preset",
        preset=TokenOptimizationAdvisoryPolicyPreset.REVIEW_FIRST,
        overrides=TokenOptimizationAdvisoryPolicyOverrides(
            require_review_for_risky_recommendations=False,
        ),
        expected_enabled=True,
        expected_mode=TokenOptimizationAdvisoryIntegrationMode.REVIEW_ONLY,
        expected_allow_strategy_enable=False,
        expected_allow_strategy_disable=True,
        expected_require_review_for_risky_recommendations=False,
        expected_overrides_applied=("require_review_for_risky_recommendations",),
    ),
)
