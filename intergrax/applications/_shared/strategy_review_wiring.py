# © Artur Czarnecki. All rights reserved.

"""Quarterly strategy review wiring (AUDIT-IDEAL-1.1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.architecture.strategy_review import (
    QuarterlyStrategyReviewReport,
    build_quarterly_strategy_review,
)


@dataclass(frozen=True, slots=True)
class StrategyReviewWiring:
    enabled: bool
    report: QuarterlyStrategyReviewReport | None


def resolve_strategy_review_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    repo_root: Path,
) -> StrategyReviewWiring:
    """Validate quarterly strategy review artifacts for product hosts."""
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return StrategyReviewWiring(enabled=False, report=None)
    if not env.governance_profile.quarterly_strategy_review_enabled:
        return StrategyReviewWiring(enabled=False, report=None)
    report = build_quarterly_strategy_review(repo_root)
    return StrategyReviewWiring(enabled=True, report=report)
