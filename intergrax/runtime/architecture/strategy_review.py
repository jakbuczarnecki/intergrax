# © Artur Czarnecki. All rights reserved.

"""Quarterly harness strategy review contracts (AUDIT-IDEAL-1.1)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class StrategyReviewDocument(BaseModel):
    relative_path: str
    present: bool


class QuarterlyStrategyReviewReport(BaseModel):
    schema_version: str = "1.0.0"
    quarter: str
    documents: list[StrategyReviewDocument] = Field(default_factory=list)
    ready: bool


_REQUIRED_DOCS: tuple[str, ...] = (
    "docs/project/maintainers/plans/IDEAL_HARNESS_L3.md",
    "docs/project/technical/guides/ARCHITECTURE_DEBT_REGISTER.md",
    "docs/project/maintainers/plans/AUDIT_IDEAL_2026.md",
    "docs/project/technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md",
)


def build_quarterly_strategy_review(
    repo_root: Path,
    *,
    quarter: str = "2026-Q2",
) -> QuarterlyStrategyReviewReport:
    """Validate quarterly strategy review inputs are present in the repository."""
    documents = [
        StrategyReviewDocument(
            relative_path=relative_path,
            present=(repo_root / relative_path).is_file(),
        )
        for relative_path in _REQUIRED_DOCS
    ]
    ready = all(document.present for document in documents)
    return QuarterlyStrategyReviewReport(quarter=quarter, documents=documents, ready=ready)
