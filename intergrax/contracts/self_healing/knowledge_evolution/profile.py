# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Derived strategy knowledge models — not source of truth (SELF-HEALING R5.4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from intergrax.contracts.self_healing.knowledge_evolution.confidence import StrategyKnowledgeConfidenceLevel


def mint_strategy_knowledge_profile_id() -> str:
    return f"sh_skp_{uuid4().hex}"


def mint_strategy_knowledge_revision_id() -> str:
    return f"sh_skr_{uuid4().hex}"


class StrategyKnowledgeEvolutionTrigger(StrEnum):
    WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"
    SCHEDULED_REBUILD = "SCHEDULED_REBUILD"
    OPERATOR_REQUEST = "OPERATOR_REQUEST"
    BACKFILL = "BACKFILL"


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeObservationSummary:
    """Aggregated references to R5.1 observations — not a copy of all experience rows."""

    experience_count: int
    experience_id_sample: tuple[str, ...]
    earliest_recorded_at: datetime | None
    latest_recorded_at: datetime | None
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.experience_count < 0:
            raise ValueError("experience_count must be >= 0")
        if self.experience_count > 0 and not self.experience_id_sample:
            raise ValueError("experience_id_sample required when experience_count > 0")


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeQualitySnapshot:
    """Point-in-time quality fields captured at revision — descriptive only."""

    execution_count: int
    success_ratio: float
    average_recovery_time_seconds: float | None
    quality_score: float
    evaluator_id: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.execution_count < 0:
            raise ValueError("execution_count must be >= 0")
        if not (0.0 <= self.success_ratio <= 1.0):
            raise ValueError("success_ratio must be in [0.0, 1.0]")
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError("quality_score must be in [0.0, 1.0]")
        if not self.evaluator_id.strip():
            raise ValueError("evaluator_id required")


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeFreshness:
    last_evidence_at: datetime | None
    staleness_policy_id: str | None
    ttl_hint_seconds: int | None

    def __post_init__(self) -> None:
        if self.ttl_hint_seconds is not None and self.ttl_hint_seconds < 0:
            raise ValueError("ttl_hint_seconds must be >= 0 when set")


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeProfile:
    """
    Derived knowledge for one strategy in a scoped context.

    Must not contain execution directives, selector weights, or lifecycle actions.
    """

    profile_id: str
    tenant_id: str
    strategy_id: str
    context_fingerprint: str
    context_refs: tuple[str, ...]
    observation_summary: StrategyKnowledgeObservationSummary
    quality_snapshot: StrategyKnowledgeQualitySnapshot | None
    confidence_label: StrategyKnowledgeConfidenceLevel
    freshness: StrategyKnowledgeFreshness
    knowledge_version: int
    supersedes_version: int | None
    derived_at: datetime
    learning_engine_id: str
    input_experience_fingerprint: str

    def __post_init__(self) -> None:
        if not self.profile_id.startswith("sh_skp_"):
            raise ValueError("profile_id must be sh_skp_*")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.context_fingerprint.strip():
            raise ValueError("context_fingerprint required")
        if not self.context_refs:
            raise ValueError("context_refs must be non-empty")
        if self.knowledge_version < 1:
            raise ValueError("knowledge_version must be >= 1")
        if self.supersedes_version is not None and self.supersedes_version >= self.knowledge_version:
            raise ValueError("supersedes_version must be < knowledge_version when set")
        if not self.learning_engine_id.strip():
            raise ValueError("learning_engine_id required")
        if not self.input_experience_fingerprint.strip():
            raise ValueError("input_experience_fingerprint required")


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeContext:
    """Scoped key for evolution and lookup."""

    tenant_id: str
    strategy_id: str
    context_fingerprint: str
    context_refs: tuple[str, ...]
    diagnostic_investigation_id: str | None = None
    time_horizon_experience_limit: int | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.context_fingerprint.strip():
            raise ValueError("context_fingerprint required")
        if not self.context_refs:
            raise ValueError("context_refs must be non-empty")
        if self.time_horizon_experience_limit is not None and self.time_horizon_experience_limit < 1:
            raise ValueError("time_horizon_experience_limit must be >= 1 when set")


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeRevision:
    """Immutable audit bundle appended on each successful evolution."""

    revision_id: str
    profile: StrategyKnowledgeProfile
    change_summary: str
    trigger: StrategyKnowledgeEvolutionTrigger
    trigger_refs: tuple[str, ...]
    input_experience_ids: tuple[str, ...]
    input_assessment_refs: tuple[str, ...]
    metric_snapshot_refs: tuple[str, ...]
    comparison_policy_id: str | None
    previous_knowledge_version: int | None

    def __post_init__(self) -> None:
        if not self.revision_id.startswith("sh_skr_"):
            raise ValueError("revision_id must be sh_skr_*")
        if not self.change_summary.strip():
            raise ValueError("change_summary required")
        if not self.trigger_refs:
            raise ValueError("trigger_refs must be non-empty")


__all__ = [
    "StrategyKnowledgeContext",
    "StrategyKnowledgeEvolutionTrigger",
    "StrategyKnowledgeFreshness",
    "StrategyKnowledgeObservationSummary",
    "StrategyKnowledgeProfile",
    "StrategyKnowledgeQualitySnapshot",
    "StrategyKnowledgeRevision",
    "mint_strategy_knowledge_profile_id",
    "mint_strategy_knowledge_revision_id",
]
