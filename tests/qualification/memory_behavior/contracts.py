# © Artur Czarnecki. All rights reserved.

"""Typed contracts for MEM-FINAL-AUDIT-6 behavioral evaluation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class BehaviorScenarioCategory(str, Enum):
    USER = "USER"
    SESSION = "SESSION"
    TASK = "TASK"
    PROJECTION_LIFECYCLE = "PROJECTION_LIFECYCLE"
    SECURITY = "SECURITY"
    METRICS = "METRICS"
    OBSERVABILITY = "OBSERVABILITY"


class BehaviorGateKind(str, Enum):
    HARD = "HARD"
    QUALITY = "QUALITY"
    DOCUMENTATION = "DOCUMENTATION"


@dataclass(frozen=True, slots=True)
class BehaviorViolationCounters:
    cross_tenant_leaks: int = 0
    cross_user_leaks: int = 0
    deleted_resurrections: int = 0
    superseded_as_current: int = 0
    projection_only_ghosts: int = 0
    identity_authority_violations: int = 0

    @property
    def has_hard_violation(self) -> bool:
        return any(
            (
                self.cross_tenant_leaks,
                self.cross_user_leaks,
                self.deleted_resurrections,
                self.superseded_as_current,
                self.projection_only_ghosts,
                self.identity_authority_violations,
            )
        )


@dataclass(frozen=True, slots=True)
class SemanticQualityMetrics:
    dataset_size: int
    hit_at_1: float
    recall_at_k: float
    mrr: float
    top_k: int


@dataclass(frozen=True, slots=True)
class MemoryBehaviorEvaluationSummary:
    scenario_count: int
    hard_passed: int
    hard_failed: int
    violations: BehaviorViolationCounters
    semantic_metrics: SemanticQualityMetrics | None = None


class BehaviorAssertion(Protocol):
    scenario_id: str
    category: BehaviorScenarioCategory
    gate: BehaviorGateKind

    async def run(self) -> None: ...


@dataclass
class BehaviorViolationLedger:
    """Session-scoped accumulator for zero-violation metrics (tests reset via fixture)."""

    counters: BehaviorViolationCounters = field(default_factory=BehaviorViolationCounters)

    def record_cross_user_leak(self, count: int = 1) -> None:
        c = self.counters
        self.counters = BehaviorViolationCounters(
            cross_tenant_leaks=c.cross_tenant_leaks,
            cross_user_leaks=c.cross_user_leaks + count,
            deleted_resurrections=c.deleted_resurrections,
            superseded_as_current=c.superseded_as_current,
            projection_only_ghosts=c.projection_only_ghosts,
            identity_authority_violations=c.identity_authority_violations,
        )

    def record_cross_tenant_leak(self, count: int = 1) -> None:
        c = self.counters
        self.counters = BehaviorViolationCounters(
            cross_tenant_leaks=c.cross_tenant_leaks + count,
            cross_user_leaks=c.cross_user_leaks,
            deleted_resurrections=c.deleted_resurrections,
            superseded_as_current=c.superseded_as_current,
            projection_only_ghosts=c.projection_only_ghosts,
            identity_authority_violations=c.identity_authority_violations,
        )

    def record_deleted_resurrection(self, count: int = 1) -> None:
        c = self.counters
        self.counters = BehaviorViolationCounters(
            cross_tenant_leaks=c.cross_tenant_leaks,
            cross_user_leaks=c.cross_user_leaks,
            deleted_resurrections=c.deleted_resurrections + count,
            superseded_as_current=c.superseded_as_current,
            projection_only_ghosts=c.projection_only_ghosts,
            identity_authority_violations=c.identity_authority_violations,
        )

    def record_superseded_as_current(self, count: int = 1) -> None:
        c = self.counters
        self.counters = BehaviorViolationCounters(
            cross_tenant_leaks=c.cross_tenant_leaks,
            cross_user_leaks=c.cross_user_leaks,
            deleted_resurrections=c.deleted_resurrections,
            superseded_as_current=c.superseded_as_current + count,
            projection_only_ghosts=c.projection_only_ghosts,
            identity_authority_violations=c.identity_authority_violations,
        )

    def record_projection_ghost(self, count: int = 1) -> None:
        c = self.counters
        self.counters = BehaviorViolationCounters(
            cross_tenant_leaks=c.cross_tenant_leaks,
            cross_user_leaks=c.cross_user_leaks,
            deleted_resurrections=c.deleted_resurrections,
            superseded_as_current=c.superseded_as_current,
            projection_only_ghosts=c.projection_only_ghosts + count,
            identity_authority_violations=c.identity_authority_violations,
        )

    def record_identity_violation(self, count: int = 1) -> None:
        c = self.counters
        self.counters = BehaviorViolationCounters(
            cross_tenant_leaks=c.cross_tenant_leaks,
            cross_user_leaks=c.cross_user_leaks,
            deleted_resurrections=c.deleted_resurrections,
            superseded_as_current=c.superseded_as_current,
            projection_only_ghosts=c.projection_only_ghosts,
            identity_authority_violations=c.identity_authority_violations + count,
        )


@dataclass
class BehaviorEvalContext:
    """Per-run evaluation context; owns the violation ledger for scenario aggregation."""

    ledger: BehaviorViolationLedger = field(default_factory=BehaviorViolationLedger)


@dataclass(frozen=True, slots=True)
class BehaviorEvalCase:
    scenario_id: str
    category: BehaviorScenarioCategory
    gate: BehaviorGateKind
    runner: Callable[[BehaviorEvalContext], Awaitable[None]]
