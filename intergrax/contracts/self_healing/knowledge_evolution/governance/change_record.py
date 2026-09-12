# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Single knowledge change description — audit-only (SELF-HEALING R5.6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import uuid4


def mint_strategy_knowledge_change_id() -> str:
    return f"sh_skc_{uuid4().hex}"


class StrategyKnowledgeChangeSource(StrEnum):
    """Origin of a knowledge change — maps from evolution triggers, not execution."""

    WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"
    SCHEDULED_REBUILD = "SCHEDULED_REBUILD"
    OPERATOR_REQUEST = "OPERATOR_REQUEST"
    BACKFILL = "BACKFILL"
    LEARNING_ENGINE = "LEARNING_ENGINE"


class StrategyKnowledgeChangeType(StrEnum):
    INITIAL_PROFILE = "INITIAL_PROFILE"
    VERSION_INCREMENT = "VERSION_INCREMENT"


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeChangeRecord:
    """
    Immutable description of one knowledge transition N → N+1.

    Must not contain execution directives, decisions, or lifecycle actions.
    """

    change_id: str
    tenant_id: str
    strategy_id: str
    context_fingerprint: str
    revision_id: str
    previous_knowledge_version: int | None
    new_knowledge_version: int
    change_source: StrategyKnowledgeChangeSource
    change_type: StrategyKnowledgeChangeType
    evolution_mechanism_id: str
    source_experience_refs: tuple[str, ...]
    rationale: str
    recorded_at: datetime

    def __post_init__(self) -> None:
        if not self.change_id.startswith("sh_skc_"):
            raise ValueError("change_id must be sh_skc_*")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.context_fingerprint.strip():
            raise ValueError("context_fingerprint required")
        if not self.revision_id.startswith("sh_skr_"):
            raise ValueError("revision_id must be sh_skr_*")
        if self.new_knowledge_version < 1:
            raise ValueError("new_knowledge_version must be >= 1")
        if self.previous_knowledge_version is not None and self.previous_knowledge_version >= self.new_knowledge_version:
            raise ValueError("previous_knowledge_version must be < new_knowledge_version when set")
        if not self.evolution_mechanism_id.strip():
            raise ValueError("evolution_mechanism_id required")
        if not self.rationale.strip():
            raise ValueError("rationale required")


__all__ = [
    "StrategyKnowledgeChangeRecord",
    "StrategyKnowledgeChangeSource",
    "StrategyKnowledgeChangeType",
    "mint_strategy_knowledge_change_id",
]
