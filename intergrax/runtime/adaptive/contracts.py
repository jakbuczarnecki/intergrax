# © Artur Czarnecki. All rights reserved.

"""Adaptive Harness Intelligence runtime contracts (Phase W-ADAPT-0.3, W-ADAPT-1.1–1.2)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

ADAPTIVE_PACKAGE_SCHEMA_VERSION = "1.0.0"


class AdaptiveLifecycleMode(str, Enum):
    """L4 runtime lifecycle modes (AHIA §12.1)."""

    OBSERVE = "l4_o"
    RECOMMEND = "l4_r"
    SHADOW = "l4_s"
    CANARY = "l4_c"
    APPLY = "l4_a"
    VERIFY = "l4_v"


class ProfileVersionStatus(str, Enum):
    """Profile version promotion states (AHIA §9.8, §12.2)."""

    DRAFT = "draft"
    SHADOW = "shadow"
    CANARY = "canary"
    ACTIVE = "active"
    RETIRED = "retired"


class ProfileArtifactType(str, Enum):
    """Versioned harness profile artifact kinds (AHIA §9.8)."""

    ORCHESTRATION = "orchestration"
    RAG = "rag"
    LLM_ROUTING = "llm_routing"
    POLICY_FRAGMENT = "policy_fragment"


class OutcomeEvalMode(str, Enum):
    """Evaluation mode recorded on a harness outcome signal."""

    OFFLINE = "offline"
    ONLINE = "online"
    SHADOW = "shadow"
    HUMAN = "human"


class ProcessPatternAction(str, Enum):
    """Suggested follow-up for a mined process pattern (AHIA §13)."""

    CREATE_SKILL_DRAFT = "create_skill_draft"
    TUNE_ROUTING = "tune_routing"
    DOCUMENT_RUNBOOK = "document_runbook"


class UtilityWeights(BaseModel):
    """Configurable utility weights (AHIA §10.2)."""

    model_config = ConfigDict(extra="forbid")

    w_quality: float = Field(default=0.50, ge=0.0, le=1.0)
    w_cost: float = Field(default=0.25, ge=0.0, le=1.0)
    w_latency: float = Field(default=0.10, ge=0.0, le=1.0)
    w_hitl: float = Field(default=0.10, ge=0.0, le=1.0)
    w_regression: float = Field(default=0.05, ge=0.0, le=1.0)
    w_business: float = Field(default=0.00, ge=0.0, le=1.0)
    latency_slo_ms: int = Field(default=30_000, ge=1)
    max_hitl_interventions: int = Field(default=3, ge=1)


class HarnessOutcomeSignal(BaseModel):
    """Normalized post-run observation for the adaptation engine (AHIA §10.1)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = ADAPTIVE_PACKAGE_SCHEMA_VERSION
    signal_id: str = Field(default_factory=lambda: f"sig_{uuid4().hex}")
    run_id: str
    tenant_id: str
    application_id: str
    agent_id: str
    task_class: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    validation_passed: bool = True
    eval_mode: OutcomeEvalMode = OutcomeEvalMode.OFFLINE

    cost_normalized: float = Field(default=0.0, ge=0.0)
    latency_ms: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    step_count: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)

    hitl_interventions: int = Field(default=0, ge=0)
    regression_flags: list[str] = Field(default_factory=list)

    business_outcome: float | None = None
    utility: float | None = None


class ProfileVersionDraft(BaseModel):
    """Draft payload attached to an adaptive loop proposal (AHIA §9.3)."""

    model_config = ConfigDict(extra="forbid")

    version_id: str
    artifact_type: ProfileArtifactType
    artifact_payload: dict[str, Any] = Field(default_factory=dict)
    parent_version_id: str | None = None
    created_by: str = "adaptation_engine"
    status: ProfileVersionStatus = ProfileVersionStatus.DRAFT

    @field_validator("version_id")
    @classmethod
    def _validate_version_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("version_id must not be empty")
        return normalized


class ProfileVersionRecord(BaseModel):
    """Immutable stored profile version (AHIA §9.8)."""

    model_config = ConfigDict(extra="forbid")

    version_id: str
    tenant_id: str
    task_class: str = ""
    artifact_type: ProfileArtifactType
    artifact_payload: dict[str, Any] = Field(default_factory=dict)
    parent_version_id: str | None = None
    created_by: str
    rollback_of: str | None = None
    status: ProfileVersionStatus = ProfileVersionStatus.DRAFT
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProcessPatternProposal(BaseModel):
    """Mined operational path recommendation (AHIA §13)."""

    model_config = ConfigDict(extra="forbid")

    pattern_id: str = Field(default_factory=lambda: f"pat_{uuid4().hex[:12]}")
    description: str
    suggested_action: ProcessPatternAction
    evidence_run_ids: list[str] = Field(default_factory=list)
    utility_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("description must not be empty")
        return normalized
