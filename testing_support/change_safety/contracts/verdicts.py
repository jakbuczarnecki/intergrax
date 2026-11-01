# © Artur Czarnecki. All rights reserved.

"""W4 verdict / dossier artifacts (Pydantic ``extra=forbid``; no approve/authorize/execute fields)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from testing_support.change_safety.contracts.enums import (
    DeployRefusalReason,
    DeploySafetyVerdict,
    WorkflowEvolutionDecision,
)


class WorkflowEvolutionVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "evol_workflow_verdict.v1"
    matrix_id: str
    decision: WorkflowEvolutionDecision
    witness: str = ""


class DeployDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: str


class DeploySafetyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "evol_deploy_assessment.v1"
    matrix_id: str
    verdict: DeploySafetyVerdict
    reason: DeployRefusalReason | None = None
    dimensions: tuple[DeployDimension, ...] = ()


class DossierAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    answer: str
    evidence_refs: tuple[str, ...] = ()


class ChangeDossier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "evol_change_dossier.v1"
    created_at_utc: str
    graph_digest: str
    inventory_digest: str
    stale: bool = False
    stale_triggers: tuple[str, ...] = ()
    sections: tuple[DossierAnswer, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    content_digest: str = Field(default="", description="sha256 over sections (excludes created_at_utc)")


class AcceptedBaselineManifest(BaseModel):
    """Digests accepted at qualification time; freshness is judged against these (T1..T9)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "evol_baseline_manifest.v1"
    accepted_at_sha: str
    digests: tuple[tuple[str, str], ...]  # (digest_key, value)


__all__ = [
    "AcceptedBaselineManifest",
    "ChangeDossier",
    "DeployDimension",
    "DeploySafetyAssessment",
    "DossierAnswer",
    "WorkflowEvolutionVerdict",
]
