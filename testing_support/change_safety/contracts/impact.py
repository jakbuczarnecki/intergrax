# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from testing_support.change_safety.contracts.enums import ImpactStatus


class AffectedNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    kind: str = Field(description="DIRECT | TRANSITIVE | CONSERVATIVE_UNKNOWN")
    witness_path: tuple[str, ...] = ()
    rule_id: str | None = None


class ImpactAssessmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "evol_impact_report.v1"
    source_commit_sha: str
    graph_digest: str
    inventory_digest: str
    ruleset_version: str
    status: ImpactStatus
    affected: tuple[AffectedNode, ...] = ()
