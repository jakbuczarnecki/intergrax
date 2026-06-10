# © Artur Czarnecki. All rights reserved.

"""Import LangGraph-compatible skill pack definitions into ``SkillManifest`` (AUDIT-IDEAL-12.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier


class LangGraphSkillImportError(ValueError):
    """Raised when a LangGraph pack cannot be converted to ``SkillManifest``."""


class LangGraphNodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str


class LangGraphEdgeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str


class LangGraphGraphSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[LangGraphNodeSpec] = Field(default_factory=list)
    edges: list[LangGraphEdgeSpec] = Field(default_factory=list)


class LangGraphSkillPackSpec(BaseModel):
    """Minimal LangGraph pack surface mapped to Intergrax ``SkillManifest``."""

    model_config = ConfigDict(extra="forbid")

    skill_id: str
    description: str
    version: str = "1.0.0"
    tools: tuple[str, ...] = ()
    graph: LangGraphGraphSpec = Field(default_factory=LangGraphGraphSpec)
    risk_tier: SkillRiskTier = SkillRiskTier.LOW


class LangGraphSkillPackImporter:
    """Parse LangGraph-style JSON packs into registry-ready manifests."""

    def import_file(self, path: Path) -> SkillManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self.import_payload(payload)

    def import_payload(self, payload: dict[str, Any]) -> SkillManifest:
        try:
            spec = LangGraphSkillPackSpec.model_validate(payload)
        except ValidationError as exc:
            raise LangGraphSkillImportError(str(exc)) from exc
        if not spec.graph.nodes:
            raise LangGraphSkillImportError("graph.nodes must be non-empty")
        return SkillManifest(
            skill_id=spec.skill_id,
            version=spec.version,
            description=spec.description,
            tool_ids=spec.tools,
            risk_tier=spec.risk_tier,
            tags=("langgraph_pack",),
        )
