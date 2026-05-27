# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shadow workspace contracts (architecture §20)."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class ShadowArtifact(BaseModel):
    artifact_id: str
    relative_path: str
    size_bytes: int
    content_type: str = "text/plain"
    sha256: str = ""


class ShadowSnapshot(BaseModel):
    workspace_id: str
    created_at_utc: str
    files: Dict[str, str] = Field(default_factory=dict)


class ShadowWorkspaceManifest(BaseModel):
    workspace_id: str
    tenant_id: str
    task_id: str
    root_path: str
    created_at_utc: str
    artifact_count: int
    artifacts: List[ShadowArtifact] = Field(default_factory=list)
