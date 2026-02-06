# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Artifact:
    """
    Persistent artifact produced during run execution.

    Artifacts are part of execution substrate (debug / replay / eval),
    not user/profile memory.
    """

    artifact_id: str
    run_id: str
    step_id: Optional[str]

    kind: str
    mime_type: str

    created_at_utc: datetime

    data: bytes
    size_bytes: int


@dataclass(frozen=True)
class ArtifactRef:
    """
    Lightweight reference that can be embedded in trace events.

    Trace MUST remain small; artifacts are retrieved via ArtifactStore.
    """

    artifact_id: str
    kind: str
    size_bytes: int
