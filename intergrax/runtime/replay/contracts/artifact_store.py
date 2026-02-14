# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from typing import Protocol, Iterable

from intergrax.runtime.replay.contracts.artifact_dto import ArtifactDTO


class ArtifactStore(Protocol):
    """Read-only access to artifacts metadata."""

    def list_for_run(self, run_id: str) -> Iterable[ArtifactDTO]:
        ...
