# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable

from intergrax.runtime.replay.contracts.artifact_dto import ArtifactDTO


class ReplayArtifactStore(ABC):
    """Read-only access to artifacts metadata."""

    @abstractmethod
    def list_for_run(self, tenant_id: str, run_id: str) -> Iterable[ArtifactDTO]:
        ...
