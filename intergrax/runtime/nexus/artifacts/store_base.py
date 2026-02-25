# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from intergrax.runtime.nexus.artifacts.models import Artifact


class ArtifactStore(ABC):
    """
    Storage port for execution artifacts.

    Implementations MUST:
    - persist artifacts durably (or in-memory for tests), 
    - allow listing artifacts for a run,
    - allow fetching a single artifact by id.

    Implementations MUST NOT:
    - embed artifacts into trace (trace stores ArtifactRef only),
    - implement replay logic (separate step),
    - implement security/scopes (separate step).
    """

    @abstractmethod
    def put(self, artifact: Artifact) -> None:
        ...

    @abstractmethod
    def get(self, tenant_id: str, artifact_id: str) -> Artifact:
        ...

    @abstractmethod
    def list_for_run(self, tenant_id: str, run_id: str) -> Iterable[Artifact]:
        ...

    @abstractmethod
    def delete_for_run(self, tenant_id: str, run_id: str) -> None:
        ...
