# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Iterable, List

from intergrax.runtime.nexus.artifacts.models import Artifact
from intergrax.runtime.nexus.artifacts.store_base import ArtifactStore


class InMemoryArtifactStore(ArtifactStore):
    """
    In-memory artifact store for tests and local development.

    Contract notes:
    - get() raises KeyError if artifact_id not found (consistent with InMemoryRunStore).
    - delete_for_run() tolerates unknown run_id.
    """

    def __init__(self) -> None:
        self._artifacts: Dict[str, Artifact] = {}
        self._by_run: Dict[str, List[str]] = {}

    def put(self, artifact: Artifact) -> None:
        self._artifacts[artifact.artifact_id] = artifact
        self._by_run.setdefault(artifact.run_id, []).append(artifact.artifact_id)

    def get(self, artifact_id: str) -> Artifact:
        return self._artifacts[artifact_id]

    def list_for_run(self, run_id: str) -> Iterable[Artifact]:
        ids = self._by_run.get(run_id, [])
        for artifact_id in ids:
            yield self._artifacts[artifact_id]

    def delete_for_run(self, run_id: str) -> None:
        ids = self._by_run.pop(run_id, [])
        for artifact_id in ids:
            self._artifacts.pop(artifact_id, None)
