# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from intergrax.runtime.nexus.artifacts.models import Artifact
from intergrax.runtime.nexus.artifacts.store_base import ArtifactStore


class InMemoryArtifactStore(ArtifactStore):
    """
    In-memory artifact store for tests and local development.

    Security:
    - Fully tenant-scoped storage.
    - No global artifact namespace.
    """

    def __init__(self) -> None:
        self._artifacts: Dict[Tuple[str, str], Artifact] = {}
        self._by_run: Dict[Tuple[str, str], List[str]] = {}

    def put(self, artifact: Artifact) -> None:
        key = (artifact.tenant_id, artifact.artifact_id)
        self._artifacts[key] = artifact

        run_key = (artifact.tenant_id, artifact.run_id)
        self._by_run.setdefault(run_key, []).append(artifact.artifact_id)

    def get(self, tenant_id: str, artifact_id: str) -> Artifact:
        key = (tenant_id, artifact_id)
        return self._artifacts[key]

    def list_for_run(self, tenant_id: str, run_id: str) -> Iterable[Artifact]:
        run_key = (tenant_id, run_id)
        ids = self._by_run.get(run_key, [])
        for artifact_id in ids:
            yield self._artifacts[(tenant_id, artifact_id)]

    def delete_for_run(self, tenant_id: str, run_id: str) -> None:
        run_key = (tenant_id, run_id)
        ids = self._by_run.pop(run_key, [])
        for artifact_id in ids:
            self._artifacts.pop((tenant_id, artifact_id), None)