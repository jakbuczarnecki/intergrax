# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Iterable, Protocol

from intergrax.runtime.nexus.artifacts.models import Artifact


class ArtifactStore(Protocol):
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

    def put(self, artifact: Artifact) -> None:
        ...

    def get(self, tenant_id: str, artifact_id: str) -> Artifact:
        ...

    def list_for_run(self, tenant_id: str, run_id: str) -> Iterable[Artifact]:
        ...

    def delete_for_run(self, tenant_id: str, run_id: str) -> None:
        ...
