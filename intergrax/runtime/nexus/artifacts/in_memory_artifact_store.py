# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime

from intergrax.runtime.nexus.artifacts.models import Artifact
from intergrax.runtime.nexus.artifacts.store_base import (
    ArtifactDeletionRefused,
    ArtifactStore,
    ArtifactVersionNotFound,
    protection_reason,
    require_owned_class,
)


class InMemoryArtifactStore(ArtifactStore):
    """
    In-memory artifact store for tests and local development.

    Storage-only and append-only: versions are kept, nothing is overwritten.

    Isolation:
    - Fully tenant-scoped storage.
    - No global artifact namespace.
    """

    def __init__(self) -> None:
        self._versions: dict[tuple[str, str], list[Artifact]] = {}
        self._holds: set[tuple[str, str]] = set()
        self._lock = threading.RLock()

    def put(self, artifact: Artifact) -> Artifact:
        require_owned_class(artifact.artifact_class)
        key = (artifact.tenant_id, artifact.artifact_id)
        with self._lock:
            history = self._versions.setdefault(key, [])
            stored = replace(artifact, version=len(history) + 1)
            history.append(stored)
            return stored

    def get(self, tenant_id: str, artifact_id: str) -> Artifact:
        with self._lock:
            return self._versions[(tenant_id, artifact_id)][-1]

    def get_version(self, tenant_id: str, artifact_id: str, version: int) -> Artifact:
        with self._lock:
            history = self._versions.get((tenant_id, artifact_id), [])
            if version < 1 or version > len(history):
                raise ArtifactVersionNotFound((tenant_id, artifact_id, version))
            return history[version - 1]

    def list_versions(self, tenant_id: str, artifact_id: str) -> Iterable[Artifact]:
        with self._lock:
            return list(self._versions.get((tenant_id, artifact_id), []))

    def list_for_run(self, tenant_id: str, run_id: str) -> Iterable[Artifact]:
        with self._lock:
            return [
                history[-1]
                for (tenant, _), history in self._versions.items()
                if tenant == tenant_id and history[-1].run_id == run_id
            ]

    def list_expired(self, tenant_id: str, now: datetime, *, limit: int = 1000) -> list[Artifact]:
        with self._lock:
            expired = [
                history[-1]
                for (tenant, _), history in self._versions.items()
                if tenant == tenant_id and history[-1].expires_at_utc is not None and history[-1].expires_at_utc <= now
            ]
        expired.sort(key=lambda a: (a.expires_at_utc, a.artifact_id))
        return expired[:limit]

    def _refusal(self, tenant_id: str, artifact_id: str) -> str | None:
        history = self._versions.get((tenant_id, artifact_id), [])
        return protection_reason(
            (a.retention_class for a in history),
            legal_hold=(tenant_id, artifact_id) in self._holds,
        )

    def delete_artifact(self, tenant_id: str, artifact_id: str) -> None:
        key = (tenant_id, artifact_id)
        with self._lock:
            if key not in self._versions:
                raise KeyError(key)
            reason = self._refusal(tenant_id, artifact_id)
            if reason is not None:
                raise ArtifactDeletionRefused(artifact_id, reason)
            del self._versions[key]

    def delete_for_run(self, tenant_id: str, run_id: str) -> None:
        with self._lock:
            targets = [
                key for key, history in self._versions.items() if key[0] == tenant_id and history[-1].run_id == run_id
            ]
            for _, artifact_id in targets:
                reason = self._refusal(tenant_id, artifact_id)
                if reason is not None:
                    raise ArtifactDeletionRefused(artifact_id, reason)
            for key in targets:
                del self._versions[key]

    def place_legal_hold(self, tenant_id: str, artifact_id: str, *, placed_by: str, reason: str) -> None:
        with self._lock:
            if (tenant_id, artifact_id) not in self._versions:
                raise KeyError((tenant_id, artifact_id))
            self._holds.add((tenant_id, artifact_id))

    def release_legal_hold(self, tenant_id: str, artifact_id: str) -> None:
        with self._lock:
            self._holds.discard((tenant_id, artifact_id))

    def is_under_legal_hold(self, tenant_id: str, artifact_id: str) -> bool:
        with self._lock:
            return (tenant_id, artifact_id) in self._holds
