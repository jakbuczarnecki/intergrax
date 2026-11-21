# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.artifact_ref import ArtifactClass, ArtifactRetentionClass
from intergrax.runtime.nexus.artifacts.models import Artifact
from intergrax.runtime.nexus.artifacts.taxonomy import STORE_ACCEPTED_CLASSES


class ArtifactClassNotOwnedError(ValueError):
    """The execution artifact store does not own this artifact class (see the taxonomy)."""


class ArtifactDeletionRefused(RuntimeError):
    """Deletion refused: audit evidence or a legal hold protects the artifact."""

    def __init__(self, artifact_id: str, reason: str) -> None:
        super().__init__(f"deletion of artifact {artifact_id!r} refused: {reason}")
        self.artifact_id = artifact_id
        self.reason = reason


class ArtifactVersionChanged(RuntimeError):
    """A conditional deletion found a different latest version than the caller observed."""

    def __init__(self, artifact_id: str, expected_version: int, actual_version: int) -> None:
        super().__init__(
            f"deletion of artifact {artifact_id!r} skipped: expected latest version {expected_version}, "
            f"found {actual_version}"
        )
        self.artifact_id = artifact_id
        self.expected_version = expected_version
        self.actual_version = actual_version


class ArtifactVersionNotFound(KeyError):
    """The artifact exists (or not) but the requested version does not."""


@dataclass(frozen=True, slots=True)
class LegalHoldRecord:
    """One legal hold on an artifact; a hold is identified by (tenant, artifact, holder) (F-D8-14).

    The tenant is the scope of the store call that returned the record and is not repeated here.

    Placing a hold again by the same holder keeps the first record (original reason and time);
    a hold by another holder adds a record. An artifact is under legal hold while any record exists.
    """

    artifact_id: str
    placed_by: str
    reason: str
    placed_at_utc: datetime


def require_owned_class(artifact_class: ArtifactClass) -> None:
    if artifact_class not in STORE_ACCEPTED_CLASSES:
        raise ArtifactClassNotOwnedError(
            f"artifact class {artifact_class.value!r} is owned elsewhere; "
            "the execution artifact store accepts only "
            f"{sorted(c.value for c in STORE_ACCEPTED_CLASSES)}"
        )


def protection_reason(
    retention_classes: Iterable[ArtifactRetentionClass],
    *,
    legal_hold: bool,
) -> str | None:
    """Why an artifact may not be deleted, or ``None`` when deletion is allowed.

    A data-integrity invariant of the store, not an authorization decision: authorization
    is the job of the governed access port, never of a store.
    """
    if any(rc is ArtifactRetentionClass.AUDIT_EVIDENCE for rc in retention_classes):
        return "audit_evidence"
    if legal_hold:
        return "legal_hold"
    return None


class ArtifactStore(ABC):
    """
    Storage port for execution artifacts.

    Implementations MUST:
    - persist artifacts durably (or in-memory for tests),
    - keep every version (append-only): ``put`` appends, it never overwrites,
    - allow listing artifacts for a run,
    - allow fetching a single artifact by id (latest) or by version,
    - refuse to delete audit evidence and artifacts under legal hold.

    Implementations MUST NOT:
    - embed artifacts into trace (trace stores ArtifactRef only),
    - implement replay logic (separate step),
    - implement security/scopes: authorization, tenant checks and audit events belong to
      the governed access port (``governed_access``), never to a store.
    """

    @abstractmethod
    def put(self, artifact: Artifact) -> Artifact:
        """Append ``artifact`` as a new version and return the stored version."""

    @abstractmethod
    def get(self, tenant_id: str, artifact_id: str) -> Artifact:
        """Return the latest version; ``KeyError`` when absent."""

    @abstractmethod
    def get_version(self, tenant_id: str, artifact_id: str, version: int) -> Artifact:
        """Return one version; ``ArtifactVersionNotFound`` when absent."""

    @abstractmethod
    def list_versions(self, tenant_id: str, artifact_id: str) -> Iterable[Artifact]:
        """All versions, oldest first (empty when the artifact is unknown)."""

    @abstractmethod
    def list_for_run(self, tenant_id: str, run_id: str) -> Iterable[Artifact]:
        """Latest version of every artifact of the run."""

    @abstractmethod
    def list_expired(self, tenant_id: str, now: datetime, *, limit: int = 1000) -> list[Artifact]:
        """Latest versions whose ``expires_at_utc`` is at or before ``now`` (candidates only)."""

    @abstractmethod
    def delete_artifact(self, tenant_id: str, artifact_id: str, *, expected_version: int | None = None) -> None:
        """Delete every version; ``ArtifactDeletionRefused`` for evidence or legal hold.

        With ``expected_version`` the deletion happens only if the latest version still equals it,
        decided atomically with the protection check; otherwise ``ArtifactVersionChanged`` is
        raised and nothing is deleted (a version written after the caller selected the artifact
        is never destroyed by a decision taken on an older one).
        """

    @abstractmethod
    def delete_for_run(self, tenant_id: str, run_id: str) -> None:
        """Delete all artifacts of the run, atomically refusing if any is protected."""

    @abstractmethod
    def place_legal_hold(self, tenant_id: str, artifact_id: str, *, placed_by: str, reason: str) -> None:
        """Add a hold record for ``placed_by``; an existing record of the same holder is kept unchanged."""

    @abstractmethod
    def release_legal_hold(self, tenant_id: str, artifact_id: str) -> None:
        """Remove every hold record of the artifact (release without a holder)."""

    @abstractmethod
    def is_under_legal_hold(self, tenant_id: str, artifact_id: str) -> bool: ...
