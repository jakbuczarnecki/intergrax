# © Artur Czarnecki. All rights reserved.

"""Focused rollback/error-path proofs for SQLiteAdaptiveProfileMutationStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.adaptive.contracts import (
    ProfileArtifactType,
    ProfileVersionDraft,
    ProfileVersionStatus,
)
from intergrax.runtime.adaptive.profile_mutation_store import SQLiteAdaptiveProfileMutationStore
from intergrax.runtime.adaptive.profile_pointer_store import SQLiteProfileActivePointerStore
from intergrax.runtime.adaptive.profile_version_store import SQLiteProfileVersionStore

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_TENANT = "tenant-a"
_TASK_CLASS = "echo.basic"


def _seed_active_baseline(db_path: Path, version_id: str = "v10") -> SQLiteAdaptiveProfileMutationStore:
    version_store = SQLiteProfileVersionStore(db_path=db_path)
    version_store.create_from_draft(
        ProfileVersionDraft(
            version_id=version_id,
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "baseline"},
            status=ProfileVersionStatus.ACTIVE,
        ),
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
    )
    pointer_store = SQLiteProfileActivePointerStore(db_path=db_path)
    pointer_store.swap_active(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id=version_id,
        expected_active_version_id=None,
    )
    return SQLiteAdaptiveProfileMutationStore(db_path=db_path)


def test_apply_unknown_target_raises_value_error_not_operational_error(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "adaptive_harness.db"
    mutation_store = _seed_active_baseline(db_path)
    version_store = SQLiteProfileVersionStore(db_path=db_path)
    pointer_store = SQLiteProfileActivePointerStore(db_path=db_path)

    with pytest.raises(ValueError, match="Unknown profile version: missing-v"):
        mutation_store.commit_apply(
            tenant_id=_TENANT,
            task_class=_TASK_CLASS,
            artifact_type=ProfileArtifactType.RAG,
            target_version_id="missing-v",
            expected_active_version_id="v10",
        )

    pointer = pointer_store.get_pointer(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert pointer is not None
    assert pointer.active_version_id == "v10"
    assert version_store.get("v10").status == ProfileVersionStatus.ACTIVE


def test_rollback_without_previous_pointer_raises_value_error(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "adaptive_harness.db"
    mutation_store = _seed_active_baseline(db_path)
    version_store = SQLiteProfileVersionStore(db_path=db_path)
    pointer_store = SQLiteProfileActivePointerStore(db_path=db_path)

    with pytest.raises(
        ValueError,
        match="No rollback pointer available for active profile version",
    ):
        mutation_store.commit_rollback(
            tenant_id=_TENANT,
            task_class=_TASK_CLASS,
            artifact_type=ProfileArtifactType.RAG,
            expected_active_version_id="v10",
        )

    pointer = pointer_store.get_pointer(
        tenant_id=_TENANT,
        task_class=_TASK_CLASS,
        artifact_type=ProfileArtifactType.RAG,
    )
    assert pointer is not None
    assert pointer.active_version_id == "v10"
    assert pointer.previous_version_id is None
    assert version_store.get("v10").status == ProfileVersionStatus.ACTIVE

