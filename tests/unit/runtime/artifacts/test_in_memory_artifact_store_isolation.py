# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.runtime.nexus.artifacts.models import Artifact
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
from intergrax.utils.time_provider import SystemTimeProvider


def _artifact(
    tenant_id: str,
    artifact_id: str,
    run_id: str,
) -> Artifact:
    return Artifact(
        tenant_id=tenant_id,
        artifact_id=artifact_id,
        run_id=run_id,
        step_id=None,
        kind="test",
        mime_type="application/octet-stream",
        created_at_utc=SystemTimeProvider.utc_now(),
        data=b"data",
        size_bytes=4,
    )


def test_cross_tenant_isolation() -> None:
    store = InMemoryArtifactStore()

    artifact_a = _artifact("tenant_A", "artifact_A", "run_1")
    artifact_b = _artifact("tenant_B", "artifact_B", "run_1")

    store.put(artifact_a)
    store.put(artifact_b)

    # --- Correct access ---
    assert store.get("tenant_A", "artifact_A").tenant_id == "tenant_A"
    assert store.get("tenant_B", "artifact_B").tenant_id == "tenant_B"

    # --- Cross-tenant get must fail ---
    with pytest.raises(KeyError):
        store.get("tenant_A", "artifact_B")

    with pytest.raises(KeyError):
        store.get("tenant_B", "artifact_A")

    # --- Cross-tenant listing isolation ---
    list_a = list(store.list_for_run("tenant_A", "run_1"))
    list_b = list(store.list_for_run("tenant_B", "run_1"))

    assert len(list_a) == 1
    assert len(list_b) == 1
    assert list_a[0].artifact_id == "artifact_A"
    assert list_b[0].artifact_id == "artifact_B"

    # --- Deleting tenant_A must not affect tenant_B ---
    store.delete_for_run("tenant_A", "run_1")

    with pytest.raises(KeyError):
        store.get("tenant_A", "artifact_A")

    # tenant_B must still exist
    assert store.get("tenant_B", "artifact_B").tenant_id == "tenant_B"