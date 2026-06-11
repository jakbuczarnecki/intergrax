# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.shared_context_bridge import persist_view, view_from_task_metadata
from intergrax.contracts.shared_context import SharedArtifactRef, SharedContextConflictError


@pytest.mark.unit
@pytest.mark.gate
def test_shared_context_handoff_two_nodes() -> None:
    metadata: dict[str, object] = {}
    view_a = view_from_task_metadata(metadata, task_id="task-1")
    view_a.put_structured_output("node-a", {"summary": "from A"})
    persist_view(metadata, view_a)

    view_b = view_from_task_metadata(metadata, task_id="task-1")
    assert view_b.get_structured_output("node-a") == {"summary": "from A"}
    view_b.put_structured_output("node-b", {"summary": "from B"}, expected_version=view_b.version)
    persist_view(metadata, view_b)

    view_c = view_from_task_metadata(metadata, task_id="task-1")
    assert view_c.version == 3
    assert view_c.get_structured_output("node-b") == {"summary": "from B"}


@pytest.mark.unit
@pytest.mark.gate
def test_shared_context_cas_conflict() -> None:
    view = view_from_task_metadata({}, task_id="task-2")
    view.register_artifact(
        "doc",
        SharedArtifactRef(artifact_id="art-1", kind="text", size_bytes=12),
    )
    with pytest.raises(SharedContextConflictError):
        view.put_structured_output("k", {"x": 1}, expected_version=0)
