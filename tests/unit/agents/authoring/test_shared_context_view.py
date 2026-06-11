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


@pytest.mark.unit
@pytest.mark.gate
def test_shared_context_parallel_graph_cas_conflict() -> None:
    metadata: dict[str, object] = {}
    node_a = view_from_task_metadata(metadata, task_id="task-parallel")
    node_a.publish("handoff", {"score": 1}, updated_by="node-a")
    persist_view(metadata, node_a)

    node_b = view_from_task_metadata(metadata, task_id="task-parallel")
    value, version = node_b.get("handoff")
    assert value == {"score": 1}
    assert version == 1
    assert node_b.compare_and_swap("handoff", version, {"score": 2}, updated_by="node-b")
    persist_view(metadata, node_b)

    node_c = view_from_task_metadata(metadata, task_id="task-parallel")
    with pytest.raises(SharedContextConflictError):
        node_c.publish("handoff", {"score": 99}, expected_version=0)
    assert node_c.compare_and_swap("handoff", 0, {"score": 99}) is False
