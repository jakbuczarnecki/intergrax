# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-G (PT-WA): relocated contracts keep one definition and identical behaviour.

NT-WA-09 (graph topology), NT-WA-15 (retrieval source scope), NT-WA-16 (workspace tool ids).
"""

from __future__ import annotations

import hashlib
import json

import pytest
from intergrax.applications.contracts import graph_spec as tier3_graph_spec
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec
from intergrax.contracts import application_graph_topology as neutral_topology
from intergrax.contracts import rag_retrieval_source_scope as neutral_scope
from intergrax.contracts import workspace_tool_ids as neutral_ids
from intergrax.tools.providers.rag import source_scope_transport as legacy_scope
from intergrax.tools.providers.workspace import tool_ids as legacy_ids

pytestmark = [pytest.mark.unit, pytest.mark.gate]

# Captured from ``ApplicationGraphSpec.model_json_schema()`` at the train head before the change.
_PRE_CHANGE_PROPERTY_ORDER = (
    "nodes",
    "edges",
    "retry_on_error",
    "trigger_capabilities",
    "pipeline_capability_suffix",
    "evaluator_loop",
    "codecraft_node",
)
_PRE_CHANGE_SCHEMA_SHA256 = "c121c3d2c11773ae0d8715a7e7694ad535eecfee4358f4a56e1bc8a2937a1c7d"


def test_wa_09_graph_spec_is_topology_subclass_schema_stable() -> None:
    schema = ApplicationGraphSpec.model_json_schema()
    assert tuple(schema["properties"]) == _PRE_CHANGE_PROPERTY_ORDER
    blob = json.dumps(
        {"properties": schema["properties"], "defs": sorted(schema["$defs"])},
        sort_keys=True,
    )
    assert hashlib.sha256(blob.encode()).hexdigest() == _PRE_CHANGE_SCHEMA_SHA256
    assert issubclass(ApplicationGraphSpec, neutral_topology.ApplicationGraphTopology)
    spec = ApplicationGraphSpec(nodes=[neutral_topology.GraphNode(agent_id="a")])
    assert isinstance(spec, neutral_topology.ApplicationGraphTopology)
    assert tier3_graph_spec.GraphEdgeKind is neutral_topology.GraphEdgeKind
    assert tier3_graph_spec.GraphEdge is neutral_topology.GraphEdge
    assert tier3_graph_spec.GraphNode is neutral_topology.GraphNode
    assert tier3_graph_spec.CodeCraftGraphBinding is neutral_topology.CodeCraftGraphBinding
    with pytest.raises(ValueError, match="duplicate graph node"):
        ApplicationGraphSpec(
            nodes=[
                neutral_topology.GraphNode(agent_id="a"),
                neutral_topology.GraphNode(agent_id="a"),
            ],
        )
    assert spec.roster_agent_ids() == frozenset({"a"})


def test_wa_15_source_scope_single_definition_and_rejections() -> None:
    assert legacy_scope.RagRetrievalSourceScopeState is neutral_scope.RagRetrievalSourceScopeState
    assert legacy_scope._RAG_RETRIEVAL_SOURCE_SCOPE is neutral_scope._RAG_RETRIEVAL_SOURCE_SCOPE
    state = neutral_scope.validated_rag_retrieval_source_scope(("doc-1", "doc-2"))
    token = neutral_scope.bind_rag_retrieval_source_scope(state)
    try:
        assert legacy_scope.current_rag_retrieval_source_scope() is state
    finally:
        neutral_scope.reset_rag_retrieval_source_scope(token)
    assert legacy_scope.current_rag_retrieval_source_scope().is_absent

    empty = neutral_scope.validated_rag_retrieval_source_scope(())
    assert empty.is_invalid and empty.error_reason == neutral_scope.SOURCE_SCOPE_EMPTY
    blank = neutral_scope.validated_rag_retrieval_source_scope(("  ",))
    assert blank.is_invalid and blank.error_reason == neutral_scope.SOURCE_SCOPE_BLANK_ONLY
    too_long = neutral_scope.validated_rag_retrieval_source_scope(("x" * 257,))
    assert too_long.is_invalid and too_long.error_reason == neutral_scope.SOURCE_SCOPE_MALFORMED
    metadata = neutral_scope.parse_task_metadata_allowed_source_ids(
        {"allowed_source_ids": ["a", " b "]},
    )
    assert metadata.allowed_source_ids == ("a", "b")


def test_wa_16_workspace_tool_ids_single_definition() -> None:
    assert legacy_ids.WORKSPACE_SEARCH_TOOL_ID is neutral_ids.WORKSPACE_SEARCH_TOOL_ID
    assert legacy_ids.WORKSPACE_WRITE_FILE_TOOL_ID is neutral_ids.WORKSPACE_WRITE_FILE_TOOL_ID
    assert neutral_ids.WORKSPACE_SEARCH_TOOL_ID == "workspace.search"
    assert neutral_ids.WORKSPACE_WRITE_FILE_TOOL_ID == "workspace.write_file"
    assert (
        legacy_ids.WORKSPACE_READ_FILE_TOOL_ID,
        legacy_ids.WORKSPACE_LIST_FILES_TOOL_ID,
        legacy_ids.WORKSPACE_SNAPSHOT_TOOL_ID,
        legacy_ids.WORKSPACE_DELETE_FILE_TOOL_ID,
        legacy_ids.WORKSPACE_EXPORT_ARTIFACT_TOOL_ID,
        legacy_ids.WORKSPACE_IMPORT_ARTIFACT_TOOL_ID,
    ) == (
        "workspace.read_file",
        "workspace.list_files",
        "workspace.snapshot",
        "workspace.delete_file",
        "workspace.export_artifact",
        "workspace.import_artifact",
    )
