# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W2-R5 — shared-context capability contract (no metadata-bag masking)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.agents.authoring.acp_runtime_session_ports import (
    AcpRuntimeSessionHooks,
    SharedContextAccessForRunPort,
)
from intergrax.agents.authoring.shared_context_access import (
    InMemorySharedContextAccess,
    RequestBoundSharedContextAccess,
    neutral_shared_context_access_for_run,
)
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.shared_context_access import SharedContextAccessPort
from intergrax.contracts.shared_context import SharedContextView

_REPO_ROOT = Path(__file__).resolve().parents[3]

_W2_SHARED_CONTEXT_SEAM_PATHS = (
    _REPO_ROOT / "intergrax" / "contracts" / "shared_context_access.py",
    _REPO_ROOT / "intergrax" / "agents" / "authoring" / "acp_runtime_session_ports.py",
    _REPO_ROOT / "intergrax" / "agents" / "authoring" / "shared_context_access.py",
)

_FORBIDDEN_ANNOTATION_FRAGMENTS = (
    "Any",
    "object",
    "MutableMapping[str, object]",
    "Mapping[str, object]",
    "dict[str, object]",
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _annotation_strings(node: ast.AST) -> list[str]:
    out: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.AnnAssign) and child.annotation is not None:
            out.append(ast.unparse(child.annotation))
        elif isinstance(child, ast.arg) and child.annotation is not None:
            out.append(ast.unparse(child.annotation))
        elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            if child.returns is not None:
                out.append(ast.unparse(child.returns))
    return out


def _assert_seam_annotations_clean(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for annotation in _annotation_strings(tree):
        for forbidden in _FORBIDDEN_ANNOTATION_FRAGMENTS:
            assert forbidden not in annotation, (path.as_posix(), annotation, forbidden)


def test_w2_shared_context_seam_rejects_metadata_bag_annotations() -> None:
    for path in _W2_SHARED_CONTEXT_SEAM_PATHS:
        _assert_seam_annotations_clean(path)


def test_shared_context_access_port_is_typed_capability() -> None:
    load = inspect.signature(SharedContextAccessPort.load)
    assert "SharedContextView" in str(load.return_annotation)
    persist = inspect.signature(SharedContextAccessPort.persist)
    assert "SharedContextView" in str(persist.parameters["view"].annotation)


def test_shared_context_access_for_run_port_uses_agent_run_request() -> None:
    sig = inspect.signature(SharedContextAccessForRunPort.__call__)
    request = sig.parameters["request"]
    assert "AgentRunRequest" in str(request.annotation)
    assert "SharedContextAccessPort" in str(sig.return_annotation)


def test_acp_runtime_session_hooks_shared_context_field_is_capability_port() -> None:
    field = AcpRuntimeSessionHooks.__dataclass_fields__["resolve_shared_context_access"]
    assert "SharedContextAccessForRunPort" in str(field.type)
    assert "object" not in str(field.type)
    assert "Any" not in str(field.type)


def test_in_memory_shared_context_access_round_trip() -> None:
    access = InMemorySharedContextAccess()
    view = SharedContextView(task_id="task-1")
    access.persist(view)
    loaded = access.load()
    assert loaded is not None
    assert loaded.task_id == "task-1"
    projected = access.project(task_id="task-2")
    assert projected.task_id == "task-2"


def test_request_bound_shared_context_access_round_trip() -> None:
    request = AgentRunRequest(
        input="hello",
        identity=RequestIdentity(tenant_id="t1"),
        agent_id="a1",
    )
    access = neutral_shared_context_access_for_run(request)
    assert isinstance(access, RequestBoundSharedContextAccess)
    view = SharedContextView(task_id="task-1")
    access.persist(view)
    loaded = access.load()
    assert loaded is not None
    assert loaded.task_id == "task-1"


def test_w2_gate_negative_metadata_bag_alias_fails_guard() -> None:
    snippet = "def load(metadata: MutableMapping[str, object]) -> None:\n    pass\n"
    tree = ast.parse(snippet)
    for annotation in _annotation_strings(tree):
        for forbidden in _FORBIDDEN_ANNOTATION_FRAGMENTS:
            if forbidden in annotation:
                return
    raise AssertionError("negative fixture must contain forbidden annotation")


def test_w2_gate_negative_object_param_fails_guard() -> None:
    snippet = "def load(metadata: object):\n    pass\n"
    tree = ast.parse(snippet)
    annotations = _annotation_strings(tree)
    assert any("object" in ann for ann in annotations)


def test_w2_gate_negative_any_dict_param_fails_guard() -> None:
    snippet = "def persist(metadata: dict[str, Any]):\n    pass\n"
    tree = ast.parse(snippet)
    annotations = _annotation_strings(tree)
    assert any("Any" in ann for ann in annotations)


def test_w2_gate_positive_capability_protocol_passes_guard() -> None:
    snippet = (
        "class SharedContextAccessPort(Protocol):\n"
        "    def load(self) -> SharedContextView | None: ...\n"
    )
    tree = ast.parse(snippet)
    for annotation in _annotation_strings(tree):
        for forbidden in _FORBIDDEN_ANNOTATION_FRAGMENTS:
            assert forbidden not in annotation
