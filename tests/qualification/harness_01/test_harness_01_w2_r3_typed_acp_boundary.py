# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W2-R3 — typed ACP runtime boundary assertions (targeted seams)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.agents.authoring.acp_runtime_session_ports import AcpRuntimeSessionHooks
from intergrax.agents.authoring.acp_session_host import ACPSessionHostContext
from intergrax.agents.authoring.llm_router import LLMAdapterCompletePort, StepLLMRouter

_REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _field_annotation_names(cls: type, field_name: str) -> set[str]:
    hints = getattr(cls, "__annotations__", {})
    raw = hints.get(field_name)
    if raw is None:
        return set()
    return {part.strip() for part in str(raw).replace("|", " ").split() if part.strip()}


def test_acp_runtime_session_hooks_fields_are_not_any_or_ellipsis_callable() -> None:
    for field in AcpRuntimeSessionHooks.__dataclass_fields__.values():
        assert "Any" not in str(field.type)
        assert "..." not in str(field.type)


def test_acp_session_host_runtime_session_hooks_is_strongly_typed() -> None:
    annotation = str(ACPSessionHostContext.model_fields["runtime_session_hooks"].annotation)
    assert "Any" not in annotation
    assert "object" not in annotation
    assert "AcpRuntimeSessionHooks" in annotation


def test_step_llm_router_has_no_runtime_config_field() -> None:
    assert "runtime_config" not in StepLLMRouter.__dataclass_fields__


def test_llm_adapter_complete_port_adapter_is_typed() -> None:
    params = inspect.signature(LLMAdapterCompletePort.__init__).parameters
    adapter = params["adapter"]
    assert "object" not in str(adapter.annotation)
    assert "LLMAdapter" in str(adapter.annotation)


def test_intergrax_agents_reference_harness_has_no_dynamic_nexus_trampoline() -> None:
    path = _REPO_ROOT / "intergrax" / "agents" / "reference_harness.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "import_module":
                raise AssertionError("reference_harness must not call importlib.import_module")
