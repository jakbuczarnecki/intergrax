# © Artur Czarnecki. All rights reserved.

"""EBH-4-R1-R3-B6 — closed-world semantic typing regression gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_INTERGRAX = _REPO / "intergrax"

_SCENARIO_DIAG = _INTERGRAX / "runtime" / "execution" / "scenario_host_diagnostic_wiring.py"
_DIAG_WIRING = (
    _INTERGRAX / "applications" / "_shared" / "diagnostic_runtime_wiring.py"
)
_ACP_HOST = _INTERGRAX / "agents" / "authoring" / "acp_session_host.py"
_STEP_KERNEL = _INTERGRAX / "runtime" / "kernel" / "step_kernel.py"
_HOST_CONTRACT = (
    _INTERGRAX / "contracts" / "host_orchestration_application_wiring_target.py"
)
_NEXUS_LOOP = _INTERGRAX / "runtime" / "nexus" / "nexus_loop.py"

_EE_CONTRACT_CONCRETE_LOCK_FORBIDDEN = (
    ("isinstance(validation_engine, NexusValidationEngine)", _NEXUS_LOOP),
    ("type(selection) is not DecisionExposureSelectionComposition", _NEXUS_LOOP),
)
_HOST_BUILDER = (
    _INTERGRAX
    / "runtime"
    / "execution"
    / "host_orchestration_environment_spec_builder.py"
)

_B6_SEMANTIC_TYPE_IGNORE_FORBIDDEN = (
    _SCENARIO_DIAG,
    _DIAG_WIRING,
    _HOST_BUILDER,
)

_ACP_CAPABILITY_FIELDS = (
    "declarative_tool_invoker",
    "decision_flow_gate",
    "notification_adapter",
    "budget_reaction_hook",
)

_STEP_KERNEL_CAPABILITY_FIELDS = (
    "budget_reaction",
    "notification_adapter",
    "budget_reaction_hook",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_scenario_diagnostic_uses_orchestration_host_keyword() -> None:
    source = _read(_SCENARIO_DIAG)
    assert "nexus_loop=" not in source
    assert "orchestration_host=" in source


def test_b6_audited_surfaces_have_no_semantic_type_ignore() -> None:
    for path in _B6_SEMANTIC_TYPE_IGNORE_FORBIDDEN:
        source = _read(path)
        assert "# type: ignore" not in source, f"semantic type ignore in {path}"


def test_acp_session_host_capabilities_are_not_any() -> None:
    tree = ast.parse(_read(_ACP_HOST))
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id not in _ACP_CAPABILITY_FIELDS:
            continue
        assert node.annotation is not None
        ann_src = ast.get_source_segment(_read(_ACP_HOST), node.annotation) or ""
        assert "Any" not in ann_src, node.target.id


def test_step_kernel_capability_fields_are_not_any() -> None:
    tree = ast.parse(_read(_STEP_KERNEL))
    class_names = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    assert "StepKernelContext" in class_names
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id not in _STEP_KERNEL_CAPABILITY_FIELDS:
            continue
        ann_src = ast.get_source_segment(_read(_STEP_KERNEL), node.annotation) or ""
        assert "Any" not in ann_src, node.target.id


def test_neutral_host_contract_does_not_import_nexus() -> None:
    tree = ast.parse(_read(_HOST_CONTRACT))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("intergrax.runtime.nexus"), node.module
            assert not node.module.startswith("intergrax.runtime."), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("intergrax.runtime.nexus"), alias.name
                assert not alias.name.startswith("intergrax.runtime."), alias.name


def test_neutral_host_contract_has_no_semantic_object_or_any() -> None:
    source = _read(_HOST_CONTRACT)
    tree = ast.parse(source)
    forbidden = ("object", "Any")
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not node.name.startswith("HostOrchestration"):
            continue
        for item in node.body:
            ann: ast.expr | None = None
            if isinstance(item, ast.AnnAssign) and item.annotation is not None:
                ann = item.annotation
            elif isinstance(item, ast.FunctionDef):
                if item.returns is not None:
                    ann = item.returns
                for arg in item.args.args:
                    if arg.annotation is not None:
                        segment = ast.get_source_segment(source, arg.annotation) or ""
                        assert not any(
                            token == forbidden[0] or token == forbidden[1]
                            for token in segment.replace("|", " ").split()
                        ), f"{node.name}.{item.name} param uses semantic escape"
            if ann is None:
                continue
            ann_src = ast.get_source_segment(source, ann) or ""
            tokens = ann_src.replace("|", " ").replace("[", " ").replace("]", " ").split()
            for token in tokens:
                if token in forbidden:
                    raise AssertionError(
                        f"{node.name} annotation uses forbidden semantic escape: {ann_src!r}",
                    )


def test_neutral_host_contract_types_terminal_diagnostic_port() -> None:
    source = _read(_HOST_CONTRACT)
    assert "port: TerminalExecutionDiagnosticPort" in source


def test_wire_terminal_execution_diagnostics_types_scenario_runtime_mode() -> None:
    source = _read(_DIAG_WIRING)
    assert "scenario_runtime_mode: ScenarioRuntimeMode | None" in source
    assert "scenario_runtime_mode: object" not in source


def test_b6_r2_audited_paths_have_no_contract_concrete_locks() -> None:
    for fragment, path in _EE_CONTRACT_CONCRETE_LOCK_FORBIDDEN:
        source = _read(path)
        assert fragment not in source, f"concrete lock {fragment!r} in {path}"
