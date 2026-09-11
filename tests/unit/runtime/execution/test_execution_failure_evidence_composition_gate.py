# © Artur Czarnecki. All rights reserved.

"""Composition gate: single RuntimeEvent bus wires execution failure evidence."""

from __future__ import annotations

import ast
from pathlib import Path


def test_orchestration_wires_runtime_event_failure_recorder() -> None:
    source = Path("intergrax/runtime/execution/orchestration.py").read_text(encoding="utf-8")
    assert "RuntimeEventExecutionFailureEvidenceRecorder" in source
    assert source.count("RuntimeEventBus(") == 0


def test_nexus_host_execution_wires_failure_recorder_not_second_bus() -> None:
    source = Path(
        "intergrax/runtime/execution/nexus_host_execution.py",
    ).read_text(encoding="utf-8")
    assert "RuntimeEventExecutionFailureEvidenceRecorder" in source
    tree = ast.parse(source)
    bus_instantiations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RuntimeEventBus"
    ]
    assert bus_instantiations == []
