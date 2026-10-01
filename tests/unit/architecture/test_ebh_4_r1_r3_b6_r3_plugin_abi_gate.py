# © Artur Czarnecki. All rights reserved.

"""EBH-4-R1-R3-B6-R3 — canonical runtime plugin registration ABI gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.host_orchestration_wiring_capabilities import (
    HostOrchestrationRuntimeEventPort,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.plugins.contract import RuntimePlugin, RuntimePluginRegisterCallback

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_INTERGRAX = _REPO / "intergrax"
_PLUGIN_CONTRACT = _INTERGRAX / "runtime" / "plugins" / "contract.py"
_PLUGIN_BOOTSTRAP = _INTERGRAX / "runtime" / "plugins" / "bootstrap.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _production_py_files() -> list[Path]:
    skip = {"__pycache__", "tests", "build", ".tmp"}
    paths: list[Path] = []
    for path in _INTERGRAX.rglob("*.py"):
        if any(part in skip for part in path.parts):
            continue
        paths.append(path)
    return paths


def test_runtime_event_bus_like_not_defined_in_production() -> None:
    assert "class RuntimeEventBusLike" not in _read(_PLUGIN_CONTRACT)
    for path in _production_py_files():
        source = _read(path)
        assert "RuntimeEventBusLike" not in source, path


def test_policy_engine_like_not_defined_in_production() -> None:
    assert "class PolicyEngineLike" not in _read(_PLUGIN_CONTRACT)
    for path in _production_py_files():
        source = _read(path)
        assert "PolicyEngineLike" not in source, path


def test_runtime_plugin_register_uses_host_event_port_only() -> None:
    source = _read(_PLUGIN_CONTRACT)
    assert "HostOrchestrationRuntimeEventPort" in source
    assert "RuntimePluginRegisterCallback" in source
    assert "HookRegistry" not in source
    assert "PolicyEngineLike" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "RuntimePlugin":
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or item.target.id != "register":
                continue
            ann = ast.get_source_segment(source, item.annotation) or ""
            assert "RuntimePluginRegisterCallback" in ann


def test_bootstrap_runtime_plugins_single_event_bus_parameter() -> None:
    tree = ast.parse(_read(_PLUGIN_BOOTSTRAP))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "bootstrap_runtime_plugins":
            continue
        kwonly = [a.arg for a in node.args.kwonlyargs]
        assert kwonly == ["event_bus"]
        assert "hook_registry" not in _read(_PLUGIN_BOOTSTRAP)
        assert "policy_engine" not in _read(_PLUGIN_BOOTSTRAP)


def test_runtime_event_bus_structurally_implements_host_event_port() -> None:
    bus: HostOrchestrationRuntimeEventPort = RuntimeEventBus()
    _ = bus


def test_runtime_plugin_register_callback_alias_matches_canonical_port() -> None:
    def _sample(event_bus: HostOrchestrationRuntimeEventPort) -> None:
        _ = event_bus

    callback: RuntimePluginRegisterCallback = _sample
    plugin = RuntimePlugin(plugin_id="gate.sample", version="1.0.0", register=callback)
    assert plugin.register is _sample
