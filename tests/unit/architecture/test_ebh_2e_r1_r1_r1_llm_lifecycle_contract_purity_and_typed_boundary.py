# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R1-R1-R1 — internal lifecycle composition purity and typed boundaries."""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_PATH = _REPO_ROOT / "intergrax/llm_adapters/base/lifecycle_binding.py"
_COMPAT_PATH = _REPO_ROOT / "intergrax/llm_adapters/contracts/runtime_lifecycle_binding.py"
_COMPOSITION_PATHS = (
    _REPO_ROOT / "intergrax/llm_adapters/_shared/provider_dependency_boundary.py",
    _REPO_ROOT / "intergrax/runtime/external_operations/provider_cancellation.py",
)
_FORBIDDEN_COMPAT_PREFIXES = (
    "intergrax.runtime.",
    "intergrax.llm_adapters.providers.",
    "intergrax.llm_adapters._shared.",
)


def _import_module_names(path: Path) -> list[str]:
    names: list[str] = []
    raw = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
    return names


def _class_defs_named(root: Path, name: str) -> list[Path]:
    hits: list[Path] = []
    for path in root.rglob("*.py"):
        if "contracts" in path.parts and path != _COMPAT_PATH:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                hits.append(path)
    return hits


def test_ebh_2e_r1_r1_r1_lifecycle_is_internal_single_authority() -> None:
    hits = _class_defs_named(_REPO_ROOT / "intergrax/llm_adapters", "LLMRuntimeLifecycleBinding")
    assert hits == [_CANONICAL_PATH]


def test_ebh_2e_r1_r1_r1_compat_reexport_identity() -> None:
    canonical = importlib.import_module("intergrax.llm_adapters.base.lifecycle_binding")
    compat = importlib.import_module(
        "intergrax.llm_adapters.contracts.runtime_lifecycle_binding"
    )
    assert compat.LLMRuntimeLifecycleBinding is canonical.LLMRuntimeLifecycleBinding


def test_ebh_2e_r1_r1_r1_compat_module_has_no_runtime_type_checking_edges() -> None:
    imports = _import_module_names(_COMPAT_PATH)
    for mod in imports:
        for prefix in _FORBIDDEN_COMPAT_PREFIXES:
            assert not mod.startswith(prefix), f"compat import {mod}"


def test_ebh_2e_r1_r1_r1_compat_module_has_no_protocol_class_def() -> None:
    tree = ast.parse(_COMPAT_PATH.read_text(encoding="utf-8-sig"))
    class_defs = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert not class_defs


def test_ebh_2e_r1_r1_r1_lifecycle_boundary_no_any() -> None:
    text = _CANONICAL_PATH.read_text(encoding="utf-8")
    assert "Any" not in text


def test_ebh_2e_r1_r1_r1_composition_helpers_reject_object_adapter_param() -> None:
    pattern = re.compile(r"adapter\s*:\s*object\b")
    offenders: list[str] = []
    for path in _COMPOSITION_PATHS:
        for line in path.read_text(encoding="utf-8").splitlines():
            if pattern.search(line):
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders


def test_ebh_2e_r1_r1_r1_bind_ports_signature_uses_llm_adapter() -> None:
    from intergrax.runtime.external_operations.provider_cancellation import (
        bind_llm_external_operation_ports,
    )

    sig = inspect.signature(bind_llm_external_operation_ports)
    adapter_param = sig.parameters["adapter"]
    assert adapter_param.annotation is not inspect.Parameter.empty
    assert "LLMAdapter" in str(adapter_param.annotation)


def test_ebh_2e_r1_r1_r1_composition_no_base_adapter_detection() -> None:
    offenders: list[str] = []
    for path in _COMPOSITION_PATHS:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "BaseLLMAdapter" in line and "isinstance" in line:
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders


def test_ebh_2e_r1_r1_r1_composition_no_reflection_for_lifecycle() -> None:
    offenders: list[str] = []
    for path in _COMPOSITION_PATHS:
        text = path.read_text(encoding="utf-8")
        for token in ("hasattr(", "getattr(", "callable("):
            if token in text and "bind_" in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders
