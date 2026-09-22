# © Artur Czarnecki. All rights reserved.

"""EBH-2E-AR1-B-R1 — provider lifecycle boundary enforcement (Model A)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.llm_adapters._shared.provider_stream_transport_registry import (
    ProviderStreamTransportRegistry,
)
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.base.lifecycle_binding import LLMRuntimeLifecycleBinding

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LLM_ADAPTERS_ROOT = _REPO_ROOT / "intergrax/llm_adapters"
_LIFECYCLE_BINDING = _REPO_ROOT / "intergrax/llm_adapters/base/lifecycle_binding.py"
_BASE_ADAPTER = _REPO_ROOT / "intergrax/llm_adapters/base/base_llm_adapter.py"
_STREAM_REGISTRY = (
    _REPO_ROOT / "intergrax/llm_adapters/_shared/provider_stream_transport_registry.py"
)


def _function_param_names(path: Path, function_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return [arg.arg for arg in node.args.args + node.args.kwonlyargs]
    raise AssertionError(f"{function_name} not found in {path}")


def _llm_adapters_imports_operation_termination() -> list[str]:
    offenders: list[str] = []
    needle = "intergrax.runtime.external_operations.operation_termination"
    for path in _LLM_ADAPTERS_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if needle in text:
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    return offenders


def test_ebh_2e_ar1_b_r1_lifecycle_binding_has_no_capabilities_param() -> None:
    params = _function_param_names(_LIFECYCLE_BINDING, "bind_external_operation_ports")
    assert "capabilities" not in params


def test_ebh_2e_ar1_b_r1_base_adapter_has_no_capabilities_param() -> None:
    params = _function_param_names(_BASE_ADAPTER, "bind_external_operation_ports")
    assert "capabilities" not in params


def test_ebh_2e_ar1_b_r1_base_adapter_resolves_capabilities_from_registry_authority() -> None:
    source = _BASE_ADAPTER.read_text(encoding="utf-8")
    assert "external_operation_capabilities_for_provider" in source
    bind_block_start = source.index("def bind_external_operation_ports")
    bind_block = source[bind_block_start : bind_block_start + 2500]
    assert "capabilities is not None" not in bind_block
    assert "capabilities:" not in bind_block.split("def bind_external_operation_ports", 1)[1].split(
        "def bind_provider_dependency_boundary", 1
    )[0]


def test_ebh_2e_ar1_b_r1_no_llm_adapters_import_operation_termination() -> None:
    assert _llm_adapters_imports_operation_termination() == []


def test_ebh_2e_ar1_b_r1_stream_registry_self_contained() -> None:
    text = _STREAM_REGISTRY.read_text(encoding="utf-8")
    assert "runtime.external_operations" not in text


def test_ebh_2e_ar1_b_r1_provider_stream_transport_registry_register_and_close() -> None:
    registry = ProviderStreamTransportRegistry()
    closed: list[str] = []

    def closer_a() -> None:
        closed.append("a")

    registry.register("op-1", closer_a)
    assert registry.close_transport("op-1") is True
    assert closed == ["a"]
    assert registry.close_transport("op-1") is False


def test_ebh_2e_ar1_b_r1_provider_stream_transport_registry_replace_closer() -> None:
    registry = ProviderStreamTransportRegistry()
    seen: list[str] = []

    registry.register("op-1", lambda: seen.append("first"))
    registry.register("op-1", lambda: seen.append("second"))
    assert registry.close_transport("op-1") is True
    assert seen == ["second"]


def test_ebh_2e_ar1_b_r1_provider_stream_transport_registry_clear() -> None:
    registry = ProviderStreamTransportRegistry()
    registry.register("op-1", lambda: None)
    registry.clear()
    assert registry.close_transport("op-1") is False


def test_ebh_2e_ar1_b_r1_provider_stream_transport_registry_invalid_operation_id() -> None:
    registry = ProviderStreamTransportRegistry()
    with pytest.raises(ValueError, match="operation_id"):
        registry.register("", lambda: None)


def test_ebh_2e_ar1_b_r1_lifecycle_protocol_surface_unchanged_except_capabilities() -> None:
    """Structural check: protocol still exposes bind_external_operation_ports."""
    assert hasattr(LLMRuntimeLifecycleBinding, "bind_external_operation_ports")
    assert hasattr(BaseLLMAdapter, "bind_external_operation_ports")
