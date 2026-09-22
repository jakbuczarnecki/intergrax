# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R1-R1 — structural LLM runtime lifecycle capability binding."""

from __future__ import annotations

import ast
import importlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters._shared.provider_dependency_boundary import (
    apply_llm_provider_dependency_boundary,
    set_llm_provider_dependency_boundary,
)
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.lifecycle_binding import LLMRuntimeLifecycleBinding
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.strict_tool_arguments import CanonicalFunctionToolDefinition
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.runtime.external_operations.provider_cancellation import (
    bind_llm_external_operation_ports,
)
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
)
from intergrax.runtime.resilience.local_dependency_concurrency_admission import (
    LocalDependencyConcurrencyAdmission,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_LIFECYCLE_PATH = _REPO_ROOT / "intergrax/llm_adapters/base/lifecycle_binding.py"
_LIFECYCLE_COMPOSITION_PATHS = (
    _REPO_ROOT / "intergrax/llm_adapters/_shared/provider_dependency_boundary.py",
    _REPO_ROOT / "intergrax/runtime/external_operations/provider_cancellation.py",
)
_BASE_LIFECYCLE_ISINSTANCE_ALLOWLIST = {
    _REPO_ROOT / "intergrax/llm_adapters/llm_provider_registry.py",
}


class _LifecycleBindingRecorder:
    dependency_boundary: object | None = None
    admission_gate: object | None = None
    external_ports: dict[str, object | None] | None = None

    def bind_provider_dependency_boundary(self, boundary: object | None) -> None:
        _LifecycleBindingRecorder.dependency_boundary = boundary

    def bind_external_operation_admission_gate(self, gate: object | None) -> None:
        _LifecycleBindingRecorder.admission_gate = gate

    def bind_external_operation_ports(
        self,
        *,
        store: object | None,
        owner: object | None = None,
        cancellation_port: object | None = None,
        status_port: object | None = None,
        termination_port: object | None = None,
        stream_registry: object | None = None,
    ) -> None:
        _LifecycleBindingRecorder.external_ports = {
            "store": store,
            "owner": owner,
            "cancellation_port": cancellation_port,
            "status_port": status_port,
            "termination_port": termination_port,
            "stream_registry": stream_registry,
        }


class _ExternalExecutionOnlyAdapter:
    provider = "external-exec-only"
    model = "ext-model"

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        return build_adapter_response(content="external")

    def supports_streaming(self) -> bool:
        return False

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        raise NotImplementedError

    def supports_tools(self) -> bool:
        return False

    def supports_strict_tool_argument_conformance(self) -> bool:
        return False

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[CanonicalFunctionToolDefinition | Mapping[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        raise NotImplementedError

    def stream_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[CanonicalFunctionToolDefinition | Mapping[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        raise NotImplementedError

    def supports_structured_output(self) -> bool:
        return False

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMStructuredResult[Any]:
        raise NotImplementedError

    def supports_vision(self) -> bool:
        return False

    def supports_audio_input(self) -> bool:
        return False

    def supports_audio_output(self) -> bool:
        return False


class _ExternalManagedAdapter(_ExternalExecutionOnlyAdapter, _LifecycleBindingRecorder):
    provider = "external-managed"
    model = "ext-managed"


def _reset_recorder() -> None:
    _LifecycleBindingRecorder.dependency_boundary = None
    _LifecycleBindingRecorder.admission_gate = None
    _LifecycleBindingRecorder.external_ports = None


def _protocol_class_defs(scan_roots: Sequence[Path]) -> list[Path]:
    hits: list[Path] = []
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                raw = path.read_text(encoding="utf-8-sig")
                tree = ast.parse(raw)
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == "LLMRuntimeLifecycleBinding":
                    hits.append(path)
    return hits


def _lifecycle_composition_base_checks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    if "BaseLLMAdapter" in text:
        for line in text.splitlines():
            if "BaseLLMAdapter" in line and "isinstance" in line:
                problems.append(f"{path.name}: {line.strip()}")
    for token in ("hasattr(", "getattr(", "callable("):
        if token in text and "bind_" in text:
            problems.append(f"{path.name}: reflection token {token}")
    return problems


def test_ebh_2e_r1_r1_single_canonical_lifecycle_protocol() -> None:
    scan_roots = (_REPO_ROOT / "intergrax/llm_adapters/base",)
    defs = _protocol_class_defs(scan_roots)
    assert defs == [_CANONICAL_LIFECYCLE_PATH]
    canonical = importlib.import_module("intergrax.llm_adapters.base.lifecycle_binding")
    compat = importlib.import_module(
        "intergrax.llm_adapters.contracts.runtime_lifecycle_binding"
    )
    assert compat.LLMRuntimeLifecycleBinding is canonical.LLMRuntimeLifecycleBinding


def test_ebh_2e_r1_r1_lifecycle_protocol_is_runtime_checkable() -> None:
    managed = _ExternalManagedAdapter()
    assert isinstance(managed, LLMRuntimeLifecycleBinding)
    exec_only = _ExternalExecutionOnlyAdapter()
    assert not isinstance(exec_only, LLMRuntimeLifecycleBinding)


def test_ebh_2e_r1_r1_lifecycle_composition_uses_contract_not_base() -> None:
    offenders: list[str] = []
    for path in _LIFECYCLE_COMPOSITION_PATHS:
        offenders.extend(_lifecycle_composition_base_checks(path))
    assert not offenders, "\n".join(offenders)


def test_ebh_2e_r1_r1_structural_managed_adapter_receives_dependency_boundary() -> None:
    _reset_recorder()
    boundary = DependencyAttemptExecutionBoundary(LocalDependencyConcurrencyAdmission({}))
    try:
        set_llm_provider_dependency_boundary(boundary)
        apply_llm_provider_dependency_boundary(_ExternalManagedAdapter())
        assert _LifecycleBindingRecorder.dependency_boundary is boundary
    finally:
        set_llm_provider_dependency_boundary(None)
        boundary.close()


def test_ebh_2e_r1_r1_structural_managed_adapter_receives_external_operation_ports() -> None:
    _reset_recorder()
    sentinel_cancel = object()
    sentinel_status = object()
    sentinel_termination = object()
    sentinel_registry = object()
    bind_llm_external_operation_ports(
        _ExternalManagedAdapter(),
        cancellation_port=sentinel_cancel,
        status_port=sentinel_status,
        termination_port=sentinel_termination,
        stream_registry=sentinel_registry,
    )
    ports = _LifecycleBindingRecorder.external_ports
    assert ports is not None
    assert ports["cancellation_port"] is sentinel_cancel
    assert ports["status_port"] is sentinel_status
    assert ports["termination_port"] is sentinel_termination
    assert ports["stream_registry"] is sentinel_registry


def test_ebh_2e_r1_r1_structural_managed_adapter_admission_gate_binding() -> None:
    _reset_recorder()
    gate = object()
    _ExternalManagedAdapter().bind_external_operation_admission_gate(gate)
    assert _LifecycleBindingRecorder.admission_gate is gate


def test_ebh_2e_r1_r1_base_llm_adapter_subclass_satisfies_lifecycle_contract() -> None:
    class _MinimalBaseAdapter(BaseLLMAdapter):
        provider = "openai"
        model = "m"

        @property
        def context_window_tokens(self) -> int:
            return 4096

        def generate_messages(
            self,
            messages: Sequence[ChatMessage],
            *,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            run_id: Optional[str] = None,
        ) -> LLMAdapterResponse:
            return build_adapter_response(content="x")

    adapter = _MinimalBaseAdapter()
    assert isinstance(adapter, LLMRuntimeLifecycleBinding)


def test_ebh_2e_r1_r1_execution_only_adapter_skips_optional_lifecycle_binding() -> None:
    _reset_recorder()
    adapter = _ExternalExecutionOnlyAdapter()
    assert isinstance(adapter, LLMAdapter)
    boundary = DependencyAttemptExecutionBoundary(LocalDependencyConcurrencyAdmission({}))
    try:
        set_llm_provider_dependency_boundary(boundary)
        apply_llm_provider_dependency_boundary(adapter)
        assert _LifecycleBindingRecorder.dependency_boundary is None
        bind_llm_external_operation_ports(adapter, cancellation_port=object(), status_port=object())
        assert _LifecycleBindingRecorder.external_ports is None
    finally:
        set_llm_provider_dependency_boundary(None)
        boundary.close()


def test_ebh_2e_r1_r1_registry_accepts_execution_only_structural_adapter() -> None:
    from tests.unit.llm_adapters.registry_state_test_support import (
        restore_registry_state,
        snapshot_registry_state,
    )

    snapshot = snapshot_registry_state()
    try:
        LLMAdapterRegistry.reset_for_testing()
        key = "ebh2e-r1-r1-exec-only"

        def _factory(**_kwargs: object) -> LLMAdapter:
            return _ExternalExecutionOnlyAdapter()

        LLMAdapterRegistry.register(key, _factory)
        created = LLMAdapterRegistry.create(key)
        assert isinstance(created, LLMAdapter)
        assert not isinstance(created, LLMRuntimeLifecycleBinding)
    finally:
        restore_registry_state(snapshot)


def test_ebh_2e_r1_r1_remaining_base_isinstance_in_llm_layer_classified() -> None:
    """Framework validation may keep BaseLLMAdapter checks; lifecycle paths may not."""
    offenders: list[str] = []
    llm_root = _REPO_ROOT / "intergrax/llm_adapters"
    for path in llm_root.rglob("*.py"):
        if path in _BASE_LIFECYCLE_ISINSTANCE_ALLOWLIST:
            continue
        if "providers" in path.parts or path.parts[-1] == "base_llm_adapter.py":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if "isinstance" in line and "BaseLLMAdapter" in line:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}: {line.strip()}")
    assert offenders == []
