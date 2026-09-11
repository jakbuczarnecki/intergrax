# © Artur Czarnecki. All rights reserved.

"""W2-A — dependency isolation inventory gates (no production behavior change)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTEGRATIONS = _REPO_ROOT / "intergrax" / "integrations"
_RUNTIME_TOOLS = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"
_LOCAL_CAPACITY = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "local_execution_capacity_admission.py"
)


def _py_files_under(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if p.is_file()]


def test_integration_circuit_breaker_registry_not_used_as_tool_bulkhead() -> None:
    """Slug breaker is for integration health/resolve — not Nexus tool invocation."""
    invoker_source = _RUNTIME_TOOLS.read_text(encoding="utf-8")
    assert "get_breaker_for_slug" not in invoker_source
    assert "IntegrationCircuitBreaker" not in invoker_source


def test_integration_breaker_call_sites_are_bounded() -> None:
    """Production breaker.call paths: health probes, RAG vector wrapper — not generic tools."""
    allowed_fragments = (
        "circuit_breaker_registry.py",
        "circuit_breaker.py",
        "health.py",
        "vector_store_circuit_breaker.py",
        "retriever_engine.py",
    )
    for path in _py_files_under(_INTEGRATIONS):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if any(rel.endswith(fragment) for fragment in allowed_fragments):
            continue
        text = path.read_text(encoding="utf-8")
        assert "get_breaker_for_slug" not in text, f"unexpected breaker registry use: {rel}"
        assert "IntegrationCircuitBreaker(" not in text, f"unexpected breaker ctor: {rel}"


def test_runtime_tool_invoker_uses_shared_default_thread_pool() -> None:
    source = _RUNTIME_TOOLS.read_text(encoding="utf-8")
    assert "self._execution_pool = ThreadPoolExecutor()" in source
    assert "ThreadPoolExecutor(max_workers" not in source


def test_local_root_capacity_admission_ignores_tenant_for_slots() -> None:
    source = _LOCAL_CAPACITY.read_text(encoding="utf-8")
    assert "tenant_id" not in source


def test_llm_default_call_config_disables_provider_circuit_and_retry() -> None:
    from intergrax.llm_adapters._shared.call_config import LLMCallConfig

    cfg = LLMCallConfig()
    assert cfg.max_retries == 0
    assert cfg.circuit_breaker_threshold == 0


_LLM_ADAPTER = _REPO_ROOT / "intergrax" / "llm_adapters" / "contracts" / "llm_adapter.py"
_LLM_PROVIDERS = _REPO_ROOT / "intergrax" / "llm_adapters" / "providers"


def test_llm_adapter_exposes_provider_dependency_seams() -> None:
    source = _LLM_ADAPTER.read_text(encoding="utf-8")
    assert "def _execute(" in source
    assert "def _execute_streaming(" in source
    assert "def _provider_dependency_attempt(" in source
    assert "LLM_PROVIDER" in source


def test_builtin_provider_network_calls_use_canonical_execute_seams() -> None:
    """Unary/stream creation must route through LLMAdapter execute seams (W2-B3)."""
    allowed_direct_sdk = (
        "gemini_adapter.py",
        "openai_responses_adapter.py",
    )
    for path in _py_files_under(_LLM_PROVIDERS):
        if path.name.startswith("_"):
            continue
        if path.name in allowed_direct_sdk:
            continue
        text = path.read_text(encoding="utf-8")
        if "stream_messages" not in text and "stream_with_tools" not in text:
            continue
        if "self._execute_streaming(" in text or "self._provider_dependency_attempt(" in text:
            continue
        if "self._execute(" not in text:
            continue
        assert False, (
            f"{path.relative_to(_REPO_ROOT).as_posix()} streaming must use "
            "_execute_streaming or _provider_dependency_attempt"
        )
