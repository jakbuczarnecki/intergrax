# © Artur Czarnecki. All rights reserved.

"""P0C-8C — independent attempt lifecycle provider selection at composition."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications._shared.host_queue_execution_wiring import (
    apply_queue_worker_environment_profile,
)
from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ReliabilityProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.attempt_lifecycle import (
    AmbiguousAttemptLifecycleProviderError,
    AttemptLifecycleError,
    AttemptLifecyclePersistenceProvider,
    AttemptLifecycleStore,
)
from intergrax.contracts.execution_terminal import (
    AmbiguousExecutionTerminalProviderError,
    ExecutionTerminalPersistenceProvider,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.attempt_lifecycle import (
    InMemoryAttemptLifecycleStore,
    KvAttemptLifecycleStore,
    resolve_attempt_lifecycle_store,
)
from intergrax.runtime.execution.attempt_lifecycle.persistence import (
    DocumentStoreAttemptLifecycleStore,
)
from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService
from intergrax.runtime.execution.execution_terminal.persistence import (
    DocumentStoreExecutionTerminalStore,
    InMemoryExecutionTerminalStore,
    KvExecutionTerminalStore,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.registry.agent_registry import AgentRegistry

from tests.unit.runtime.background_execution.reentry_admission_doubles import InMemoryKVStore

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSITION_ROOTS = (
    _REPO_ROOT / "intergrax/applications/_shared/nexus_factory.py",
)
_FORBIDDEN_ATTEMPT_RESOLUTION_NAMES = frozenset(
    {
        "ExecutionTerminalPersistenceProvider",
        "execution_terminal_persistence_provider",
        "resolve_execution_terminal_provider",
        "resolve_platform_store_for_terminal_provider",
    },
)


def _env(
    *,
    terminal_provider: ExecutionTerminalPersistenceProvider | None = None,
    attempt_provider: AttemptLifecyclePersistenceProvider | None = None,
) -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.lab_defaults(profile_id="p0c8c.attempt.provider").model_copy(
        update={
            "reliability_profile": ReliabilityProfile(
                execution_terminal_persistence_provider=terminal_provider,
                attempt_lifecycle_persistence_provider=attempt_provider,
            ),
        },
    )


def test_decoupled_terminal_kv_attempt_document_store() -> None:
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_env(
            terminal_provider=ExecutionTerminalPersistenceProvider.KV,
            attempt_provider=AttemptLifecyclePersistenceProvider.DOCUMENT_STORE,
        ),
        key_value_cache=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
    )
    assert loop.execution_terminal is not None
    assert isinstance(loop.execution_terminal.store, KvExecutionTerminalStore)
    assert loop._attempt_lifecycle is not None  # noqa: SLF001
    assert isinstance(loop._attempt_lifecycle.store, DocumentStoreAttemptLifecycleStore)  # noqa: SLF001


def test_reverse_decoupled_terminal_document_store_attempt_kv() -> None:
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_env(
            terminal_provider=ExecutionTerminalPersistenceProvider.DOCUMENT_STORE,
            attempt_provider=AttemptLifecyclePersistenceProvider.KV,
        ),
        key_value_cache=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
    )
    assert loop.execution_terminal is not None
    assert isinstance(loop.execution_terminal.store, DocumentStoreExecutionTerminalStore)
    assert loop._attempt_lifecycle is not None  # noqa: SLF001
    assert isinstance(loop._attempt_lifecycle.store, KvAttemptLifecycleStore)  # noqa: SLF001


def test_custom_terminal_service_with_attempt_kv() -> None:
    custom = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_env(attempt_provider=AttemptLifecyclePersistenceProvider.KV),
        key_value_cache=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
        execution_terminal=custom,
    )
    assert loop.execution_terminal is custom
    assert loop._attempt_lifecycle is not None  # noqa: SLF001
    assert isinstance(loop._attempt_lifecycle.store, KvAttemptLifecycleStore)  # noqa: SLF001


def test_custom_terminal_store_with_attempt_document_store() -> None:
    custom_store = InMemoryExecutionTerminalStore()
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_env(attempt_provider=AttemptLifecyclePersistenceProvider.DOCUMENT_STORE),
        key_value_cache=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
        execution_terminal_store=custom_store,
    )
    assert loop.execution_terminal is not None
    assert loop.execution_terminal.store is custom_store
    assert loop._attempt_lifecycle is not None  # noqa: SLF001
    assert isinstance(loop._attempt_lifecycle.store, DocumentStoreAttemptLifecycleStore)  # noqa: SLF001


def test_attempt_ambiguity_fails_closed_even_when_terminal_selector_is_kv() -> None:
    with pytest.raises(AmbiguousAttemptLifecycleProviderError):
        build_nexus_loop_from_environment(
            AgentRegistry(),
            env=_env(terminal_provider=ExecutionTerminalPersistenceProvider.KV),
            key_value_cache=InMemoryKVStore(),
            document_store=InMemoryDocumentStore(),
        )


def test_terminal_ambiguity_independent_when_attempt_selector_is_kv() -> None:
    with pytest.raises(AmbiguousExecutionTerminalProviderError):
        build_nexus_loop_from_environment(
            AgentRegistry(),
            env=_env(attempt_provider=AttemptLifecyclePersistenceProvider.KV),
            key_value_cache=InMemoryKVStore(),
            document_store=InMemoryDocumentStore(),
        )


def test_same_provider_selected_independently() -> None:
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_env(
            terminal_provider=ExecutionTerminalPersistenceProvider.KV,
            attempt_provider=AttemptLifecyclePersistenceProvider.KV,
        ),
        key_value_cache=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
    )
    assert isinstance(loop.execution_terminal.store, KvExecutionTerminalStore)
    assert isinstance(loop._attempt_lifecycle.store, KvAttemptLifecycleStore)  # noqa: SLF001


def test_queue_worker_environment_profile_sets_independent_kv_providers() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="queue.host.p0c8c")
    env = apply_queue_worker_environment_profile(env)
    assert (
        env.reliability_profile.execution_terminal_persistence_provider
        is ExecutionTerminalPersistenceProvider.KV
    )
    assert (
        env.reliability_profile.attempt_lifecycle_persistence_provider
        is AttemptLifecyclePersistenceProvider.KV
    )
    assert (
        env.reliability_profile.execution_terminal_persistence_provider
        is not env.reliability_profile.attempt_lifecycle_persistence_provider
    )


def test_production_strict_nexus_with_custom_terminal_and_durable_attempt_kv() -> None:
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="p0c8c.production")
    env = base.model_copy(
        update={
            "meta": base.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT}),
            "reliability_profile": ReliabilityProfile(
                attempt_lifecycle_persistence_provider=AttemptLifecyclePersistenceProvider.KV,
                execution_terminal_persistence_provider=ExecutionTerminalPersistenceProvider.KV,
            ),
        },
    )
    custom = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=env,
        key_value_cache=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
        execution_terminal=custom,
        run_budget=RunBudget(),
    )
    assert loop.execution_terminal is custom
    assert isinstance(loop._attempt_lifecycle.store, KvAttemptLifecycleStore)  # noqa: SLF001


def test_missing_attempt_capability_fails_closed() -> None:
    with pytest.raises(AttemptLifecycleError, match="document_store"):
        resolve_attempt_lifecycle_store(
            provider=AttemptLifecyclePersistenceProvider.DOCUMENT_STORE,
            kv_store=InMemoryKVStore(),
        )


def test_explicit_attempt_store_bypasses_ambiguity() -> None:
    custom_attempt: AttemptLifecycleStore = InMemoryAttemptLifecycleStore()
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_env(terminal_provider=ExecutionTerminalPersistenceProvider.KV),
        key_value_cache=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
        attempt_lifecycle_store=custom_attempt,
    )
    assert loop._attempt_lifecycle is not None  # noqa: SLF001
    assert loop._attempt_lifecycle.store is custom_attempt  # noqa: SLF001
    assert isinstance(loop.execution_terminal.store, KvExecutionTerminalStore)


def test_single_available_attempt_provider_auto_selects() -> None:
    store = resolve_attempt_lifecycle_store(kv_store=InMemoryKVStore())
    assert isinstance(store, KvAttemptLifecycleStore)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _provider_keyword_count(call: ast.Call) -> int:
    provider_keywords = {"kv_store", "document_store"}
    return sum(1 for keyword in call.keywords if keyword.arg in provider_keywords)


def _attempt_resolution_block(source: str) -> ast.AST:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) != "resolve_attempt_lifecycle_store":
            continue
        return node
    raise AssertionError("resolve_attempt_lifecycle_store call not found")


def test_composition_roots_do_not_couple_attempt_selection_to_terminal_provider() -> None:
    path = _COMPOSITION_ROOTS[0]
    source = path.read_text(encoding="utf-8")
    block = _attempt_resolution_block(source)
    for node in ast.walk(block):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_ATTEMPT_RESOLUTION_NAMES:
            pytest.fail(
                f"{path.relative_to(_REPO_ROOT)} attempt resolution references terminal "
                f"selector symbol {node.id}",
            )
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTEMPT_RESOLUTION_NAMES:
            pytest.fail(
                f"{path.relative_to(_REPO_ROOT)} attempt resolution references terminal "
                f"selector symbol {node.attr}",
            )


def test_composition_roots_do_not_forward_multiple_attempt_providers() -> None:
    violations: list[str] = []
    for path in _COMPOSITION_ROOTS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node.func) != "wire_attempt_lifecycle_store":
                continue
            if _provider_keyword_count(node) > 1:
                violations.append(f"{rel}:{node.lineno}")
    assert violations == []
