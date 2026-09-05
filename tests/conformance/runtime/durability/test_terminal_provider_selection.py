# © Artur Czarnecki. All rights reserved.

"""P0C-8B — explicit execution terminal provider selection at composition."""

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
from intergrax.contracts.attempt_lifecycle import AttemptLifecyclePersistenceProvider
from intergrax.contracts.execution_terminal import (
    AmbiguousExecutionTerminalProviderError,
    ExecutionTerminalError,
    ExecutionTerminalPersistenceProvider,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService
from intergrax.runtime.execution.execution_terminal.persistence import (
    CheckpointStoreExecutionTerminalStore,
    DocumentStoreExecutionTerminalStore,
    InMemoryExecutionTerminalStore,
    KvExecutionTerminalStore,
)
from intergrax.runtime.execution.execution_terminal.wiring import resolve_execution_terminal_store
from intergrax.runtime.registry.agent_registry import AgentRegistry

from tests.unit.runtime.background_execution.reentry_admission_doubles import InMemoryKVStore

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSITION_ROOTS = (
    _REPO_ROOT / "intergrax/applications/_shared/nexus_factory.py",
)


def _env(
    *,
    provider: ExecutionTerminalPersistenceProvider | None = None,
    attempt_provider: AttemptLifecyclePersistenceProvider | None = None,
) -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.lab_defaults(profile_id="p0c8b.terminal.provider").model_copy(
        update={
            "reliability_profile": ReliabilityProfile(
                execution_terminal_persistence_provider=provider,
                attempt_lifecycle_persistence_provider=attempt_provider,
            ),
        },
    )


def test_ambiguous_multi_provider_fails_closed_without_selector() -> None:
    with pytest.raises(AmbiguousExecutionTerminalProviderError):
        resolve_execution_terminal_store(
            kv_store=InMemoryKVStore(),
            document_store=InMemoryDocumentStore(),
        )


def test_explicit_kv_provider_selects_kv_terminal_store() -> None:
    store = resolve_execution_terminal_store(
        provider=ExecutionTerminalPersistenceProvider.KV,
        kv_store=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
    )
    assert isinstance(store, KvExecutionTerminalStore)


def test_explicit_document_store_provider_selects_document_terminal_store() -> None:
    store = resolve_execution_terminal_store(
        provider=ExecutionTerminalPersistenceProvider.DOCUMENT_STORE,
        kv_store=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
    )
    assert isinstance(store, DocumentStoreExecutionTerminalStore)


def test_missing_selected_capability_fails_closed() -> None:
    with pytest.raises(ExecutionTerminalError, match="document_store"):
        resolve_execution_terminal_store(
            provider=ExecutionTerminalPersistenceProvider.DOCUMENT_STORE,
            kv_store=InMemoryKVStore(),
        )


def test_explicit_execution_terminal_service_wins_over_ambiguous_platform_stores() -> None:
    custom = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_env(
            provider=ExecutionTerminalPersistenceProvider.KV,
            attempt_provider=AttemptLifecyclePersistenceProvider.KV,
        ),
        key_value_cache=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
        execution_terminal=custom,
    )
    assert loop.execution_terminal is custom


def test_explicit_execution_terminal_store_wins_over_ambiguous_platform_stores() -> None:
    custom_store = InMemoryExecutionTerminalStore()
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_env(
            provider=ExecutionTerminalPersistenceProvider.KV,
            attempt_provider=AttemptLifecyclePersistenceProvider.KV,
        ),
        key_value_cache=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
        execution_terminal_store=custom_store,
    )
    assert loop.execution_terminal is not None
    assert loop.execution_terminal.store is custom_store


def test_single_available_provider_auto_selects() -> None:
    store = resolve_execution_terminal_store(kv_store=InMemoryKVStore())
    assert isinstance(store, KvExecutionTerminalStore)


def test_nexus_factory_with_kv_and_document_store_and_explicit_provider() -> None:
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_env(
            provider=ExecutionTerminalPersistenceProvider.KV,
            attempt_provider=AttemptLifecyclePersistenceProvider.KV,
        ),
        key_value_cache=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
    )
    assert loop.execution_terminal is not None
    assert isinstance(loop.execution_terminal.store, KvExecutionTerminalStore)


def test_queue_worker_environment_profile_sets_kv_terminal_provider() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="queue.host")
    env = apply_queue_worker_environment_profile(env)
    assert (
        env.reliability_profile.execution_terminal_persistence_provider
        is ExecutionTerminalPersistenceProvider.KV
    )


def test_checkpoint_provider_selects_checkpoint_terminal_store(tmp_path: Path) -> None:
    from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore

    checkpoint_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "checkpoint-terminal.db")
    store = resolve_execution_terminal_store(
        provider=ExecutionTerminalPersistenceProvider.CHECKPOINT,
        kv_store=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
        checkpoint_store=checkpoint_store,
    )
    assert isinstance(store, CheckpointStoreExecutionTerminalStore)


def test_nexus_factory_ambiguous_platform_stores_fail_closed() -> None:
    from intergrax.contracts.attempt_lifecycle import AttemptLifecyclePersistenceProvider

    with pytest.raises(AmbiguousExecutionTerminalProviderError):
        build_nexus_loop_from_environment(
            AgentRegistry(),
            env=_env(
                provider=None,
            ).model_copy(
                update={
                    "reliability_profile": ReliabilityProfile(
                        attempt_lifecycle_persistence_provider=(
                            AttemptLifecyclePersistenceProvider.KV
                        ),
                    ),
                },
            ),
            key_value_cache=InMemoryKVStore(),
            document_store=InMemoryDocumentStore(),
        )


def test_production_host_path_with_kv_document_store_and_provider() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="legal.queue.host")
    env = apply_queue_worker_environment_profile(env)
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=env,
        key_value_cache=InMemoryKVStore(),
        document_store=InMemoryDocumentStore(),
    )
    assert isinstance(loop.execution_terminal.store, KvExecutionTerminalStore)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _provider_keyword_count(call: ast.Call) -> int:
    provider_keywords = {"kv_store", "document_store", "checkpoint_store"}
    return sum(1 for keyword in call.keywords if keyword.arg in provider_keywords)


def test_composition_roots_do_not_forward_multiple_terminal_providers() -> None:
    violations: list[str] = []
    for path in _COMPOSITION_ROOTS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node.func) != "wire_execution_terminal_store":
                continue
            if _provider_keyword_count(node) > 1:
                violations.append(f"{rel}:{node.lineno}")
    assert violations == []
