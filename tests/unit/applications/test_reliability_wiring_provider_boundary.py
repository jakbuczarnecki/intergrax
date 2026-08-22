# © Artur Czarnecki. All rights reserved.

"""WIN-FIX-2A3 — reliability wiring provider boundary regression guards."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications._shared.reliability_assembly_resolver import (
    assert_reliability_assembly_valid,
    validate_reliability_wiring,
)
from intergrax.applications._shared.reliability_wiring import wire_application_reliability
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ReliabilityProfile,
)
from intergrax.applications.contracts.environment_profile.bundles import HostMeta
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.contracts.persistence_topology import (
    DeploymentTopology,
    PersistenceTopology,
    resolve_idempotency_store_topology,
)
from intergrax.distributed.providers.redis_idempotency_store import RedisIdempotencyStore
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.reference_idempotency_store import (
    resolve_reference_idempotency_store,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

REPO = Path(__file__).resolve().parents[3]
RELIABILITY_WIRING_SOURCE = (
    REPO / "intergrax" / "applications" / "_shared" / "reliability_wiring.py"
)

_FORBIDDEN_RELIABILITY_IMPORTS = frozenset(
    {
        "SQLiteIdempotencyStore",
        "InMemoryIdempotencyStore",
        "RedisIdempotencyStore",
        "resolve_idempotency_db_path",
        "create_sqlite_idempotency_store",
    }
)


def _imported_names(module_ast: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(module_ast):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
    return names


def test_reliability_wiring_does_not_import_concrete_idempotency_providers() -> None:
    source = RELIABILITY_WIRING_SOURCE.read_text(encoding="utf-8")
    module_ast = ast.parse(source)
    imported = _imported_names(module_ast)
    leaked = imported & _FORBIDDEN_RELIABILITY_IMPORTS
    assert not leaked, f"reliability_wiring leaked provider imports: {sorted(leaked)}"


def test_resolve_reference_process_local_returns_process_local_store() -> None:
    store = resolve_reference_idempotency_store(PersistenceTopology.PROCESS_LOCAL)
    assert store is not None
    assert resolve_idempotency_store_topology(store) is PersistenceTopology.PROCESS_LOCAL


def test_resolve_reference_durable_single_host_returns_durable_store(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INTERGRAX_SQLITE_DATA_DIR", str(tmp_path))
    store = resolve_reference_idempotency_store(PersistenceTopology.DURABLE_SINGLE_HOST)
    assert store is not None
    assert resolve_idempotency_store_topology(store) is PersistenceTopology.DURABLE_SINGLE_HOST


def test_resolve_reference_shared_multi_host_returns_none_without_injection() -> None:
    store = resolve_reference_idempotency_store(PersistenceTopology.SHARED_MULTI_HOST)
    assert store is None


def test_resolve_reference_shared_multi_host_with_db_path_returns_none(tmp_path) -> None:
    db_path = tmp_path / "must_not_create.db"
    store = resolve_reference_idempotency_store(
        PersistenceTopology.SHARED_MULTI_HOST,
        db_path=db_path,
    )
    assert store is None
    assert not db_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_resolve_reference_durable_single_host_with_explicit_db_path(tmp_path) -> None:
    db_path = tmp_path / "idempotency.db"
    store = resolve_reference_idempotency_store(
        PersistenceTopology.DURABLE_SINGLE_HOST,
        db_path=db_path,
    )
    assert store is not None
    assert resolve_idempotency_store_topology(store) is PersistenceTopology.DURABLE_SINGLE_HOST
    assert db_path.exists()


def test_resolve_reference_process_local_ignores_db_path(tmp_path) -> None:
    db_path = tmp_path / "ignored.db"
    store = resolve_reference_idempotency_store(
        PersistenceTopology.PROCESS_LOCAL,
        db_path=db_path,
    )
    assert store is not None
    assert resolve_idempotency_store_topology(store) is PersistenceTopology.PROCESS_LOCAL
    assert not db_path.exists()


def test_wire_application_reliability_accepts_injected_shared_store() -> None:
    class _FakeRedis:
        def register_script(self, _script: str) -> object:
            return object()

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rel.inject.redis")
    env.meta = env.meta.model_copy(
        update={"deployment_topology": DeploymentTopology.MULTI_HOST},
    )
    injected = RedisIdempotencyStore(_FakeRedis())
    wiring = wire_application_reliability(env, idempotency_store=injected)
    assert wiring.idempotency_store is injected
    assert_reliability_assembly_valid(wiring, env)


def test_wire_application_reliability_shared_without_injection_fails_closed() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rel.shared.none")
    env.meta = env.meta.model_copy(
        update={"deployment_topology": DeploymentTopology.MULTI_HOST},
    )
    wiring = wire_application_reliability(env)
    assert wiring.idempotency_store is None
    result = validate_reliability_wiring(wiring, env)
    assert not result.valid


def test_wire_application_reliability_rejects_insufficient_injected_topology() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="rel.inject.insufficient")
    wiring = wire_application_reliability(
        env,
        idempotency_store=InMemoryIdempotencyStore(),
    )
    result = validate_reliability_wiring(wiring, env)
    assert not result.valid
    assert any("required=durable_single_host provided=process_local" in error for error in result.errors)


def test_wire_application_reliability_disabled_ignores_injected_store() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rel.inject.disabled")
    env.reliability_profile = ReliabilityProfile(idempotency_enabled=False)
    wiring = wire_application_reliability(
        env,
        idempotency_store=InMemoryIdempotencyStore(),
    )
    assert wiring.idempotency_store is None
