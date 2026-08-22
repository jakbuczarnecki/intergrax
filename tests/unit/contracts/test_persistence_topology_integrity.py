# © Artur Czarnecki. All rights reserved.

"""PCM-PERSISTENCE-TOPOLOGY-INTEGRITY — topology qualification tests (R3-A)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import pytest
from pydantic import BaseModel, ValidationError

from intergrax.applications._shared.reliability_assembly_resolver import (
    assert_reliability_assembly_valid,
    validate_reliability_wiring,
)
from intergrax.applications._shared.reliability_wiring import (
    ApplicationReliabilityWiring,
    wire_application_reliability,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ReliabilityProfile,
)
from intergrax.applications.contracts.environment_profile.bundles import HostMeta
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.idempotency_store import (
    ClaimResult,
    IdempotencyStore,
    InvocationClaim,
    InvocationStatus,
)
from intergrax.contracts.persistence_topology import (
    DeploymentTopology,
    PersistenceTopology,
    required_persistence_for_deployment,
    resolve_idempotency_store_topology,
)
from intergrax.distributed.providers.redis_idempotency_store import RedisIdempotencyStore
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore
from intergrax.tools.execution_models import ToolExecutionResult

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class _DummyOutput(BaseModel):
    value: str


class _UndeclaredTopologyIdempotencyStore(IdempotencyStore):
    """Test double with non-canonical topology declaration."""

    @property
    def persistence_topology(self) -> Any:
        return "process_local"

    def get_status(self, tenant_id: str, key: str) -> Optional[InvocationStatus]:
        return None

    def claim(
        self,
        tenant_id: str,
        key: str,
        owner_id: str,
        lease_seconds: int,
    ) -> ClaimResult:
        del tenant_id, key, owner_id, lease_seconds
        raise NotImplementedError

    def complete_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        del tenant_id, key, claim, result, completed_ttl_seconds
        raise NotImplementedError

    def record_started(
        self,
        tenant_id: str,
        key: str,
        lease_seconds: Optional[int] = None,
    ) -> None:
        return None

    def record_completed(
        self,
        tenant_id: str,
        key: str,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        return None

    def get_completed_result(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[ToolExecutionResult[BaseModel]]:
        return None


class _BrandNameOnlyIdempotencyStore(IdempotencyStore):
    """Provider name alone must not grant SHARED_MULTI_HOST."""

    @property
    def persistence_topology(self) -> Any:
        return None

    def get_status(self, tenant_id: str, key: str) -> Optional[InvocationStatus]:
        return None

    def claim(
        self,
        tenant_id: str,
        key: str,
        owner_id: str,
        lease_seconds: int,
    ) -> ClaimResult:
        del tenant_id, key, owner_id, lease_seconds
        raise NotImplementedError

    def complete_with_claim(
        self,
        tenant_id: str,
        key: str,
        claim: InvocationClaim,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        del tenant_id, key, claim, result, completed_ttl_seconds
        raise NotImplementedError

    def record_started(
        self,
        tenant_id: str,
        key: str,
        lease_seconds: Optional[int] = None,
    ) -> None:
        return None

    def record_completed(
        self,
        tenant_id: str,
        key: str,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        return None

    def get_completed_result(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[ToolExecutionResult[BaseModel]]:
        return None


class _DummyRelationalStore:
    def connect(self) -> None:
        return None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        return None

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        return ()

    def close(self) -> None:
        return None


_REQUIRED_TO_DEPLOYMENT: dict[PersistenceTopology, DeploymentTopology] = {
    PersistenceTopology.PROCESS_LOCAL: DeploymentTopology.PROCESS_LOCAL,
    PersistenceTopology.DURABLE_SINGLE_HOST: DeploymentTopology.SINGLE_HOST,
    PersistenceTopology.SHARED_MULTI_HOST: DeploymentTopology.MULTI_HOST,
}


def _env_with_deployment(
    deployment: DeploymentTopology,
    *,
    profile_id: str = "pcm.topology",
) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    env.meta = env.meta.model_copy(update={"deployment_topology": deployment})
    return env


def _env_with_topology(
    topology: PersistenceTopology,
    *,
    profile_id: str = "pcm.topology",
) -> ApplicationEnvironmentProfile:
    return _env_with_deployment(
        _REQUIRED_TO_DEPLOYMENT[topology],
        profile_id=profile_id,
    )


def _fake_redis_store() -> RedisIdempotencyStore:
    class _FakeRedis:
        def register_script(self, _script: str) -> object:
            return object()

    return RedisIdempotencyStore(_FakeRedis())


def _wiring_with_store(store: IdempotencyStore | None) -> ApplicationReliabilityWiring:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="pcm.wiring")
    wiring = wire_application_reliability(env)
    return ApplicationReliabilityWiring(
        options=wiring.options,
        idempotency_store=store,
        circuit_breaker_config=wiring.circuit_breaker_config,
    )


def test_inmemory_store_classified_process_local() -> None:
    store = InMemoryIdempotencyStore()
    assert store.persistence_topology is PersistenceTopology.PROCESS_LOCAL


def test_sqlite_store_classified_durable_single_host() -> None:
    store = SQLiteIdempotencyStore(":memory:")
    assert store.persistence_topology is PersistenceTopology.DURABLE_SINGLE_HOST


def test_redis_store_classified_shared_multi_host() -> None:
    store = _fake_redis_store()
    assert store.persistence_topology is PersistenceTopology.SHARED_MULTI_HOST


def test_durable_requirement_rejects_inmemory() -> None:
    env = _env_with_topology(PersistenceTopology.DURABLE_SINGLE_HOST)
    wiring = _wiring_with_store(InMemoryIdempotencyStore())
    result = validate_reliability_wiring(wiring, env)
    assert not result.valid
    assert any("idempotency persistence topology mismatch" in e for e in result.errors)
    assert any("required=durable_single_host provided=process_local" in e for e in result.errors)


def test_durable_requirement_accepts_sqlite(tmp_path) -> None:
    env = _env_with_topology(PersistenceTopology.DURABLE_SINGLE_HOST)
    wiring = wire_application_reliability(env, idempotency_db_path=tmp_path / "idempotency.db")
    result = validate_reliability_wiring(wiring, env)
    assert result.valid


def test_shared_requirement_rejects_inmemory() -> None:
    env = _env_with_topology(PersistenceTopology.SHARED_MULTI_HOST)
    wiring = _wiring_with_store(InMemoryIdempotencyStore())
    result = validate_reliability_wiring(wiring, env)
    assert not result.valid


def test_shared_requirement_rejects_sqlite(tmp_path) -> None:
    env = _env_with_topology(PersistenceTopology.SHARED_MULTI_HOST)
    wiring = wire_application_reliability(env, idempotency_db_path=tmp_path / "idempotency.db")
    result = validate_reliability_wiring(wiring, env)
    assert not result.valid
    assert any(
        "required=shared_multi_host provided=durable_single_host" in error
        for error in result.errors
    )


def test_unknown_store_capability_fails_closed() -> None:
    env = _env_with_topology(PersistenceTopology.DURABLE_SINGLE_HOST)
    store = _UndeclaredTopologyIdempotencyStore()
    assert resolve_idempotency_store_topology(store) is None
    wiring = _wiring_with_store(store)
    result = validate_reliability_wiring(wiring, env)
    assert not result.valid
    assert any("provided=unknown" in error for error in result.errors)


def test_process_local_lab_remains_supported() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="pcm.lab")
    assert env.meta.required_persistence_topology is PersistenceTopology.PROCESS_LOCAL
    wiring = wire_application_reliability(env)
    assert_reliability_assembly_valid(wiring, env)


def test_strictness_does_not_imply_shared_topology() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="pcm.strict")
    env.execution_mode = ExecutionMode.STRICT
    assert env.meta.required_persistence_topology is PersistenceTopology.PROCESS_LOCAL
    wiring = wire_application_reliability(env)
    assert_reliability_assembly_valid(wiring, env)

    product = ApplicationEnvironmentProfile.product_defaults(profile_id="pcm.product")
    assert product.execution_mode is ExecutionMode.STRICT
    assert product.meta.required_persistence_topology is PersistenceTopology.DURABLE_SINGLE_HOST
    assert product.meta.required_persistence_topology is not PersistenceTopology.SHARED_MULTI_HOST


def test_generic_relational_store_is_not_domain_qualification() -> None:
    store = _DummyRelationalStore()
    assert isinstance(store, RelationalStore)
    assert resolve_idempotency_store_topology(store) is None  # type: ignore[arg-type]


def test_provider_name_does_not_grant_capability() -> None:
    postgres_named = _BrandNameOnlyIdempotencyStore()
    assert resolve_idempotency_store_topology(postgres_named) is None
    env = _env_with_topology(PersistenceTopology.SHARED_MULTI_HOST)
    wiring = _wiring_with_store(postgres_named)
    result = validate_reliability_wiring(wiring, env)
    assert not result.valid


def test_topology_mismatch_error_evidence_without_secrets() -> None:
    env = _env_with_topology(PersistenceTopology.SHARED_MULTI_HOST)
    wiring = _wiring_with_store(InMemoryIdempotencyStore())
    result = validate_reliability_wiring(wiring, env)
    assert len(result.errors) == 1
    error = result.errors[0]
    assert "idempotency persistence topology mismatch" in error
    assert "required=shared_multi_host" in error
    assert "provided=process_local" in error
    assert "redis://" not in error
    assert "postgres" not in error.lower()


def test_r1_1_multi_host_automatically_requires_shared() -> None:
    meta = HostMeta.product(deployment_topology=DeploymentTopology.MULTI_HOST)
    assert meta.deployment_topology is DeploymentTopology.MULTI_HOST
    assert meta.required_persistence_topology is PersistenceTopology.SHARED_MULTI_HOST
    assert (
        required_persistence_for_deployment(DeploymentTopology.MULTI_HOST)
        is PersistenceTopology.SHARED_MULTI_HOST
    )


def test_r1_2_multi_host_sqlite_fails_assembly(tmp_path) -> None:
    env = _env_with_deployment(DeploymentTopology.MULTI_HOST, profile_id="pcm.r1.2")
    assert env.reliability_profile.idempotency_enabled
    assert env.meta.required_persistence_topology is PersistenceTopology.SHARED_MULTI_HOST
    wiring = wire_application_reliability(env, idempotency_db_path=tmp_path / "idempotency.db")
    result = validate_reliability_wiring(wiring, env)
    assert not result.valid
    assert any(
        "required=shared_multi_host provided=durable_single_host" in error
        for error in result.errors
    )


def test_r1_3_multi_host_inmemory_fails_assembly() -> None:
    env = _env_with_deployment(DeploymentTopology.MULTI_HOST, profile_id="pcm.r1.3")
    wiring = _wiring_with_store(InMemoryIdempotencyStore())
    result = validate_reliability_wiring(wiring, env)
    assert not result.valid
    assert any("required=shared_multi_host provided=process_local" in error for error in result.errors)


def test_r1_4_multi_host_redis_passes_topology_gate() -> None:
    env = _env_with_deployment(DeploymentTopology.MULTI_HOST, profile_id="pcm.r1.4")
    wiring = _wiring_with_store(_fake_redis_store())
    result = validate_reliability_wiring(wiring, env)
    assert result.valid


def test_r1_5_single_host_product_sqlite_passes(tmp_path) -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="pcm.r1.5")
    assert env.meta.deployment_topology is DeploymentTopology.SINGLE_HOST
    assert env.meta.required_persistence_topology is PersistenceTopology.DURABLE_SINGLE_HOST
    wiring = wire_application_reliability(env, idempotency_db_path=tmp_path / "idempotency.db")
    result = validate_reliability_wiring(wiring, env)
    assert result.valid


def test_product_wiring_auto_selects_sqlite_without_explicit_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INTERGRAX_SQLITE_DATA_DIR", str(tmp_path))
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="pcm.product.auto")
    wiring = wire_application_reliability(env)
    assert isinstance(wiring.idempotency_store, SQLiteIdempotencyStore)
    assert_reliability_assembly_valid(wiring, env)


def test_product_wiring_does_not_use_inmemory_without_explicit_path() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="pcm.product.no_mem")
    wiring = wire_application_reliability(env)
    assert not isinstance(wiring.idempotency_store, InMemoryIdempotencyStore)


def test_r1_6_single_host_product_inmemory_fails() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="pcm.r1.6")
    wiring = _wiring_with_store(InMemoryIdempotencyStore())
    result = validate_reliability_wiring(wiring, env)
    assert not result.valid
    assert any("required=durable_single_host provided=process_local" in error for error in result.errors)


def test_r1_7_strict_single_host_does_not_require_shared() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="pcm.r1.7")
    assert env.execution_mode is ExecutionMode.STRICT
    assert env.meta.execution_mode is ExecutionMode.STRICT
    assert env.meta.deployment_topology is DeploymentTopology.SINGLE_HOST
    assert env.meta.required_persistence_topology is PersistenceTopology.DURABLE_SINGLE_HOST
    assert env.meta.required_persistence_topology is not PersistenceTopology.SHARED_MULTI_HOST


def test_r1_8_balanced_multi_host_still_requires_shared(tmp_path) -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="pcm.r1.8")
    env.execution_mode = ExecutionMode.BALANCED
    env.meta = env.meta.model_copy(
        update={
            "execution_mode": ExecutionMode.BALANCED,
            "deployment_topology": DeploymentTopology.MULTI_HOST,
        },
    )
    assert env.execution_mode is ExecutionMode.BALANCED
    assert env.meta.required_persistence_topology is PersistenceTopology.SHARED_MULTI_HOST
    wiring = wire_application_reliability(env, idempotency_db_path=tmp_path / "idempotency.db")
    result = validate_reliability_wiring(wiring, env)
    assert not result.valid
    assert any(
        "required=shared_multi_host provided=durable_single_host" in error
        for error in result.errors
    )


def test_r1_9_contradictory_config_impossible() -> None:
    with pytest.raises(ValidationError, match="required_persistence_topology contradicts deployment_topology"):
        HostMeta(
            deployment_topology=DeploymentTopology.MULTI_HOST,
            required_persistence_topology=PersistenceTopology.DURABLE_SINGLE_HOST,
        )
    meta = HostMeta.product(deployment_topology=DeploymentTopology.MULTI_HOST)
    with pytest.raises(ValidationError, match="required_persistence_topology contradicts deployment_topology"):
        meta.model_copy(
            update={"required_persistence_topology": PersistenceTopology.DURABLE_SINGLE_HOST},
        )
