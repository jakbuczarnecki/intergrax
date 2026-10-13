# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-PIS-02 — lifecycle-owned SQLite ProviderInvocationStore binding."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from applications.governed_contractor_application.host.provider_invocation_lifecycle import (
    GovernedProviderInvocationDispatchGate,
)
from governed_contractor_application.host.agent_builders import (
    build_governed_contractor_agent_builders,
)
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    compose_governed_contractor_host_runtime,
)
from governed_contractor_application.host.main import create_governed_contractor_process_app
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.manifest import build_governed_contractor_manifest
from intergrax.applications._shared.harness_host_orchestration_topology_wiring import (
    HarnessHostOrchestrationTopologyReliabilityCompositionError,
)
from intergrax.applications._shared.production_agent_platform_runtime import (
    build_production_agent_platform_runtime,
)
from intergrax.applications._shared.production_execution_continuation_store import (
    open_production_execution_continuation_state_store,
)
from intergrax.applications._shared.production_platform_persistence import (
    build_reference_production_platform_persistence,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.reference_production_process_lifecycle import (
    ReferenceProductionProcessLifecycleError,
    open_reference_production_process_lifecycle,
)
from intergrax.contracts.execution_continuation import ExecutionContinuationError
from intergrax.contracts.provider_invocation import ProviderInvocation
from intergrax.contracts.provider_invocation_store import (
    ProviderInvocationPersistenceError,
    ProviderInvocationStore,
)
from intergrax.runtime.enterprise_reliability.sqlite_provider_invocation_store import (
    SQLiteProviderInvocationStore,
    open_provider_invocation_store_at,
)
from intergrax.runtime.execution.continuation.durable_serialized_seal_store import (
    SerializedSealExecutionContinuationStateStore,
)
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection
from testing_support.host_fixture_wiring import (
    install_diagnostic_cursor_secret,
    install_host_llm_stub,
)
from testing_support.orchestration.orchestration_consequential_effect_reliability_doubles import (
    DurableTestProviderInvocationStore,
)

pytestmark = [pytest.mark.unit]

_T0 = datetime(2026, 10, 13, 12, 0, 0, tzinfo=timezone.utc)


def _invocation(invocation_id: str = "inv-pis-02") -> ProviderInvocation:
    return ProviderInvocation(
        invocation_id=invocation_id,
        provider_id="provider.test",
        operation="create_work",
        task_id="task-pis-02",
        run_id="run-pis-02",
        request_digest="sha256:" + ("ab" * 32),
        started_at=_T0,
    )


def _strict_composition_builder(tmp_path: Path):
    def _build(
        store: SerializedSealExecutionContinuationStateStore,
        invocation_store: ProviderInvocationStore,
    ) -> ProductionProcessComposition:
        platform_persistence = build_reference_production_platform_persistence(
            db_path=tmp_path / "platform-kv.db",
        )
        return ProductionProcessComposition(
            agent_platform_runtime=build_production_agent_platform_runtime(
                platform_persistence=platform_persistence,
            ),
            provider_invocation_store=invocation_store,
            execution_continuation_state_store=store,
        )

    return _build


def _seed_active_registry_projection(
    composition: ProductionProcessComposition,
    *,
    application_id: str,
    application_environment_id: str,
    projection: object,
) -> None:
    stores = composition.agent_platform_runtime.stores
    stores.registry_projection_store.put(projection)
    stores.serving_store.atomic_swap_serving_revision(
        application_id=application_id,
        application_environment_id=application_environment_id,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=projection.evidence.runtime_revision_id,  # type: ignore[attr-defined]
        prior_revision_id=None,
        committed_at=datetime.now(UTC),
    )


def _host_runtime_for_startup(settings: GovernedContractorBackendSettings):
    from applications.governed_contractor_application.tests.host.test_gr6_wire_production_decision_governance import (
        _test_policy_bundle,
    )
    from external_contractor_adapter.tests.fakes.deterministic_external_work import (
        DeterministicExternalWorkFake,
    )
    from governed_contractor_application.host.collaborative_work_local_fixture import (
        build_in_memory_collaborative_work_repositories,
    )
    from intergrax.contracts.execution_identity import mint_task_id
    from tests.unit.runtime.governance.gr3_test_support import StaticActiveTaskScope

    fake = DeterministicExternalWorkFake()
    cw = build_in_memory_collaborative_work_repositories()
    settings = replace(
        settings,
        include_mcp=False,
        include_scheduler=False,
        include_interaction_routes=False,
        include_task_control=False,
        runtime_policy_bundle=_test_policy_bundle(),  # type: ignore[arg-type]
    )
    return compose_governed_contractor_host_runtime(
        settings,
        integration=fake,
        task_scope=StaticActiveTaskScope(mint_task_id()),
        collaborative_work_repositories=cw,
    )


def _build_serving_app(lifecycle, *, settings: GovernedContractorBackendSettings, host_runtime):
    manifest = build_governed_contractor_manifest()
    env = build_governed_contractor_environment_profile(settings)
    projection = build_test_registry_projection(
        manifest,
        env,
        builders=build_governed_contractor_agent_builders(host_runtime),
        revision_id="prod-q-r1-pis-02",
        settings=settings,
    )
    _seed_active_registry_projection(
        lifecycle.composition,
        application_id=manifest.app_id,
        application_environment_id=env.profile_id,
        projection=projection,
    )
    return create_governed_contractor_process_app(
        process_composition=lifecycle.composition,
        settings=settings,
        host_runtime=host_runtime,
    )


def test_pis_02_a_lifecycle_supplies_durable_sqlite_and_continuation(
    tmp_path: Path,
) -> None:
    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    invocation_db = tmp_path / "provider_invocations.db"
    lifecycle = open_reference_production_process_lifecycle(
        continuation_state_file=tmp_path / "continuation.json",
        provider_invocation_db=invocation_db,
        env=env,
        missing_file_policy="empty",
        composition_builder=_strict_composition_builder(tmp_path),
    )
    try:
        assert lifecycle.execution_continuation_state_store.is_durable is True
        store = lifecycle.provider_invocation_store
        assert isinstance(store, SQLiteProviderInvocationStore)
        assert store.is_durable is True
        assert store.db_path == invocation_db
        assert lifecycle.composition.provider_invocation_store is store
        assert lifecycle.composition.execution_continuation_state_store is (
            lifecycle.execution_continuation_state_store
        )
    finally:
        lifecycle.close_process_resources()


def test_pis_02_b_gca_and_orchestration_share_composition_invocation_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
    install_host_llm_stub(monkeypatch)
    monkeypatch.setenv("INTERGRAX_RELATIONAL_DB", str(tmp_path / "relational.db"))

    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    host_runtime = _host_runtime_for_startup(settings)

    with open_reference_production_process_lifecycle(
        continuation_state_file=tmp_path / "continuation.json",
        provider_invocation_db=tmp_path / "provider_invocations.db",
        env=env,
        missing_file_policy="empty",
        composition_builder=_strict_composition_builder(tmp_path),
    ) as lifecycle:
        composition_store = lifecycle.composition.provider_invocation_store
        app = _build_serving_app(
            lifecycle,
            settings=settings,
            host_runtime=host_runtime,
        )
        harness = app.state.harness_runtime
        topology = harness.orchestration_topology
        assert topology is not None
        assert topology.provider_invocation_store is composition_store
        with TestClient(app):
            assert lifecycle.provider_invocation_store is composition_store


def test_pis_02_c_startup_failure_after_open_releases_owned_stores(
    tmp_path: Path,
) -> None:
    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    state_file = tmp_path / "continuation.json"
    invocation_db = tmp_path / "provider_invocations.db"

    with patch(
        "intergrax.applications._shared.reference_production_process_lifecycle."
        "wire_governed_reference_production_launcher",
        side_effect=RuntimeError("injected lifecycle wiring failure"),
    ):
        with pytest.raises(RuntimeError, match="injected lifecycle wiring failure"):
            open_reference_production_process_lifecycle(
                continuation_state_file=state_file,
                provider_invocation_db=invocation_db,
                env=env,
                missing_file_policy="empty",
                composition_builder=_strict_composition_builder(tmp_path),
            )

    continuation = open_production_execution_continuation_state_store(
        state_file,
        missing_file_policy="empty",
    )
    continuation.close()
    reopened = open_provider_invocation_store_at(invocation_db)
    reopened.close()


def test_pis_02_d_normal_shutdown_closes_invocation_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
    install_host_llm_stub(monkeypatch)
    monkeypatch.setenv("INTERGRAX_RELATIONAL_DB", str(tmp_path / "relational.db"))

    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    host_runtime = _host_runtime_for_startup(settings)

    lifecycle = open_reference_production_process_lifecycle(
        continuation_state_file=tmp_path / "continuation.json",
        provider_invocation_db=tmp_path / "provider_invocations.db",
        env=env,
        missing_file_policy="empty",
        composition_builder=_strict_composition_builder(tmp_path),
    )
    store = lifecycle.provider_invocation_store
    assert store is not None
    app = _build_serving_app(
        lifecycle,
        settings=settings,
        host_runtime=host_runtime,
    )
    with TestClient(app):
        store.put_invocation(_invocation())
        assert store.get_invocation("inv-pis-02") is not None

    lifecycle.close_process_resources()
    with pytest.raises(ProviderInvocationPersistenceError, match="closed"):
        store.put_invocation(_invocation("inv-after-close"))


def test_pis_02_e_restart_reads_committed_intent_from_same_sqlite_db(
    tmp_path: Path,
) -> None:
    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    state_file = tmp_path / "continuation.json"
    invocation_db = tmp_path / "provider_invocations.db"
    inv = _invocation("inv-pis-02-restart")

    lifecycle_a = open_reference_production_process_lifecycle(
        continuation_state_file=state_file,
        provider_invocation_db=invocation_db,
        env=env,
        missing_file_policy="empty",
        composition_builder=_strict_composition_builder(tmp_path),
    )
    store_a = lifecycle_a.provider_invocation_store
    assert store_a is not None
    store_a.put_invocation(inv)
    lifecycle_a.close_process_resources()

    lifecycle_b = open_reference_production_process_lifecycle(
        continuation_state_file=state_file,
        provider_invocation_db=invocation_db,
        env=env,
        missing_file_policy="empty",
        composition_builder=_strict_composition_builder(tmp_path),
    )
    try:
        store_b = lifecycle_b.provider_invocation_store
        assert store_b is not None
        assert store_b is not lifecycle_a.composition.provider_invocation_store
        assert store_b.get_invocation(inv.invocation_id) == inv
    finally:
        lifecycle_b.close_process_resources()


def test_pis_02_f_strict_rejection_conflicting_and_missing_bindings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    external = DurableTestProviderInvocationStore()

    with pytest.raises(ReferenceProductionProcessLifecycleError, match="mutually exclusive"):
        open_reference_production_process_lifecycle(
            continuation_state_file=tmp_path / "continuation.json",
            provider_invocation_db=tmp_path / "provider_invocations.db",
            provider_invocation_store=external,
            env=env,
            missing_file_policy="empty",
            composition_builder=_strict_composition_builder(tmp_path),
        )

    def _wrong_invocation_binding(
        store: SerializedSealExecutionContinuationStateStore,
        invocation_store: ProviderInvocationStore,
    ) -> ProductionProcessComposition:
        del invocation_store
        return _strict_composition_builder(tmp_path)(store, DurableTestProviderInvocationStore())

    with pytest.raises(ReferenceProductionProcessLifecycleError, match="provider_invocation_store"):
        open_reference_production_process_lifecycle(
            continuation_state_file=tmp_path / "continuation.json",
            provider_invocation_db=tmp_path / "provider_invocations.db",
            env=env,
            missing_file_policy="empty",
            composition_builder=_wrong_invocation_binding,
        )

    lifecycle = open_reference_production_process_lifecycle(
        continuation_state_file=tmp_path / "continuation.json",
        env=env,
        missing_file_policy="empty",
    )
    try:
        assert lifecycle.composition.provider_invocation_store is None
        with pytest.raises(HarnessHostOrchestrationTopologyReliabilityCompositionError):
            _build_serving_app(
                lifecycle,
                settings=settings,
                host_runtime=_host_runtime_for_startup(settings),
            )
    finally:
        lifecycle.close_process_resources()


def test_pis_02_g_unresolved_intent_blocks_dispatch_after_restart(
    tmp_path: Path,
) -> None:
    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    invocation_db = tmp_path / "provider_invocations.db"
    inv = _invocation("inv-pis-02-unresolved")

    lifecycle_a = open_reference_production_process_lifecycle(
        continuation_state_file=tmp_path / "continuation.json",
        provider_invocation_db=invocation_db,
        env=env,
        missing_file_policy="empty",
        composition_builder=_strict_composition_builder(tmp_path),
    )
    store_a = lifecycle_a.provider_invocation_store
    assert store_a is not None
    store_a.put_invocation(inv)
    lifecycle_a.close_process_resources()

    lifecycle_b = open_reference_production_process_lifecycle(
        continuation_state_file=tmp_path / "continuation.json",
        provider_invocation_db=invocation_db,
        env=env,
        missing_file_policy="empty",
        composition_builder=_strict_composition_builder(tmp_path),
    )
    store_b = lifecycle_b.provider_invocation_store
    assert isinstance(store_b, SQLiteProviderInvocationStore)
    gate = GovernedProviderInvocationDispatchGate(store=store_b, clock=lambda: _T0)
    calls = 0

    def _execute() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    try:
        with pytest.raises(ProviderInvocationPersistenceError, match="dispatch_not_admitted"):
            gate.dispatch_after_intent_persisted(inv, _execute)
        assert calls == 0
    finally:
        lifecycle_b.close_process_resources()


def test_pis_02_r1_missing_continuation_releases_owned_sqlite(tmp_path: Path) -> None:
    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    invocation_db = tmp_path / "provider_invocations.db"
    state_file = tmp_path / "continuation.json"

    with pytest.raises(ExecutionContinuationError, match="missing"):
        open_reference_production_process_lifecycle(
            continuation_state_file=state_file,
            provider_invocation_db=invocation_db,
            env=env,
            missing_file_policy="fail",
            composition_builder=_strict_composition_builder(tmp_path),
        )

    reopened = open_provider_invocation_store_at(invocation_db)
    reopened.close()


def test_pis_02_r1_continuation_writer_lock_conflict_releases_owned_sqlite(
    tmp_path: Path,
) -> None:
    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    invocation_db = tmp_path / "provider_invocations.db"
    state_file = tmp_path / "continuation.json"
    holder = open_production_execution_continuation_state_store(
        state_file,
        missing_file_policy="empty",
    )
    try:
        with pytest.raises(ExecutionContinuationError, match="conflicting"):
            open_reference_production_process_lifecycle(
                continuation_state_file=state_file,
                provider_invocation_db=invocation_db,
                env=env,
                missing_file_policy="empty",
                composition_builder=_strict_composition_builder(tmp_path),
            )
    finally:
        holder.close()

    reopened = open_provider_invocation_store_at(invocation_db)
    reopened.close()


def test_pis_02_r1_composition_builder_failure_releases_both_owned_stores(
    tmp_path: Path,
) -> None:
    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    state_file = tmp_path / "continuation.json"
    invocation_db = tmp_path / "provider_invocations.db"

    def _failing_builder(
        store: SerializedSealExecutionContinuationStateStore,
        invocation_store: ProviderInvocationStore,
    ) -> ProductionProcessComposition:
        del store, invocation_store
        raise RuntimeError("composition construction failed")

    with pytest.raises(RuntimeError, match="composition construction failed"):
        open_reference_production_process_lifecycle(
            continuation_state_file=state_file,
            provider_invocation_db=invocation_db,
            env=env,
            missing_file_policy="empty",
            composition_builder=_failing_builder,
        )

    continuation = open_production_execution_continuation_state_store(
        state_file,
        missing_file_policy="empty",
    )
    continuation.close()
    reopened = open_provider_invocation_store_at(invocation_db)
    reopened.close()


def test_pis_02_r1_injected_invocation_close_failure_still_closes_continuation(
    tmp_path: Path,
) -> None:
    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    lifecycle = open_reference_production_process_lifecycle(
        continuation_state_file=tmp_path / "continuation.json",
        provider_invocation_db=tmp_path / "provider_invocations.db",
        env=env,
        missing_file_policy="empty",
        composition_builder=_strict_composition_builder(tmp_path),
    )
    owned_invocation = lifecycle._owned_invocation_store
    assert owned_invocation is not None
    with (
        patch.object(
            owned_invocation,
            "close",
            side_effect=RuntimeError("injected invocation close failure"),
        ),
        patch.object(lifecycle._owned_continuation_store, "close") as continuation_close,
    ):
        with pytest.raises(RuntimeError, match="injected invocation close failure"):
            lifecycle.close_process_resources()
        continuation_close.assert_called_once()


def test_pis_02_r1_injected_continuation_close_failure_still_closes_invocation(
    tmp_path: Path,
) -> None:
    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    lifecycle = open_reference_production_process_lifecycle(
        continuation_state_file=tmp_path / "continuation.json",
        provider_invocation_db=tmp_path / "provider_invocations.db",
        env=env,
        missing_file_policy="empty",
        composition_builder=_strict_composition_builder(tmp_path),
    )
    owned_invocation = lifecycle._owned_invocation_store
    assert owned_invocation is not None
    with (
        patch.object(
            lifecycle._owned_continuation_store,
            "close",
            side_effect=RuntimeError("injected continuation close failure"),
        ),
        patch.object(owned_invocation, "close") as invocation_close,
    ):
        with pytest.raises(RuntimeError, match="injected continuation close failure"):
            lifecycle.close_process_resources()
        invocation_close.assert_called_once()


def test_pis_02_r1_repeated_close_after_successful_cleanup_is_harmless(
    tmp_path: Path,
) -> None:
    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    lifecycle = open_reference_production_process_lifecycle(
        continuation_state_file=tmp_path / "continuation.json",
        provider_invocation_db=tmp_path / "provider_invocations.db",
        env=env,
        missing_file_policy="empty",
        composition_builder=_strict_composition_builder(tmp_path),
    )
    lifecycle.close_process_resources()
    lifecycle.close_process_resources()


def test_pis_02_r1_external_invocation_store_stays_open_after_lifecycle_failure(
    tmp_path: Path,
) -> None:
    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)
    external = DurableTestProviderInvocationStore()

    with patch(
        "intergrax.applications._shared.reference_production_process_lifecycle."
        "wire_governed_reference_production_launcher",
        side_effect=RuntimeError("injected lifecycle wiring failure"),
    ):
        with pytest.raises(RuntimeError, match="injected lifecycle wiring failure"):
            open_reference_production_process_lifecycle(
                continuation_state_file=tmp_path / "continuation.json",
                provider_invocation_store=external,
                env=env,
                missing_file_policy="empty",
                composition_builder=_strict_composition_builder(tmp_path),
            )

    external.put_invocation(_invocation("inv-external-survives"))
    assert external.get_invocation("inv-external-survives") is not None
