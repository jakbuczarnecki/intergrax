# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R6 — composition launcher owns continuation store open/close lifecycle."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

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
    open_reference_production_process_lifecycle,
)
from intergrax.applications._shared.strict_host_dependency_concurrency_admission import (
    resolve_dependency_concurrency_admission_configuration,
)
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests
from intergrax.runtime.execution.continuation.durable_serialized_seal_store import (
    SerializedSealExecutionContinuationStateStore,
)
from intergrax.runtime.execution.continuation.durability_policy import (
    validate_execution_continuation_for_composition,
)
from intergrax.runtime.execution.continuation.persistence import (
    InMemoryExecutionContinuationStateStore,
)
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection
from testing_support.host_fixture_wiring import (
    install_diagnostic_cursor_secret,
    install_host_llm_stub,
)
pytestmark = [pytest.mark.unit]


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


def _build_serving_app(
    lifecycle,
    *,
    settings: GovernedContractorBackendSettings,
    host_runtime,
) -> object:
    manifest = build_governed_contractor_manifest()
    env = build_governed_contractor_environment_profile(settings)
    projection = build_test_registry_projection(
        manifest,
        env,
        builders=build_governed_contractor_agent_builders(host_runtime),
        revision_id="prod-q-r1-r6-lifecycle",
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


def test_launcher_owned_normal_startup_serve_and_shutdown_releases_writer_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
    install_host_llm_stub(monkeypatch)
    monkeypatch.setenv("INTERGRAX_RELATIONAL_DB", str(tmp_path / "relational.db"))

    state_file = tmp_path / "continuation.json"
    settings = GovernedContractorBackendSettings.from_env()
    host_runtime = _host_runtime_for_startup(settings)
    env = build_governed_contractor_environment_profile(settings)

    with open_reference_production_process_lifecycle(
        continuation_state_file=state_file,
        provider_invocation_db=tmp_path / "provider_invocations.db",
        env=env,
        missing_file_policy="empty",
        composition_builder=_strict_composition_builder(tmp_path),
    ) as lifecycle:
        store = lifecycle.execution_continuation_state_store
        assert store.is_durable is True
        app = _build_serving_app(
            lifecycle,
            settings=settings,
            host_runtime=host_runtime,
        )
        harness = app.state.harness_runtime
        assert harness is not None
        runtime_env = harness.env_wiring.build_context.environment
        admission = resolve_dependency_concurrency_admission_configuration(
            runtime_env.reliability_profile.dependency_concurrency_admission,
            production_mode=runtime_env.execution_mode.value == "strict",
        )
        assert admission is not None
        pending = PendingExecutionContinuation(
            continuation_id="cont_r6_serve",
            identity=ExecutionContinuationIdentity(
                task_id=canonical_task_id_for_tests("r6-lifecycle"),
                run_id=canonical_run_id_for_tests("r6-lifecycle"),
                attempt_id=mint_attempt_id(),
                execution_id=mint_execution_id(),
            ),
            lifecycle_state=ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
            revision=1,
            reason=ContinuationReason.SECURITY,
            pause_id="pause",
            human_request_id="hr",
        )
        with TestClient(app) as client:
            assert client.app is app
            assert store.begin_current_episode_if_predecessor_allows(pending)

    reopened = open_production_execution_continuation_state_store(
        state_file,
        missing_file_policy="empty",
    )
    reopened.close()


def test_launcher_owned_startup_failure_releases_writer_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
    state_file = tmp_path / "continuation.json"
    settings = GovernedContractorBackendSettings.from_env()
    env = build_governed_contractor_environment_profile(settings)

    with pytest.raises(RuntimeError, match="injected startup failure"):
        with open_reference_production_process_lifecycle(
            continuation_state_file=state_file,
            provider_invocation_db=tmp_path / "provider_invocations.db",
            env=env,
            missing_file_policy="empty",
            composition_builder=_strict_composition_builder(tmp_path),
        ) as lifecycle:
            with patch(
                "governed_contractor_application.host.factory.build_harness_host_runtime",
                side_effect=RuntimeError("injected startup failure"),
            ):
                _build_serving_app(
                    lifecycle,
                    settings=settings,
                    host_runtime=_host_runtime_for_startup(settings),
                )

    reopened = open_production_execution_continuation_state_store(
        state_file,
        missing_file_policy="empty",
    )
    reopened.close()


def test_factory_shutdown_does_not_close_composition_owned_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
    install_host_llm_stub(monkeypatch)
    monkeypatch.setenv("INTERGRAX_RELATIONAL_DB", str(tmp_path / "relational.db"))

    state_file = tmp_path / "continuation.json"
    settings = GovernedContractorBackendSettings.from_env()
    host_runtime = _host_runtime_for_startup(settings)
    env = build_governed_contractor_environment_profile(settings)

    lifecycle = open_reference_production_process_lifecycle(
        continuation_state_file=state_file,
        provider_invocation_db=tmp_path / "provider_invocations.db",
        env=env,
        missing_file_policy="empty",
        composition_builder=_strict_composition_builder(tmp_path),
    )
    try:
        app = _build_serving_app(
            lifecycle,
            settings=settings,
            host_runtime=host_runtime,
        )
        with TestClient(app):
            pass
        with pytest.raises(ExecutionContinuationError) as exc:
            open_production_execution_continuation_state_store(
                state_file,
                missing_file_policy="empty",
            )
        assert exc.value.code is ExecutionContinuationErrorCode.STORE_QUERY_FAILED
    finally:
        lifecycle.close_process_resources()

    second = open_production_execution_continuation_state_store(
        state_file,
        missing_file_policy="empty",
    )
    second.close()


def test_strict_continuation_fail_closed_non_durable_unchanged() -> None:
    lab_store = InMemoryExecutionContinuationStateStore()
    with pytest.raises(ExecutionContinuationError) as exc:
        validate_execution_continuation_for_composition(
            production_mode=True,
            state_store=lab_store,
            continuation_explicitly_wired=False,
            continuation_disabled=False,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE
