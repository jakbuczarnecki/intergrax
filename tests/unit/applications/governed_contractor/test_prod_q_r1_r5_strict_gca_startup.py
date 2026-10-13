# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R5 — real strict GCA startup (sanctioned path, unmocked harness)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from governed_contractor_application.host.agent_builders import (
    build_governed_contractor_agent_builders,
)
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    compose_governed_contractor_host_runtime,
)
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
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
from intergrax.applications._shared.strict_host_dependency_concurrency_admission import (
    resolve_dependency_concurrency_admission_configuration,
)
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
)
from intergrax.runtime.execution.continuation.durable_serialized_seal_store import (
    SerializedSealExecutionContinuationStateStore,
)
from intergrax.runtime.execution.continuation.persistence import (
    InMemoryExecutionContinuationStateStore,
)
from intergrax.runtime.execution.continuation.durability_policy import (
    validate_execution_continuation_for_composition,
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


def _reference_strict_composition(
    tmp_path: Path,
    *,
    continuation_store: SerializedSealExecutionContinuationStateStore,
) -> ProductionProcessComposition:
    platform_persistence = build_reference_production_platform_persistence(
        db_path=tmp_path / "platform-kv.db",
    )
    return ProductionProcessComposition(
        agent_platform_runtime=build_production_agent_platform_runtime(
            platform_persistence=platform_persistence,
        ),
        provider_invocation_store=DurableTestProviderInvocationStore(),
        execution_continuation_state_store=continuation_store,
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


def test_sanctioned_process_app_real_harness_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
    install_host_llm_stub(monkeypatch)
    monkeypatch.setenv("INTERGRAX_RELATIONAL_DB", str(tmp_path / "relational.db"))

    state_file = tmp_path / "continuation.json"
    continuation_store = open_production_execution_continuation_state_store(
        state_file,
        missing_file_policy="empty",
    )
    composition = _reference_strict_composition(
        tmp_path,
        continuation_store=continuation_store,
    )
    settings = GovernedContractorBackendSettings.from_env()
    host_runtime = _host_runtime_for_startup(settings)
    manifest = build_governed_contractor_manifest()
    env = build_governed_contractor_environment_profile(settings)
    projection = build_test_registry_projection(
        manifest,
        env,
        builders=build_governed_contractor_agent_builders(host_runtime),
        revision_id="prod-q-r1-r5-real-startup",
        settings=settings,
    )
    _seed_active_registry_projection(
        composition,
        application_id=manifest.app_id,
        application_environment_id=env.profile_id,
        projection=projection,
    )

    app = create_governed_contractor_process_app(
        process_composition=composition,
        settings=settings,
        host_runtime=host_runtime,
    )
    harness = app.state.harness_runtime
    assert harness is not None
    assert composition.provider_invocation_store is not None
    runtime_env = harness.env_wiring.build_context.environment
    admission = resolve_dependency_concurrency_admission_configuration(
        runtime_env.reliability_profile.dependency_concurrency_admission,
        production_mode=runtime_env.execution_mode.value == "strict",
    )
    assert admission is not None

    assert harness.orchestration_topology is not None
    assert harness.orchestration_topology.submission_port is not None
    assert continuation_store.is_durable is True
    assert harness.registry_projection_evidence is not None

    with TestClient(app) as client:
        assert client.app is app

    continuation_store.close()


def test_sanctioned_process_app_fail_closed_second_writer_same_state_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
    state_file = tmp_path / "continuation.json"
    first = open_production_execution_continuation_state_store(
        state_file,
        missing_file_policy="empty",
    )
    with pytest.raises(ExecutionContinuationError) as exc:
        open_production_execution_continuation_state_store(
            state_file,
            missing_file_policy="empty",
        )
    assert exc.value.code is ExecutionContinuationErrorCode.STORE_QUERY_FAILED
    first.close()


def test_strict_startup_fail_closed_on_non_durable_composition_store() -> None:
    lab_store = InMemoryExecutionContinuationStateStore()
    assert lab_store.is_durable is False
    with pytest.raises(ExecutionContinuationError) as exc:
        validate_execution_continuation_for_composition(
            production_mode=True,
            state_store=lab_store,
            continuation_explicitly_wired=False,
            continuation_disabled=False,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE
