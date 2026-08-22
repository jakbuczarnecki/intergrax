# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi import FastAPI
from pathlib import Path

from intergrax.applications._shared.harness_registry_authority import HarnessHostRegistryAuthorityError
from intergrax.applications._shared.reliability_assembly_resolver import (
    assert_reliability_assembly_valid,
)
from intergrax.applications._shared.reliability_wiring import wire_application_reliability
from intergrax.contracts.persistence_topology import (
    PersistenceTopology,
    resolve_idempotency_store_topology,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import IDEMPOTENCY_DB_NAME
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleLauncher,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.main import (
    app,
    create_app,
    create_local_workspace_process_app,
    run_reference_production,
)
from local_workspace_application.host.reference_lifecycle_input import (
    build_local_workspace_reference_lifecycle_input,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_create_app_without_composition_fails_closed() -> None:
    with pytest.raises(HarnessHostRegistryAuthorityError, match="STRICT production"):
        create_app()


@pytest.mark.asyncio
async def test_module_app_placeholder_cannot_serve_production() -> None:
    with pytest.raises(HarnessHostRegistryAuthorityError, match="cannot serve STRICT production"):
        await app({}, lambda: None, lambda _message: None)


def test_run_reference_production_activates_composition_before_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-lkw-production-bootstrap-key")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_MCP", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_SCHEDULER", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_TASK_CONTROL", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_INTERACTIONS", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    launcher_calls: list[tuple[object, object]] = []
    served_app: list[FastAPI] = []
    composition_holder: list[ProductionProcessComposition] = []

    original_deploy = ReferenceProductionLifecycleLauncher.deploy_and_activate
    original_create = create_reference_production_process_composition

    def _track_create() -> ProductionProcessComposition:
        composition = original_create()
        composition_holder.append(composition)
        return composition

    def _track_deploy(
        self: ReferenceProductionLifecycleLauncher,
        projection_input: object,
        activation_request: object,
    ) -> object:
        launcher_calls.append((projection_input, activation_request))
        return original_deploy(self, projection_input, activation_request)

    def _capture_uvicorn_run(asgi_app: FastAPI, **_kwargs: object) -> None:
        served_app.append(asgi_app)

    monkeypatch.setattr(
        "local_workspace_application.host.main.create_reference_production_process_composition",
        _track_create,
    )
    monkeypatch.setattr(
        ReferenceProductionLifecycleLauncher,
        "deploy_and_activate",
        _track_deploy,
    )
    monkeypatch.setattr("uvicorn.run", _capture_uvicorn_run)

    run_reference_production()

    assert len(launcher_calls) == 1
    assert len(composition_holder) == 1
    composition = composition_holder[0]
    env = build_local_workspace_environment_profile(LocalWorkspaceBackendSettings.from_env())
    resolved = bootstrap_production_registry_projection(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        application_environment_id=env.profile_id,
        stores=composition.agent_platform_runtime.stores,
    )
    assert resolved.evidence.runtime_revision_id == "local-workspace-reference-runtime-revision"
    assert len(served_app) == 1
    assert served_app[0].title
    assert "Local Workspace" in served_app[0].title


def test_run_reference_production_uses_same_composition_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-lkw-store-continuity-key")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_MCP", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_SCHEDULER", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_TASK_CONTROL", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_INTERACTIONS", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    composition_holder: list[ProductionProcessComposition] = []
    served_app: list[FastAPI] = []

    original_create = create_reference_production_process_composition

    def _track_create() -> ProductionProcessComposition:
        composition = original_create()
        composition_holder.append(composition)
        return composition

    def _capture_uvicorn_run(asgi_app: FastAPI, **_kwargs: object) -> None:
        served_app.append(asgi_app)

    monkeypatch.setattr(
        "local_workspace_application.host.main.create_reference_production_process_composition",
        _track_create,
    )
    monkeypatch.setattr("uvicorn.run", _capture_uvicorn_run)

    run_reference_production()

    assert len(composition_holder) == 1
    composition = composition_holder[0]
    settings = LocalWorkspaceBackendSettings.from_env()
    env = build_local_workspace_environment_profile(settings)
    resolved = bootstrap_production_registry_projection(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        application_environment_id=env.profile_id,
        stores=composition.agent_platform_runtime.stores,
    )
    assert resolved.evidence.runtime_revision_id == "local-workspace-reference-runtime-revision"
    assert len(served_app) == 1
    assert create_local_workspace_process_app(
        process_composition=composition,
        settings=settings,
    ).title == served_app[0].title


def test_lkw_product_wiring_uses_durable_idempotency_store_under_data_home(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORKSPACE_DATA_HOME", str(tmp_path / "lkw-data"))

    settings = LocalWorkspaceBackendSettings.from_env()
    env = build_local_workspace_environment_profile(settings)
    wiring = wire_application_reliability(
        env,
        idempotency_db_path=Path(settings.idempotency_db_path),
    )

    assert wiring.idempotency_store is not None
    assert resolve_idempotency_store_topology(wiring.idempotency_store) is (
        PersistenceTopology.DURABLE_SINGLE_HOST
    )
    assert_reliability_assembly_valid(wiring, env)
    assert settings.idempotency_db_path.endswith(f"data/sqlite/{IDEMPOTENCY_DB_NAME}")
    assert Path(settings.idempotency_db_path).exists()
