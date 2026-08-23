# © Artur Czarnecki. All rights reserved.

"""LKW background worker production registry authority proofs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime

from intergrax.applications._shared.harness_registry_authority import HarnessHostRegistryAuthorityError
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    create_reference_production_process_composition,
)
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleLauncher,
)
from local_workspace_application.background_ingest.contracts import LKW_BACKGROUND_INGEST_TASK_NAME
from local_workspace_application.host.background_worker_factory import (
    build_local_workspace_background_worker_wiring,
)
from local_workspace_application.host.background_worker_main import (
    activate_local_workspace_reference_production_authority,
    main,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _settings(monkeypatch: pytest.MonkeyPatch) -> LocalWorkspaceBackendSettings:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-lkw-worker-authority-key")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_REDIS", "true")
    return LocalWorkspaceBackendSettings.from_env()


def _activated_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, MaterializedRegistryProjection]:
    settings = _settings(monkeypatch)
    composition, projection = activate_local_workspace_reference_production_authority(settings)
    return composition, projection


def test_activate_reference_production_authority_uses_one_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    composition, projection = activate_local_workspace_reference_production_authority(settings)
    env = build_local_workspace_environment_profile(settings)
    resolved = bootstrap_production_registry_projection(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        application_environment_id=env.profile_id,
        stores=composition.agent_platform_runtime.stores,
    )
    assert resolved.evidence.runtime_revision_id == projection.evidence.runtime_revision_id
    assert resolved.agent_registry.list_agent_ids() == projection.agent_registry.list_agent_ids()


def test_worker_wiring_receives_materialized_registry_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    _, projection = _activated_projection(monkeypatch)
    with (
        patch(
            "local_workspace_application.host.background_worker_factory.create_redis_kv_store",
            return_value=MagicMock(),
        ),
        patch(
            "local_workspace_application.host.background_worker_factory.create_kafka_worker",
            return_value=MagicMock(),
        ),
    ):
        wiring = build_local_workspace_background_worker_wiring(
            manifest=LOCAL_WORKSPACE_APPLICATION_MANIFEST,
            registry_projection=projection,
            settings=settings,
        )
    assert wiring.runtime.registry_projection_evidence is not None
    assert (
        wiring.runtime.registry_projection_evidence.runtime_revision_id
        == projection.evidence.runtime_revision_id
    )
    assert "local_indexer" in wiring.runtime.nexus_loop.registry.list_agent_ids()


def test_worker_wiring_without_projection_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    with patch(
        "intergrax.applications._shared.harness_registry_authority.build_manifest_development_registry",
    ) as manifest_builder:
        with pytest.raises(HarnessHostRegistryAuthorityError, match="MaterializedRegistryProjection"):
            build_local_workspace_background_worker_wiring(
                manifest=LOCAL_WORKSPACE_APPLICATION_MANIFEST,
                registry_projection=None,  # type: ignore[arg-type]
                settings=settings,
            )
    manifest_builder.assert_not_called()


def test_authority_assembly_order_before_worker_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch)
    events: list[str] = []

    def _track(event: str, func: object, *args: object, **kwargs: object) -> object:
        events.append(event)
        return func(*args, **kwargs)

    worker = MagicMock()
    worker.start = MagicMock(side_effect=lambda: events.append("start_worker"))
    original_deploy = ReferenceProductionLifecycleLauncher.deploy_and_activate

    def tracked_deploy(
        self: ReferenceProductionLifecycleLauncher,
        *args: object,
        **kwargs: object,
    ) -> object:
        events.append("deploy_activate")
        return original_deploy(self, *args, **kwargs)

    with (
        patch(
            "local_workspace_application.host.background_worker_main.create_reference_production_process_composition",
            side_effect=lambda: _track(
                "create_composition",
                create_reference_production_process_composition,
            ),
        ),
        patch.object(
            ReferenceProductionLifecycleLauncher,
            "deploy_and_activate",
            tracked_deploy,
        ),
        patch(
            "local_workspace_application.host.background_worker_main.bootstrap_production_registry_projection",
            side_effect=lambda **kwargs: _track(
                "bootstrap_projection",
                bootstrap_production_registry_projection,
                **kwargs,
            ),
        ),
        patch(
            "local_workspace_application.host.background_worker_factory.build_harness_host_runtime",
            side_effect=lambda *args, **kwargs: _track(
                "build_runtime",
                build_harness_host_runtime,
                *args,
                **kwargs,
            ),
        ),
        patch(
            "local_workspace_application.host.background_worker_factory.create_redis_kv_store",
            return_value=MagicMock(),
        ),
        patch(
            "local_workspace_application.host.background_worker_factory.create_kafka_worker",
            side_effect=lambda **kwargs: _track("create_worker", lambda **_: worker, **kwargs),
        ),
    ):
        assert main() == 0

    assert events == [
        "create_composition",
        "deploy_activate",
        "bootstrap_projection",
        "build_runtime",
        "create_worker",
        "start_worker",
    ]


def test_bootstrap_failure_gates_worker_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch)
    with (
        patch(
            "local_workspace_application.host.background_worker_main.bootstrap_production_registry_projection",
            side_effect=HarnessHostRegistryAuthorityError("projection bootstrap failed"),
        ),
        patch(
            "local_workspace_application.host.background_worker_factory.create_kafka_worker",
        ) as create_worker,
    ):
        with pytest.raises(HarnessHostRegistryAuthorityError, match="projection bootstrap failed"):
            main()
    create_worker.assert_not_called()


def test_message_bus_disabled_returns_exit_code_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS", "false")
    with patch(
        "local_workspace_application.host.background_worker_factory.build_local_workspace_background_worker_wiring",
    ) as build_wiring:
        assert main() == 1
    build_wiring.assert_not_called()


def test_worker_registers_background_ingest_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    _, projection = _activated_projection(monkeypatch)
    with (
        patch(
            "local_workspace_application.host.background_worker_factory.create_redis_kv_store",
            return_value=MagicMock(),
        ),
        patch(
            "local_workspace_application.host.background_worker_factory.create_kafka_worker",
            return_value=MagicMock(),
        ),
    ):
        wiring = build_local_workspace_background_worker_wiring(
            manifest=LOCAL_WORKSPACE_APPLICATION_MANIFEST,
            registry_projection=projection,
            settings=settings,
        )
    handler = wiring.registry.get_handler(LKW_BACKGROUND_INGEST_TASK_NAME)
    assert callable(handler)
