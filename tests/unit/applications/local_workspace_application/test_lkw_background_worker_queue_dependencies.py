# © Artur Czarnecki. All rights reserved.

"""LKW background worker canonical queue execution dependency wiring."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from intergrax.applications._shared.host_queue_execution_wiring import (
    HostQueueExecutionDependencies,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)
from local_workspace_application.host.background_worker_factory import (
    build_local_workspace_background_worker_wiring,
)
from local_workspace_application.host.background_worker_main import (
    activate_local_workspace_reference_production_authority,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class _StubQueueDependencies:
    kv_store: DistributedKVStore
    causal_evidence_persistence: CausalEvidencePersistence


def _settings(monkeypatch: pytest.MonkeyPatch) -> LocalWorkspaceBackendSettings:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-lkw-worker-queue-deps-key")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_REDIS", "true")
    return LocalWorkspaceBackendSettings.from_env()


def _projection(monkeypatch: pytest.MonkeyPatch):
    settings = _settings(monkeypatch)
    _, projection = activate_local_workspace_reference_production_authority(settings)
    return settings, projection


def test_worker_wiring_uses_canonical_queue_execution_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, projection = _projection(monkeypatch)
    kv_store = MagicMock(spec=DistributedKVStore)
    causal_persistence = MagicMock(spec=CausalEvidencePersistence)
    queue_dependencies = HostQueueExecutionDependencies(
        kv_store=kv_store,
        causal_evidence_persistence=causal_persistence,
    )
    captured: dict[str, object] = {}

    def _capture_worker(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    with (
        patch(
            "local_workspace_application.host.background_worker_factory.resolve_host_queue_execution_dependencies",
            return_value=queue_dependencies,
        ) as resolve_deps,
        patch(
            "local_workspace_application.host.background_worker_factory.create_kafka_worker",
            side_effect=_capture_worker,
        ),
    ):
        wiring = build_local_workspace_background_worker_wiring(
            manifest=LOCAL_WORKSPACE_APPLICATION_MANIFEST,
            registry_projection=projection,
            settings=settings,
        )

    resolve_deps.assert_called_once()
    assert captured["kv_store"] is kv_store
    assert captured["causal_evidence_persistence"] is causal_persistence
    assert wiring.kv_store is kv_store


def test_worker_wiring_does_not_create_lkw_specific_causal_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, projection = _projection(monkeypatch)
    kv_store = MagicMock(spec=DistributedKVStore)
    causal_persistence = MagicMock(spec=CausalEvidencePersistence)
    queue_dependencies = HostQueueExecutionDependencies(
        kv_store=kv_store,
        causal_evidence_persistence=causal_persistence,
    )

    with (
        patch(
            "local_workspace_application.host.background_worker_factory.resolve_host_queue_execution_dependencies",
            return_value=queue_dependencies,
        ),
        patch(
            "local_workspace_application.host.background_worker_factory.create_kafka_worker",
            return_value=MagicMock(),
        ),
        patch(
            "intergrax.runtime.observability.document_store_causal_evidence_persistence.wire_causal_evidence_persistence",
        ) as wire_causal,
        patch(
            "intergrax.integrations.providers.key_value_cache.redis.bundle.create_redis_kv_store",
        ) as create_redis,
    ):
        build_local_workspace_background_worker_wiring(
            manifest=LOCAL_WORKSPACE_APPLICATION_MANIFEST,
            registry_projection=projection,
            settings=settings,
        )

    create_redis.assert_not_called()
    wire_causal.assert_not_called()


def test_worker_wiring_fails_closed_when_platform_queue_dependencies_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, projection = _projection(monkeypatch)
    with (
        patch(
            "local_workspace_application.host.background_worker_factory.resolve_host_queue_execution_dependencies",
            side_effect=ValueError("queue-enabled host requires platform key_value_cache"),
        ),
        patch(
            "local_workspace_application.host.background_worker_factory.create_kafka_worker",
        ) as create_worker,
    ):
        with pytest.raises(ValueError, match="key_value_cache"):
            build_local_workspace_background_worker_wiring(
                manifest=LOCAL_WORKSPACE_APPLICATION_MANIFEST,
                registry_projection=projection,
                settings=settings,
            )
    create_worker.assert_not_called()


def test_create_kafka_worker_requires_causal_evidence_persistence_regression() -> None:
    from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_worker

    with pytest.raises(TypeError, match="causal_evidence_persistence"):
        create_kafka_worker(
            kv_store=MagicMock(spec=DistributedKVStore),
            execution_registry=MagicMock(),
        )
