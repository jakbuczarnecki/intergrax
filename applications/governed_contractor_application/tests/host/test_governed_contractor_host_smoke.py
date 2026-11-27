# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from intergrax.applications._shared.production_execution_continuation_store import (
    open_production_execution_continuation_state_store,
)
from intergrax.runtime.execution.continuation.durable_serialized_seal_store import (
    SerializedSealExecutionContinuationStateStore,
)
from testing_support.host_fixture_wiring import (
    install_diagnostic_cursor_secret,
    install_host_llm_stub,
    reference_host_document_store,
)

from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/governed_contractor"


@pytest.fixture
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    install_host_llm_stub(monkeypatch)


@pytest.fixture
def _diagnostic_cursor_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    install_diagnostic_cursor_secret(monkeypatch)


@pytest.fixture(autouse=True)
def _gca_smoke_otlp_qualification_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.applications._shared.package_wiring import (
        PACKAGE_GATE_OTLP_QUALIFICATION_ENDPOINT,
    )

    from governed_contractor_application.host.environment_profile import (
        build_governed_contractor_environment_profile,
    )

    def _profile_with_qualification_otlp(settings):
        profile = build_governed_contractor_environment_profile(settings)
        return profile.model_copy(
            update={
                "observability_profile": profile.observability_profile.model_copy(
                    update={
                        "otlp_export_endpoint": PACKAGE_GATE_OTLP_QUALIFICATION_ENDPOINT,
                    },
                ),
            },
        )

    monkeypatch.setattr(
        "governed_contractor_application.host.factory.build_governed_contractor_environment_profile",
        _profile_with_qualification_otlp,
    )


@pytest.fixture
def execution_continuation_state_store(
    tmp_path: Path,
) -> Iterator[SerializedSealExecutionContinuationStateStore]:
    store = open_production_execution_continuation_state_store(
        tmp_path / "continuation.json",
        missing_file_policy="empty",
    )
    try:
        yield store
    finally:
        store.close()


def test_governed_contractor_backend_health(
    _stub_host_llm: None,
    _diagnostic_cursor_secret: None,
    execution_continuation_state_store: SerializedSealExecutionContinuationStateStore,
):
    with TestClient(
        create_governed_contractor_backend_app(
            registry_projection=build_governed_contractor_test_registry_projection(),
            document_store=reference_host_document_store(),
            execution_continuation_state_store=execution_continuation_state_store,
        )
    ) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_governed_contractor_backend_lists_agents(
    _stub_host_llm: None,
    _diagnostic_cursor_secret: None,
    execution_continuation_state_store: SerializedSealExecutionContinuationStateStore,
):
    with TestClient(
        create_governed_contractor_backend_app(
            registry_projection=build_governed_contractor_test_registry_projection(),
            document_store=reference_host_document_store(),
            execution_continuation_state_store=execution_continuation_state_store,
        )
    ) as client:
        response = client.get(f"{_PREFIX}/agents")
        assert response.status_code == 200
        assert "agents" in response.json()


def test_governed_contractor_backend_run(
    _stub_host_llm: None,
    _diagnostic_cursor_secret: None,
    execution_continuation_state_store: SerializedSealExecutionContinuationStateStore,
):
    with TestClient(
        create_governed_contractor_backend_app(
            registry_projection=build_governed_contractor_test_registry_projection(),
            document_store=reference_host_document_store(),
            execution_continuation_state_store=execution_continuation_state_store,
        )
    ) as client:
        response = client.post(
            f"{_PREFIX}/run",
            json={"message": "hello", "capability": "external_contractor.adapt"},
        )
        assert response.status_code == 200
        assert response.json().get("state") == "completed"
