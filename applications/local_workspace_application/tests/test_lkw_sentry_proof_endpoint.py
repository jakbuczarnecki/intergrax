# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
from intergrax.fastapi_core.config import ApiEnvironment
from intergrax.runtime.observability.operator_wiring import (
    ObservabilityExportBackendRegistry,
    ObservabilityExportOperatorConfig,
    SentryExportOperatorConfig,
)
from intergrax.runtime.observability.sentry_export_wiring import (
    build_sentry_observability_integration,
)
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.tests.lkw_ac3_projection import build_lkw_test_registry_projection
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.serving.sentry_proof_routes import (
    emit_local_workspace_sentry_proof_failure,
)

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SENTRY_COMPOSE_OVERLAY = (
    _PROJECT_ROOT
    / "applications"
    / "local_workspace_application"
    / "docker"
    / "docker-compose.sentry.yml"
)
_SENTRY_SERVICES_FRAGMENT = (
    _PROJECT_ROOT
    / "applications"
    / "local_workspace_application"
    / "docker"
    / "sentry.services.yml"
)

_FORBIDDEN_EXPORT_SAMPLES = (
    "secret prompt",
    "raw body",
    "secret-api-key",
    "tool_arguments",
    "raw_chunks",
    "/home/user/",
    "c:\\users\\",
)


class FakeSentryTransport:
    def __init__(self) -> None:
        self.payloads: list[Any] = []

    async def send_observability_payload(self, payload: object) -> None:
        self.payloads.append(payload)


def _enabled_sentry_config() -> ObservabilityExportOperatorConfig:
    return ObservabilityExportOperatorConfig(
        enabled=True,
        export_content=False,
        backend_id="sentry",
        sentry=SentryExportOperatorConfig(
            dsn="http://example@relay:3000/1",
            environment="local-proof",
            release="lkw-sentry-proof",
            server_name="intergrax-lkw-local",
            shutdown_timeout_seconds=2.0,
            flush_after_capture=True,
        ),
    )


def _sentry_registry_with_transport(
    transport: FakeSentryTransport,
) -> ObservabilityExportBackendRegistry:
    registry = ObservabilityExportBackendRegistry()
    registry.register(
        "sentry",
        lambda config: build_sentry_observability_integration(config, transport=transport),
    )
    return registry


def test_sentry_compose_overlay_is_local_docker_proof() -> None:
    overlay = _SENTRY_COMPOSE_OVERLAY.read_text(encoding="utf-8")
    services = _SENTRY_SERVICES_FRAGMENT.read_text(encoding="utf-8")

    assert "INTERGRAX_SENTRY_DSN" not in overlay
    assert "LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND: sentry" in overlay
    assert 'LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_CONTENT: "false"' in overlay
    assert "9000:80" in services
    assert "getsentry/sentry:24.8.0" in services
    assert "sentry-bootstrap" in services
    assert "sentry-upgrade" in services
    assert "SENTRY_SECRET_KEY" in services
    # Host proof directory is mounted read-only; container reads /proof/generated.env.
    assert re.search(r"\./sentry-proof:/proof:ro", overlay)
    start_script = (
        Path(__file__).resolve().parents[1]
        / "docker"
        / "start-local-workspace-sentry-proof.sh"
    ).read_text(encoding="utf-8")
    assert "/proof/generated.env" in start_script


def test_sentry_compose_overlay_does_not_require_external_dsn() -> None:
    overlay = _SENTRY_COMPOSE_OVERLAY.read_text(encoding="utf-8")
    assert "${INTERGRAX_SENTRY_DSN" not in overlay


@pytest.mark.asyncio
async def test_proof_endpoint_emits_controlled_problem_via_platform_path() -> None:
    transport = FakeSentryTransport()
    settings = LocalWorkspaceBackendSettings(
        environment=ApiEnvironment.DEV,
        default_agent_id="local_search",
    )
    response = await emit_local_workspace_sentry_proof_failure(
        settings=settings,
        observability_export=_enabled_sentry_config(),
        run_id="run-proof-endpoint",
        correlation_id="corr-proof-endpoint",
        registry=_sentry_registry_with_transport(transport),
    )

    assert response.proof_result == "PASS"
    assert response.problem_kind == "lkw.proof_controlled_failure"
    assert response.problem_error_code == "LKW_PROOF_CONTROLLED_FAILURE"
    assert response.safety_check == "passed"
    assert len(transport.payloads) == 1
    payload = transport.payloads[0]
    assert payload.problem_kind == "lkw.proof_controlled_failure"
    assert payload.problem_error_code == "LKW_PROOF_CONTROLLED_FAILURE"
    serialized = json.dumps(payload.model_dump(mode="json"))
    for sample in _FORBIDDEN_EXPORT_SAMPLES:
        assert sample not in serialized.lower()


def test_proof_http_route_reaches_fake_sentry_transport() -> None:
    transport = FakeSentryTransport()
    config = _enabled_sentry_config()

    with (
        patch(
            "local_workspace_application.host.factory.build_local_workspace_observability_plugins",
            return_value=(),
        ),
        patch(
            "local_workspace_application.serving.sentry_proof_routes.build_observability_export_integration",
            side_effect=lambda cfg, registry=None: build_sentry_observability_integration(
                cfg,
                transport=transport,
            ),
        ),
    ):
        app = create_local_workspace_backend_app(registry_projection=build_lkw_test_registry_projection(LocalWorkspaceBackendSettings(
                environment=ApiEnvironment.DEV), settings=LocalWorkspaceBackendSettings(
                environment=ApiEnvironment.DEV, default_agent_id="local_search",),
            observability_export=config,
        )
        client = TestClient(app)
        response = client.post(
            "/v1/local_workspace/proof/sentry-error",
            json={"run_id": "run-http-proof", "correlation_id": "corr-http-proof"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["proof_result"] == "PASS"
    assert body["backend"] == "sentry"
    assert body["problem_kind"] == "lkw.proof_controlled_failure"
    assert body["problem_error_code"] == "LKW_PROOF_CONTROLLED_FAILURE"
    assert body["safety_check"] == "passed"
    assert len(transport.payloads) == 1


def test_proof_route_disabled_in_prod_environment() -> None:
    with patch(
        "local_workspace_application.host.factory.build_local_workspace_observability_plugins",
        return_value=(),
    ):
        app = create_local_workspace_backend_app(registry_projection=build_lkw_test_registry_projection(LocalWorkspaceBackendSettings(
                environment=ApiEnvironment.PROD), settings=LocalWorkspaceBackendSettings(
                environment=ApiEnvironment.PROD, default_agent_id="local_search",
                api_keys_map={
                    "proof-key": ApiKeyIdentity(tenant_id="t", user_id="u", scopes=("*",))
                },
            ),
            observability_export=None,
        )
    client = TestClient(app)
    response = client.post("/v1/local_workspace/proof/sentry-error", json={})
    assert response.status_code == 404


def test_sentry_proof_routes_do_not_import_sentry_sdk() -> None:
    source = (
        _PROJECT_ROOT
        / "applications"
        / "local_workspace_application"
        / "serving"
        / "sentry_proof_routes.py"
    ).read_text(encoding="utf-8")
    forbidden = ("sentry_sdk", "integrations.providers.observability_backend.sentry")
    for token in forbidden:
        assert token not in source
