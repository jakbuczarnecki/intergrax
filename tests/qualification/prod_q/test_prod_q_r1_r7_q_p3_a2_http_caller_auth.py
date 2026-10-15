# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-P3-A2 — strict HTTP caller authentication and tenant authorization."""

from __future__ import annotations

from dataclasses import replace
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.serving import fastapi_router as gca_fastapi_router
from intergrax.fastapi_core.protocol import ApiHeaders
from intergrax.runtime.task.task import TaskResult, TaskState
from tests.qualification.prod_q._prod_q_strict_http_caller_auth import (
    _QUAL_API_KEY,
    _QUAL_USER,
    _WRONG_TENANT_KEY,
    strict_http_auth_headers,
    with_strict_http_api_keys,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_p2_c_http_execution_admission import (
    _INNER_DELEGATE,
    _build_strict_http_fixture,
    _completed_task_result,
    _run_request_json,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "77fe07634b63ff16357f4570b88763d23f713fbd"
_COLOCATED_ADMIT = (
    "intergrax.applications._shared.colocated_serving_unit_task_execution."
    "ColocatedServingUnitAdmissionTaskExecutionPort.execute"
)


def test_p3a2_g00_qualification_baseline_ancestor() -> None:
    import subprocess as sp

    from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import repo_root

    sp.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def test_p3a2_t01_anonymous_run_denied_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p3a2-t01-anon",
    )
    inner_mock = AsyncMock(return_value=_completed_task_result())
    try:
        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                )
        assert response.status_code == 401
        inner_mock.assert_not_called()
    finally:
        if not fixture.session._closed:
            fixture.session.close()


def test_p3a2_t02_invalid_api_key_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p3a2-t02-invalid-key",
    )
    inner_mock = AsyncMock(return_value=_completed_task_result())
    try:
        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers={ApiHeaders.API_KEY: "not-a-valid-key"},
                )
        assert response.status_code == 401
        inner_mock.assert_not_called()
    finally:
        if not fixture.session._closed:
            fixture.session.close()


def test_p3a2_t03_authenticated_matching_tenant_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p3a2-t03-positive",
    )
    inner_mock = AsyncMock(return_value=_completed_task_result())
    try:
        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
        assert response.status_code == 200
        assert response.json()["state"] == TaskState.COMPLETED.value
        assert inner_mock.call_count == 1
    finally:
        if not fixture.session._closed:
            fixture.session.close()


def test_p3a2_t04_wrong_tenant_api_key_denied_before_admission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p3a2-t04-wrong-tenant-key",
    )
    inner_mock = AsyncMock(return_value=_completed_task_result())
    try:
        with patch(_INNER_DELEGATE, inner_mock), patch(_COLOCATED_ADMIT, inner_mock):
            with TestClient(fixture.app) as client:
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(_WRONG_TENANT_KEY),
                )
        assert response.status_code == 403
        inner_mock.assert_not_called()
    finally:
        if not fixture.session._closed:
            fixture.session.close()


def test_p3a2_t05_forged_tenant_header_without_auth_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p3a2-t05-forged-header",
    )
    inner_mock = AsyncMock(return_value=_completed_task_result())
    try:
        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers={"X-Tenant-Id": fixture.tenant_id},
                )
        assert response.status_code == 401
        inner_mock.assert_not_called()
    finally:
        if not fixture.session._closed:
            fixture.session.close()


def test_p3a2_t06_admitted_principal_reflects_api_key_user_not_host_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p3a2-t06-principal",
    )
    captured_principal_ids: list[str] = []
    real_identity = gca_fastapi_router.request_identity_from_strict_production_http_auth

    def _capturing_identity(auth):
        principal = real_identity(auth)
        captured_principal_ids.append(principal.user_id)
        return principal

    inner_mock = AsyncMock(return_value=_completed_task_result())
    try:
        with patch(_INNER_DELEGATE, inner_mock), patch.object(
            gca_fastapi_router,
            "request_identity_from_strict_production_http_auth",
            side_effect=_capturing_identity,
        ):
            with TestClient(fixture.app) as client:
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                    headers=strict_http_auth_headers(),
                )
        assert response.status_code == 200
        assert captured_principal_ids == [_QUAL_USER]
        assert captured_principal_ids[0] != "p3a-production-launcher"
    finally:
        if not fixture.session._closed:
            fixture.session.close()


def test_p3a2_t07_body_tenant_tamper_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p3a2-t07-body-tamper",
    )
    inner_mock = AsyncMock(return_value=_completed_task_result())
    try:
        with patch(_INNER_DELEGATE, inner_mock):
            with TestClient(fixture.app) as client:
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json("forged-tenant-id"),
                    headers=strict_http_auth_headers(),
                )
        assert response.status_code == 400
        inner_mock.assert_not_called()
    finally:
        if not fixture.session._closed:
            fixture.session.close()


def test_p3a2_t10_caller_deny_skips_colocated_admission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p3a2-t10-admission",
    )
    colocated_mock = AsyncMock(return_value=_completed_task_result())
    try:
        with patch(_COLOCATED_ADMIT, colocated_mock):
            with TestClient(fixture.app) as client:
                response = client.post(
                    fixture.run_url,
                    json=_run_request_json(fixture.tenant_id),
                )
        assert response.status_code == 401
        colocated_mock.assert_not_called()
    finally:
        if not fixture.session._closed:
            fixture.session.close()


def test_p3a2_t11_agents_route_requires_authentication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = _build_strict_http_fixture(
        monkeypatch,
        tmp_path,
        revision_id="rev-p3a2-t11-agents",
    )
    prefix = fixture.run_url.rsplit("/run", 1)[0]
    try:
        with TestClient(fixture.app) as client:
            anon = client.get(f"{prefix}/agents")
            assert anon.status_code == 401
            authed = client.get(
                f"{prefix}/agents",
                headers=strict_http_auth_headers(),
            )
            assert authed.status_code == 200
    finally:
        if not fixture.session._closed:
            fixture.session.close()


