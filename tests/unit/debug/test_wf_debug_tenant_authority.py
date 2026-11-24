# © Artur Czarnecki. All rights reserved.

"""PT-WF — debug router authenticated tenant authority (ARCH-FREEZE AF-W3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.debug.app import create_debug_app
from intergrax.debug.router import create_debug_router
from intergrax.experiments.models import ExperimentDecision, RegisterExperimentRequest
from intergrax.experiments.persistence_contract import ExperimentPersistence
from intergrax.integrations.contracts.identity_provider import IdentityUser
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import RunMetadata, RunStats

pytestmark = pytest.mark.unit


class _TenantIdentityProvider:
    def __init__(self, *, valid_token: str, tenant_id: str, user_id: str = "user-1") -> None:
        self._valid_token = valid_token
        self._tenant_id = tenant_id
        self._user_id = user_id

    def verify_token(self, token: str) -> IdentityUser:
        if token != self._valid_token:
            raise ValueError("invalid token")
        return IdentityUser(user_id=self._user_id, tenant_id=self._tenant_id)

    def userinfo(self, token: str) -> IdentityUser:
        return self.verify_token(token)

    def list_tenants(self, *, limit: int = 50) -> tuple[()]:
        return ()


@dataclass
class _SpyExperimentStore:
    """Minimal experiment persistence spy for tenant rejection tests."""

    backend: ExperimentPersistence

    def list_experiments(self, **kwargs: Any):
        return self.backend.list_experiments(**kwargs)

    def get(self, experiment_id: str, *, tenant_id: str | None = None):
        return self.backend.get(experiment_id, tenant_id=tenant_id)

    def register(self, request: RegisterExperimentRequest):
        return self.backend.register(request)

    def set_decision(self, experiment_id: str, decision: ExperimentDecision, **kwargs: Any):
        return self.backend.set_decision(experiment_id, decision, **kwargs)

    def link_run(self, experiment_id: str, run_id: str, **kwargs: Any):
        return self.backend.link_run(experiment_id, run_id, **kwargs)


class _RecordingExperimentStore(_SpyExperimentStore):
    calls: list[str]

    def __init__(self, backend: ExperimentPersistence) -> None:
        super().__init__(backend)
        self.calls = []

    def list_experiments(self, **kwargs: Any):
        self.calls.append("list_experiments")
        return super().list_experiments(**kwargs)

    def get(self, experiment_id: str, *, tenant_id: str | None = None):
        self.calls.append("get")
        return super().get(experiment_id, tenant_id=tenant_id)

    def register(self, request: RegisterExperimentRequest):
        self.calls.append("register")
        return super().register(request)

    def set_decision(self, experiment_id: str, decision: ExperimentDecision, **kwargs: Any):
        self.calls.append("set_decision")
        return super().set_decision(experiment_id, decision, **kwargs)

    def link_run(self, experiment_id: str, run_id: str, **kwargs: Any):
        self.calls.append("link_run")
        return super().link_run(experiment_id, run_id, **kwargs)


def _auth_app(
    *,
    harness_auth: HarnessAuthState | None = None,
    trace_store: InMemoryRunTraceStore | None = None,
    experiment_store: ExperimentPersistence | None = None,
) -> FastAPI:
    trace = trace_store or InMemoryRunTraceStore()
    app = FastAPI()
    if harness_auth is not None:
        app.state.harness_auth = harness_auth
    app.include_router(
        create_debug_router(
            trace_store=trace,
            experiment_store=experiment_store,
        )
    )
    return app


def _seed_trace(store: InMemoryRunTraceStore, *, run_id: str, tenant_id: str) -> None:
    store.finalize_run(
        run_id,
        RunMetadata(
            run_id=run_id,
            session_id="s1",
            user_id="u1",
            tenant_id=tenant_id,
            started_at_utc="2026-01-01T00:00:00Z",
            stats=RunStats(duration_ms=1, llm_usage={}),
        ),
    )


@pytest.fixture
def monkeypatch_harness_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    monkeypatch.delenv("CUSTOM_INTERGRAX_KEY", raising=False)


def test_wf_debug_authenticated_implicit_tenant_is_bound(tmp_path, monkeypatch_harness_env: None) -> None:
    experiments_db = tmp_path / "exp.db"
    app = create_debug_app(experiments_db_path=experiments_db)
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_TenantIdentityProvider(valid_token="tok", tenant_id="tenant-a"),
        require_api_key=True,
        resolved_api_key=None,
    )
    with TestClient(app) as client:
        response = client.post(
            "/debug/experiments",
            headers={"Authorization": "Bearer tok"},
            json={"hypothesis": "h", "capability": "echo.basic"},
        )
    assert response.status_code == 201, response.text
    assert response.json()["tenant_id"] == "tenant-a"


def test_wf_debug_authenticated_foreign_query_tenant_rejected(
    monkeypatch_harness_env: None,
) -> None:
    store = InMemoryRunTraceStore()
    _seed_trace(store, run_id="run-1", tenant_id="tenant-a")
    recording = _RecordingExperimentStore(
        _SpyExperimentStore(
            backend=type(
                "Empty",
                (),
                {
                    "list_experiments": lambda *a, **k: [],
                    "get": lambda *a, **k: (_ for _ in ()).throw(ValueError("missing")),
                    "register": lambda *a, **k: (_ for _ in ()).throw(AssertionError("register")),
                    "set_decision": lambda *a, **k: (_ for _ in ()).throw(AssertionError("set_decision")),
                    "link_run": lambda *a, **k: (_ for _ in ()).throw(AssertionError("link_run")),
                },
            )()
        )
    )
    app = _auth_app(
        harness_auth=HarnessAuthState(
            api_key_principal_tenant_id="tenant-a",
            require_api_key=True,
            resolved_api_key="secret",
        ),
        trace_store=store,
        experiment_store=recording,
    )
    with TestClient(app) as client:
        response = client.get(
            "/debug/tasks",
            headers={"X-Api-Key": "secret"},
            params={"tenant": "tenant-b"},
        )
    assert response.status_code == 403
    assert recording.calls == []


def test_wf_debug_authenticated_foreign_body_tenant_rejected(tmp_path, monkeypatch_harness_env: None) -> None:
    app = create_debug_app(experiments_db_path=tmp_path / "exp.db")
    app.state.harness_auth = HarnessAuthState(
        api_key_principal_tenant_id="tenant-a",
        require_api_key=True,
        resolved_api_key="secret",
    )
    with TestClient(app) as client:
        response = client.post(
            "/debug/experiments",
            headers={"X-Api-Key": "secret"},
            json={
                "tenant_id": "tenant-b",
                "hypothesis": "h",
                "capability": "echo.basic",
            },
        )
    assert response.status_code == 403


def test_wf_debug_authenticated_matching_tenant_allowed(tmp_path, monkeypatch_harness_env: None) -> None:
    app = create_debug_app(experiments_db_path=tmp_path / "exp.db")
    app.state.harness_auth = HarnessAuthState(
        api_key_principal_tenant_id="tenant-a",
        require_api_key=True,
        resolved_api_key="secret",
    )
    with TestClient(app) as client:
        response = client.post(
            "/debug/experiments",
            headers={"X-Api-Key": "secret"},
            params={"tenant": "tenant-a"},
            json={"hypothesis": "h", "capability": "echo.basic"},
        )
    assert response.status_code == 201


def test_wf_debug_tenant_required_without_principal_rejected(monkeypatch_harness_env: None) -> None:
    store = InMemoryRunTraceStore()
    _seed_trace(store, run_id="run-1", tenant_id="tenant-a")
    app = _auth_app(
        harness_auth=HarnessAuthState(
            require_api_key=True,
            resolved_api_key="secret",
            tenant_required=True,
            api_key_principal_tenant_id=None,
        ),
        trace_store=store,
    )
    with TestClient(app) as client:
        response = client.get(
            "/debug/tasks",
            headers={"X-Api-Key": "secret"},
            params={"tenant": "tenant-b"},
        )
    assert response.status_code == 403


def test_wf_debug_unbound_api_key_cannot_select_foreign_tenant(monkeypatch_harness_env: None) -> None:
    store = InMemoryRunTraceStore()
    _seed_trace(store, run_id="run-1", tenant_id="default")
    app = _auth_app(
        harness_auth=HarnessAuthState(
            require_api_key=True,
            resolved_api_key="secret",
            api_key_principal_tenant_id=None,
        ),
        trace_store=store,
    )
    with TestClient(app) as client:
        response = client.get(
            "/debug/tasks",
            headers={"X-Api-Key": "secret"},
            params={"tenant": "foreign-tenant"},
        )
    assert response.status_code == 403


def test_wf_debug_identity_provider_tenant_authority_enforced(tmp_path, monkeypatch_harness_env: None) -> None:
    app = create_debug_app(experiments_db_path=tmp_path / "exp.db")
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_TenantIdentityProvider(valid_token="tok", tenant_id="tenant-a"),
        require_api_key=True,
    )
    with TestClient(app) as client:
        foreign = client.get(
            "/debug/experiments",
            headers={"Authorization": "Bearer tok"},
            params={"tenant": "tenant-b"},
        )
        assert foreign.status_code == 403
        allowed = client.get(
            "/debug/experiments",
            headers={"Authorization": "Bearer tok"},
            params={"tenant": "tenant-a"},
        )
        assert allowed.status_code == 200


def test_wf_debug_experiment_cross_tenant_mutations_rejected(tmp_path, monkeypatch_harness_env: None) -> None:
    app = create_debug_app(experiments_db_path=tmp_path / "exp.db")
    app.state.harness_auth = HarnessAuthState(
        api_key_principal_tenant_id="tenant-a",
        require_api_key=True,
        resolved_api_key="secret",
    )
    with TestClient(app) as client:
        created = client.post(
            "/debug/experiments",
            headers={"X-Api-Key": "secret"},
            json={"hypothesis": "h", "capability": "echo.basic"},
        ).json()
    experiment_id = created["experiment_id"]
    with TestClient(app) as client:
        decide = client.post(
            f"/debug/experiments/{experiment_id}/decision",
            headers={"X-Api-Key": "secret"},
            params={"tenant": "tenant-b"},
            json={"decision": ExperimentDecision.KEEP.value},
        )
        link = client.post(
            f"/debug/experiments/{experiment_id}/runs/run-1",
            headers={"X-Api-Key": "secret"},
            params={"tenant": "tenant-b"},
        )
        show = client.get(
            f"/debug/experiments/{experiment_id}",
            headers={"X-Api-Key": "secret"},
            params={"tenant": "tenant-b"},
        )
    assert decide.status_code == 403
    assert link.status_code == 403
    assert show.status_code == 403


def test_wf_debug_human_response_foreign_tenant_rejected(monkeypatch_harness_env: None) -> None:
    store = InMemoryRunTraceStore()
    app = _auth_app(
        harness_auth=HarnessAuthState(
            api_key_principal_tenant_id="tenant-a",
            require_api_key=True,
            resolved_api_key="secret",
        ),
        trace_store=store,
    )
    with TestClient(app) as client:
        response = client.post(
            "/debug/tasks/task-1/human-response",
            headers={"X-Api-Key": "secret"},
            params={"tenant": "tenant-b"},
            json={"response": "approve"},
        )
    assert response.status_code == 403


def test_wf_debug_laboratory_default_preserved(tmp_path, monkeypatch_harness_env: None) -> None:
    app = create_debug_app(experiments_db_path=tmp_path / "exp.db")
    with TestClient(app) as client:
        response = client.post(
            "/debug/experiments",
            json={"hypothesis": "lab", "capability": "echo.basic"},
        )
    assert response.status_code == 201


_TENANT_SCOPED_ROUTES = [
    ("GET", "/debug/tasks", {}),
    ("GET", "/debug/tasks/run-1", {}),
    ("GET", "/debug/tasks/run-1/trace", {}),
    ("GET", "/debug/tasks/run-1/metrics", {}),
    ("GET", "/debug/tasks/task-1/events", {}),
    ("GET", "/debug/tasks/task-1/checkpoints", {}),
    ("GET", "/debug/tasks/task-1/progress", {}),
    ("POST", "/debug/tasks/task-1/human-response", {"json": {"response": "approve"}}),
    ("GET", "/debug/experiments", {}),
    ("POST", "/debug/experiments", {"json": {"hypothesis": "h", "capability": "cap"}}),
    ("GET", "/debug/experiments/exp-1", {}),
    ("POST", "/debug/experiments/exp-1/decision", {"json": {"decision": "keep"}}),
    ("POST", "/debug/experiments/exp-1/runs/run-1", {}),
]


@pytest.mark.parametrize("method,path,extra", _TENANT_SCOPED_ROUTES)
def test_wf_debug_handlers_reject_foreign_query_tenant(
    method: str,
    path: str,
    extra: dict[str, Any],
    monkeypatch_harness_env: None,
) -> None:
    store = InMemoryRunTraceStore()
    _seed_trace(store, run_id="run-1", tenant_id="tenant-a")
    recording = _RecordingExperimentStore(
        _SpyExperimentStore(
            backend=type(
                "Empty",
                (),
                {
                    "list_experiments": lambda *a, **k: [],
                    "get": lambda *a, **k: (_ for _ in ()).throw(ValueError("missing")),
                    "register": lambda *a, **k: (_ for _ in ()).throw(AssertionError("register")),
                    "set_decision": lambda *a, **k: (_ for _ in ()).throw(AssertionError("set_decision")),
                    "link_run": lambda *a, **k: (_ for _ in ()).throw(AssertionError("link_run")),
                },
            )()
        )
    )
    app = _auth_app(
        harness_auth=HarnessAuthState(
            api_key_principal_tenant_id="tenant-a",
            require_api_key=True,
            resolved_api_key="secret",
        ),
        trace_store=store,
        experiment_store=recording,
    )
    with TestClient(app) as client:
        request = getattr(client, method.lower())
        response = request(
            path,
            headers={"X-Api-Key": "secret"},
            params={"tenant": "tenant-b"},
            **extra,
        )
    assert response.status_code == 403, (method, path, response.status_code, response.text)
    assert recording.calls == []
