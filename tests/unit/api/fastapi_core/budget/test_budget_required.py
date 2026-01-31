from fastapi.testclient import TestClient
from fastapi import Request

from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.auth.context import AuthContext
from intergrax.fastapi_core.auth.provider import AuthProvider
from intergrax.fastapi_core.budget.policy import BudgetPolicy
from intergrax.fastapi_core.config import ApiConfig
from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.errors.error_types import ApiErrorType
from intergrax.fastapi_core.runs.models import RunResponse
from intergrax.fastapi_core.runs.store_base import RunStore


class RejectAllBudgetPolicy(BudgetPolicy):
    def check_create_run(self, context: RequestContext) -> bool:
        return False


class AllowAllAuthProvider(AuthProvider):
    def authenticate(self, request: Request) -> AuthContext:
        return AuthContext(
            is_authenticated=True,
            tenant_id="t1",
            user_id="u1",
            scopes=("runs:create",),
        )


class DummyRunStore(RunStore):
    def create(self):
        raise AssertionError("RunStore must not be reached when budget blocks")

    def get(self, run_id: str) -> RunResponse:
        raise AssertionError("Not part of this test")

    def cancel(self, run_id: str) -> RunResponse:
        raise AssertionError("Not part of this test")


def test_budget_policy_blocks_create_run() -> None:
    app = create_app(
        ApiConfig(
            run_store=DummyRunStore(),
            auth_provider=AllowAllAuthProvider(),
        )
    )

    app.dependency_overrides[BudgetPolicy] = lambda: RejectAllBudgetPolicy()

    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/runs", json={})

    assert response.status_code == 429
    body = response.json()
    assert body["error_type"] == ApiErrorType.RATE_LIMITED.value
    assert body["request_id"] is not None
