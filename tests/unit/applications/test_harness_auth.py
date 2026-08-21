# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from intergrax.applications._shared.harness_auth import (
    HarnessAuthState,
    HarnessApiKeyMiddleware,
    apply_harness_auth_middleware,
    is_harness_api_key_valid,
    require_agent_platform_admin_auth,
    require_harness_api_key,
    require_harness_auth,
    resolve_harness_api_key,
)
from intergrax.applications._shared.identity_wiring import wire_application_identity
from intergrax.applications.contracts.environment_profile import IdentityProfile
from intergrax.integrations.contracts.identity_provider import (
    AGENT_PLATFORM_ADMIN_ROLE,
    AGENT_PLATFORM_ADMIN_SCOPE,
    IdentityUser,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class _FakeIdentityProvider:
    def __init__(
        self,
        *,
        valid_token: str,
        user_id: str = "user-1",
        roles: tuple[str, ...] = (),
        scopes: tuple[str, ...] = (),
    ) -> None:
        self._valid_token = valid_token
        self._user_id = user_id
        self._roles = roles
        self._scopes = scopes

    def verify_token(self, token: str) -> IdentityUser:
        if token != self._valid_token:
            raise ValueError("invalid token")
        return IdentityUser(
            user_id=self._user_id,
            roles=self._roles,
            scopes=self._scopes,
        )

    def userinfo(self, token: str) -> IdentityUser:
        return self.verify_token(token)

    def list_tenants(self, *, limit: int = 50) -> tuple[()]:
        return ()


def _idp_only_auth_state(*, valid_token: str = "valid-idp-token") -> HarnessAuthState:
    return HarnessAuthState(
        identity_provider=_FakeIdentityProvider(valid_token=valid_token),
        require_api_key=True,
        resolved_api_key=None,
    )


def _api_key_only_auth_state(*, api_key: str = "lab-key") -> HarnessAuthState:
    return HarnessAuthState(
        require_api_key=True,
        resolved_api_key=api_key,
    )


def _local_dev_auth_state() -> HarnessAuthState:
    return HarnessAuthState(
        require_api_key=False,
        resolved_api_key=None,
        identity_provider=None,
    )


def test_resolve_harness_api_key_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    assert resolve_harness_api_key() is None
    assert is_harness_api_key_valid() is True


def test_resolve_harness_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "secret")
    assert resolve_harness_api_key() == "secret"
    assert is_harness_api_key_valid(x_api_key="secret") is True
    assert is_harness_api_key_valid(x_api_key="wrong") is False
    assert is_harness_api_key_valid(authorization="Bearer secret") is True


def test_require_harness_api_key_dependency_allows_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()

    @app.get("/probe", dependencies=[Depends(require_harness_api_key)])
    def probe() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/probe").status_code == 200


def test_require_harness_api_key_dependency_rejects_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "lab-key")
    app = FastAPI()

    @app.get("/probe", dependencies=[Depends(require_harness_api_key)])
    def probe() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/probe").status_code == 401
    assert client.get("/probe", headers={"X-Api-Key": "lab-key"}).status_code == 200


def test_require_agent_platform_admin_auth_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()

    @app.get("/probe", dependencies=[Depends(require_agent_platform_admin_auth)])
    def probe() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/probe").status_code == 401


def test_require_agent_platform_admin_auth_allows_dev_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(require_api_key=False)

    @app.get("/probe", dependencies=[Depends(require_agent_platform_admin_auth)])
    def probe() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/probe").status_code == 200


def test_sec_bnd_01_custom_env_key_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_INTERGRAX_KEY", "custom-secret")
    app = FastAPI()
    wire_application_identity(
        app,
        IdentityProfile(
            require_api_key=True,
            api_key_env="CUSTOM_INTERGRAX_KEY",
        ),
    )
    apply_harness_auth_middleware(app, require_auth=True)

    @app.get("/protected")
    def protected() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/protected").status_code == 401
    assert client.get("/protected", headers={"X-Api-Key": "custom-secret"}).status_code == 200
    assert client.get("/protected", headers={"X-Api-Key": "wrong"}).status_code == 401


def test_sec_bnd_01_default_env_does_not_override_custom_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUSTOM_INTERGRAX_KEY", "custom-secret")
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "default-secret")
    app = FastAPI()
    wire_application_identity(
        app,
        IdentityProfile(
            require_api_key=True,
            api_key_env="CUSTOM_INTERGRAX_KEY",
        ),
    )
    apply_harness_auth_middleware(app, require_auth=True)

    @app.get("/protected")
    def protected() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/protected", headers={"X-Api-Key": "custom-secret"}).status_code == 200
    assert client.get("/protected", headers={"X-Api-Key": "default-secret"}).status_code == 401


def test_sec_bnd_01_required_auth_fails_closed_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUSTOM_INTERGRAX_KEY", raising=False)
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()
    with pytest.raises(ValueError, match="CUSTOM_INTERGRAX_KEY is required"):
        wire_application_identity(
            app,
            IdentityProfile(
                require_api_key=True,
                api_key_env="CUSTOM_INTERGRAX_KEY",
            ),
        )


def test_sec_bnd_02_ordinary_idp_user_is_not_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(valid_token="ordinary-token"),
        require_api_key=True,
    )

    @app.get("/admin", dependencies=[Depends(require_agent_platform_admin_auth)])
    def admin_probe() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/admin").status_code == 401
    assert (
        client.get("/admin", headers={"Authorization": "Bearer ordinary-token"}).status_code
        == 403
    )


def test_sec_bnd_02_explicit_admin_role_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(
            valid_token="admin-token",
            roles=(AGENT_PLATFORM_ADMIN_ROLE,),
        ),
        require_api_key=True,
    )

    @app.get("/admin", dependencies=[Depends(require_agent_platform_admin_auth)])
    def admin_probe() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/admin", headers={"Authorization": "Bearer admin-token"}).status_code == 200


def test_sec_bnd_02_explicit_admin_scope_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(
            valid_token="scope-admin-token",
            scopes=(AGENT_PLATFORM_ADMIN_SCOPE,),
        ),
        require_api_key=True,
    )

    @app.get("/admin", dependencies=[Depends(require_agent_platform_admin_auth)])
    def admin_probe() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert (
        client.get("/admin", headers={"Authorization": "Bearer scope-admin-token"}).status_code
        == 200
    )


def test_sec_bnd_02_invalid_token_still_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(
            valid_token="admin-token",
            roles=(AGENT_PLATFORM_ADMIN_ROLE,),
        ),
        require_api_key=True,
    )

    @app.get("/admin", dependencies=[Depends(require_agent_platform_admin_auth)])
    def admin_probe() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/admin", headers={"Authorization": "Bearer wrong-token"}).status_code == 401


def test_sec_r1_idp_only_dependency_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()
    app.state.harness_auth = _idp_only_auth_state()

    @app.get("/protected", dependencies=[Depends(require_harness_auth)])
    def protected() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/protected").status_code == 401
    assert (
        client.get("/protected", headers={"Authorization": "Bearer wrong-token"}).status_code
        == 401
    )
    assert (
        client.get("/protected", headers={"Authorization": "Bearer valid-idp-token"}).status_code
        == 200
    )


def test_sec_r1_idp_only_middleware_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()
    app.state.harness_auth = _idp_only_auth_state()
    app.add_middleware(HarnessApiKeyMiddleware)

    @app.get("/protected")
    def protected() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/protected").status_code == 401
    assert (
        client.get("/protected", headers={"Authorization": "Bearer wrong-token"}).status_code
        == 401
    )
    assert (
        client.get("/protected", headers={"Authorization": "Bearer valid-idp-token"}).status_code
        == 200
    )


def test_sec_r1_api_key_only_require_harness_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()
    app.state.harness_auth = _api_key_only_auth_state()

    @app.get("/protected", dependencies=[Depends(require_harness_auth)])
    def protected() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/protected").status_code == 401
    assert client.get("/protected", headers={"X-Api-Key": "wrong-key"}).status_code == 401
    assert client.get("/protected", headers={"X-Api-Key": "lab-key"}).status_code == 200


def test_sec_r1_api_key_plus_idp_or_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()
    app.state.harness_auth = HarnessAuthState(
        identity_provider=_FakeIdentityProvider(valid_token="valid-idp-token"),
        require_api_key=True,
        resolved_api_key="lab-key",
    )

    @app.get("/protected", dependencies=[Depends(require_harness_auth)])
    def protected() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/protected").status_code == 401
    assert (
        client.get("/protected", headers={"Authorization": "Bearer wrong-token"}).status_code
        == 401
    )
    assert client.get("/protected", headers={"X-Api-Key": "lab-key"}).status_code == 200
    assert (
        client.get("/protected", headers={"Authorization": "Bearer valid-idp-token"}).status_code
        == 200
    )


def test_sec_r1_explicit_local_dev_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    app = FastAPI()
    app.state.harness_auth = _local_dev_auth_state()

    @app.get("/protected", dependencies=[Depends(require_harness_auth)])
    def protected() -> dict[str, str]:
        return {"ok": "yes"}

    client = TestClient(app)
    assert client.get("/protected").status_code == 200
