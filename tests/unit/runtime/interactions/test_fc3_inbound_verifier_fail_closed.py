# © Artur Czarnecki. All rights reserved.

"""FC-3 Part E: inbound verifier fail-closed (NT-FC3-INT1..INT12)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.core.security.security_configuration import (
    INBOUND_VERIFIER_MODE_UNSET,
    INBOUND_VERIFIER_NONE_FORBIDDEN,
    INBOUND_VERIFIER_SLACK_SECRET_MISSING,
    INBOUND_VERIFIER_SLACK_VERIFY_DISABLED,
    INBOUND_VERIFIER_TEAMS_TOKEN_MISSING,
    LAB_OPT_IN_FORBIDDEN_IN_PRODUCT,
    SecurityConfigurationError,
)
from intergrax.integrations.contracts.base import LabSelectorOptIn
from intergrax.runtime.interactions.intake_service import InteractionIntakeService
from intergrax.runtime.interactions.router import create_interaction_intake_router
from intergrax.runtime.interactions.verification.contract import NullInboundRequestVerifier
from intergrax.runtime.interactions.verification.factory import (
    ENV_INBOUND_VERIFIER,
    InboundVerifierMode,
    create_inbound_verifier,
    resolve_inbound_verifier_settings,
)

pytestmark = pytest.mark.unit

_LAB_OPT_IN = LabSelectorOptIn(
    selector=InboundVerifierMode.NONE.value,
    reason="FC-3 inbound verifier unit negative-test positive control",
)


def test_nt_fc3_int1_create_inbound_verifier_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_INBOUND_VERIFIER, raising=False)
    with pytest.raises(SecurityConfigurationError) as exc:
        create_inbound_verifier()
    assert exc.value.reason_code == INBOUND_VERIFIER_MODE_UNSET


def test_nt_fc3_int2_invalid_mode_raises_with_and_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_INBOUND_VERIFIER, "slak")
    with pytest.raises(SecurityConfigurationError):
        create_inbound_verifier()
    with pytest.raises(SecurityConfigurationError):
        create_inbound_verifier(lab_opt_in=_LAB_OPT_IN)


def test_nt_fc3_int3_slack_mode_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_INBOUND_VERIFIER, "slack")
    monkeypatch.delenv("INTERGRAX_SLACK_SIGNING_SECRET", raising=False)
    with pytest.raises(SecurityConfigurationError) as exc:
        create_inbound_verifier()
    assert exc.value.reason_code == INBOUND_VERIFIER_SLACK_SECRET_MISSING


def test_nt_fc3_int4_slack_secret_verify_disabled() -> None:
    with pytest.raises(SecurityConfigurationError) as exc:
        create_inbound_verifier(
            resolve_inbound_verifier_settings(
                mode="slack",
                slack_signing_secret="secret",
                slack_verify_enabled=False,
            ),
        )
    assert exc.value.reason_code == INBOUND_VERIFIER_SLACK_VERIFY_DISABLED


def test_nt_fc3_int5_teams_mode_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_INBOUND_VERIFIER, "teams")
    monkeypatch.delenv("INTERGRAX_TEAMS_SECURITY_TOKEN", raising=False)
    with pytest.raises(SecurityConfigurationError) as exc:
        create_inbound_verifier()
    assert exc.value.reason_code == INBOUND_VERIFIER_TEAMS_TOKEN_MISSING


def test_nt_fc3_int6_explicit_none_without_opt_in() -> None:
    with pytest.raises(SecurityConfigurationError) as exc:
        create_inbound_verifier(resolve_inbound_verifier_settings(mode="none"))
    assert exc.value.reason_code == INBOUND_VERIFIER_NONE_FORBIDDEN


def test_nt_fc3_int7_none_with_opt_in_returns_null_verifier() -> None:
    verifier = create_inbound_verifier(lab_opt_in=_LAB_OPT_IN)
    assert isinstance(verifier, NullInboundRequestVerifier)


def test_nt_fc3_int8_intake_service_verifier_none_raises() -> None:
    with pytest.raises(SecurityConfigurationError):
        InteractionIntakeService(verifier=None)


def test_nt_fc3_int9_wire_strict_without_verifier_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_INBOUND_VERIFIER, raising=False)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc3.inbound.strict")
    env = env.model_copy(
        update={"meta": env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT})},
    )
    with pytest.raises(SecurityConfigurationError):
        wire_interaction_intake_service(env=env)


def test_nt_fc3_int10_wire_product_without_verifier_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_INBOUND_VERIFIER, raising=False)
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="fc3.inbound.product")
    with pytest.raises(SecurityConfigurationError):
        wire_interaction_intake_service(env=env)


def test_nt_fc3_int11_product_env_lab_opt_in_forbidden() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="fc3.inbound.product.optin")
    with pytest.raises(SecurityConfigurationError) as exc:
        wire_interaction_intake_service(env=env, lab_opt_in=_LAB_OPT_IN)
    assert exc.value.reason_code == LAB_OPT_IN_FORBIDDEN_IN_PRODUCT


def test_nt_fc3_int12_strict_blocks_router_lab_accepts_unsigned_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_INBOUND_VERIFIER, raising=False)
    strict_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc3.inbound.router.strict")
    strict_env = strict_env.model_copy(
        update={"meta": strict_env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT})},
    )
    with pytest.raises(SecurityConfigurationError):
        wire_interaction_intake_service(env=strict_env)

    lab_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc3.inbound.router.lab")
    assert lab_env.application_profile is ApplicationProfile.LAB
    assert lab_env.execution_mode is not ExecutionMode.STRICT
    service = wire_interaction_intake_service(
        env=lab_env,
        interaction_surface="lab_json",
    )
    app = FastAPI()
    app.include_router(create_interaction_intake_router(service), prefix="/v1/interactions")
    with TestClient(app) as client:
        response = client.post(
            "/v1/interactions/intake",
            params={"tenant": "t1", "execute": "false"},
            json={
                "message": "hello",
                "capability": "echo.basic",
                "user_id": "u1",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "hello"
    assert body["capability"] == "echo.basic"
