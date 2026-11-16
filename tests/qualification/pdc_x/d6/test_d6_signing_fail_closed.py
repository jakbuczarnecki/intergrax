# © Artur Czarnecki. All rights reserved.

"""PDC-D6-R008: missing signing secret fails closed outside lab; no literal fallback."""

from __future__ import annotations

import ast
import inspect

import pytest
from intergrax.applications._shared import critical_action_signing_wiring as wiring_module
from intergrax.applications._shared.critical_action_signing_wiring import (
    LabEphemeralSigningOptIn,
    resolve_critical_action_signing_wiring,
)
from intergrax.core.security.raw_secret_gate import KNOWN_FORBIDDEN_LITERALS
from intergrax.core.security.security_configuration import (
    LAB_OPT_IN_FORBIDDEN_IN_PRODUCT,
    SIGNING_SECRET_ENV_INVALID,
    SIGNING_SECRET_MISSING,
    SecurityConfigurationError,
)

from ._support import generated_secret, lab_env, product_env

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SECRET_ENV = "INTERGRAX_CRITICAL_ACTION_SIGNING_KEY"


def test_d6_r008_missing_signing_secret_fails_closed_in_product_profile() -> None:
    with pytest.raises(SecurityConfigurationError) as excinfo:
        resolve_critical_action_signing_wiring(product_env(), environ={})
    assert excinfo.value.reason_code == SIGNING_SECRET_MISSING
    assert excinfo.value.item == "critical_action_signing"


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_d6_r008_blank_signing_secret_fails_closed(blank: str) -> None:
    with pytest.raises(SecurityConfigurationError) as excinfo:
        resolve_critical_action_signing_wiring(product_env(), environ={_SECRET_ENV: blank})
    assert excinfo.value.reason_code == SIGNING_SECRET_MISSING


def test_d6_r008_blank_secret_env_name_is_malformed() -> None:
    env = product_env()
    env.identity_profile = env.identity_profile.model_copy(update={"critical_action_signing_secret_env": "  "})
    with pytest.raises(SecurityConfigurationError) as excinfo:
        resolve_critical_action_signing_wiring(env, environ={})
    assert excinfo.value.reason_code == SIGNING_SECRET_ENV_INVALID


def test_d6_r008_configured_secret_signs_and_verifies() -> None:
    secret = generated_secret()
    result = resolve_critical_action_signing_wiring(product_env(), environ={_SECRET_ENV: secret})
    assert result.enabled is True
    assert result.bootstrap_signature is not None
    assert secret not in repr(result)


def test_d6_r008_error_message_never_contains_a_secret_value() -> None:
    secret = generated_secret()
    with pytest.raises(SecurityConfigurationError) as excinfo:
        resolve_critical_action_signing_wiring(product_env(), environ={"UNRELATED_VARIABLE": secret})
    assert secret not in str(excinfo.value)


def test_d6_r008_lab_profile_is_disabled_without_explicit_opt_in() -> None:
    result = resolve_critical_action_signing_wiring(lab_env(), environ={})
    assert result.enabled is False
    assert result.bootstrap_signature is None


def test_d6_r008_lab_opt_in_signs_with_a_random_per_process_key() -> None:
    opt_in = LabEphemeralSigningOptIn(reason="lab proof run")
    first = resolve_critical_action_signing_wiring(lab_env(), environ={}, lab_opt_in=opt_in)
    second = resolve_critical_action_signing_wiring(lab_env(), environ={}, lab_opt_in=opt_in)
    assert first.enabled is True and first.bootstrap_signature is not None
    # random key per call: signatures differ, so no shared literal key exists
    assert first.bootstrap_signature != second.bootstrap_signature


def test_d6_r008_lab_opt_in_requires_a_reason() -> None:
    with pytest.raises(ValueError):
        LabEphemeralSigningOptIn(reason=" ")


def test_d6_r008_lab_opt_in_is_rejected_on_a_product_profile() -> None:
    with pytest.raises(SecurityConfigurationError) as excinfo:
        resolve_critical_action_signing_wiring(
            product_env(),
            environ={_SECRET_ENV: generated_secret()},
            lab_opt_in=LabEphemeralSigningOptIn(reason="must not be honoured"),
        )
    assert excinfo.value.reason_code == LAB_OPT_IN_FORBIDDEN_IN_PRODUCT


def test_d6_r008_secret_is_read_at_call_time_not_import_time() -> None:
    """R006 use site: the wiring module holds no secret and re-reads the environment per call."""
    first_secret = generated_secret()
    second_secret = generated_secret()
    first = resolve_critical_action_signing_wiring(product_env(), environ={_SECRET_ENV: first_secret})
    second = resolve_critical_action_signing_wiring(product_env(), environ={_SECRET_ENV: second_secret})
    assert first.bootstrap_signature != second.bootstrap_signature
    assert not any(
        isinstance(value, str) and value in {first_secret, second_secret} for value in vars(wiring_module).values()
    )


def test_d6_r008_wiring_source_has_no_forbidden_literal_or_default_secret() -> None:
    tree = ast.parse(inspect.getsource(wiring_module))
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert not (literals & KNOWN_FORBIDDEN_LITERALS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            # mapping reads of the secret must not carry a default value
            assert len(node.args) == 1, "secret env read must not take a default"


def test_d6_r008_signature_depends_on_the_secret() -> None:
    first = resolve_critical_action_signing_wiring(product_env(), environ={_SECRET_ENV: generated_secret()})
    second = resolve_critical_action_signing_wiring(product_env(), environ={_SECRET_ENV: generated_secret()})
    assert first.bootstrap_signature is not None
    assert first.bootstrap_signature != second.bootstrap_signature
