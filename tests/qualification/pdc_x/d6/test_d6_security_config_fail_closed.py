# © Artur Czarnecki. All rights reserved.

"""PDC-D6-R024: missing or malformed security configuration fails closed with typed errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from intergrax.agent_distribution.trust import AgentPackageTrustPolicy
from intergrax.applications._shared.security_assembly_resolver import (
    assert_strict_security_configuration,
    validate_strict_security_configuration,
)
from intergrax.core.security.security_configuration import (
    AUDIT_REGIONS_INVALID,
    AUDIT_SINK_MISSING,
    PLUGIN_TRUST_POLICY_MISSING,
    RETRIEVAL_POISONING_PROFILE_MALFORMED,
    SIGNING_SECRET_ENV_INVALID,
    SIGNING_SECRET_MISSING,
    SecurityConfigurationError,
)

from ._support import generated_secret, lab_env, open_evidence_plane, product_env

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SECRET_ENV = "INTERGRAX_CRITICAL_ACTION_SIGNING_KEY"


def _complete_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "environ": {_SECRET_ENV: generated_secret()},
        "audit_evidence_persistence": open_evidence_plane(tmp_path / "evidence.db"),
        "plugin_entry_points_enabled": True,
        "plugin_trust_policy": AgentPackageTrustPolicy(),
    }


def _codes(issues: tuple[SecurityConfigurationError, ...]) -> set[str]:
    return {issue.reason_code for issue in issues}


def test_d6_r024_complete_configuration_is_accepted(tmp_path: Path) -> None:
    kwargs = _complete_kwargs(tmp_path)
    assert validate_strict_security_configuration(product_env(), **kwargs) == ()
    assert_strict_security_configuration(product_env(), **kwargs)


def test_d6_r024_missing_signing_secret_is_rejected_not_defaulted(tmp_path: Path) -> None:
    kwargs = _complete_kwargs(tmp_path)
    kwargs["environ"] = {}
    with pytest.raises(SecurityConfigurationError) as excinfo:
        assert_strict_security_configuration(product_env(), **kwargs)
    assert excinfo.value.reason_code == SIGNING_SECRET_MISSING


def test_d6_r024_malformed_signing_secret_env_name_is_rejected(tmp_path: Path) -> None:
    env = product_env()
    env.identity_profile = env.identity_profile.model_copy(update={"critical_action_signing_secret_env": ""})
    issues = validate_strict_security_configuration(env, **_complete_kwargs(tmp_path))
    assert _codes(issues) == {SIGNING_SECRET_ENV_INVALID}


def test_d6_r024_malformed_poisoning_profile_is_rejected(tmp_path: Path) -> None:
    env = product_env()
    env.security_profile = env.security_profile.model_construct(
        **{
            **env.security_profile.model_dump(),
            "retrieval_poisoning_defense_enabled": "off",
        }
    )
    issues = validate_strict_security_configuration(env, **_complete_kwargs(tmp_path))
    assert _codes(issues) == {RETRIEVAL_POISONING_PROFILE_MALFORMED}


def test_d6_r024_missing_audit_sink_is_rejected(tmp_path: Path) -> None:
    kwargs = _complete_kwargs(tmp_path)
    kwargs["audit_evidence_persistence"] = None
    issues = validate_strict_security_configuration(product_env(), **kwargs)
    assert _codes(issues) == {AUDIT_SINK_MISSING}


def test_d6_r024_malformed_audit_regions_are_rejected(tmp_path: Path) -> None:
    issues = validate_strict_security_configuration(
        product_env(audit_trail_regions=["eu-central-1"]), **_complete_kwargs(tmp_path)
    )
    assert _codes(issues) == {AUDIT_REGIONS_INVALID}


def test_d6_r024_missing_plugin_trust_policy_is_rejected(tmp_path: Path) -> None:
    kwargs = _complete_kwargs(tmp_path)
    kwargs["plugin_trust_policy"] = None
    issues = validate_strict_security_configuration(product_env(), **kwargs)
    assert _codes(issues) == {PLUGIN_TRUST_POLICY_MISSING}


def test_d6_r024_every_missing_item_is_reported_and_none_is_defaulted() -> None:
    issues = validate_strict_security_configuration(product_env(), environ={}, plugin_entry_points_enabled=True)
    assert _codes(issues) == {
        SIGNING_SECRET_MISSING,
        AUDIT_SINK_MISSING,
        PLUGIN_TRUST_POLICY_MISSING,
    }
    assert {issue.item for issue in issues} == {
        "critical_action_signing",
        "security_audit_trail",
        "plugin_provenance",
    }
    for issue in issues:
        assert isinstance(issue, SecurityConfigurationError)
        assert isinstance(issue, ValueError)


def test_d6_r024_lab_profile_has_no_mandatory_items() -> None:
    assert validate_strict_security_configuration(lab_env(), environ={}) == ()


def test_d6_r024_errors_are_typed_and_secret_free() -> None:
    error = SecurityConfigurationError(item="x", reason_code="code", message="why")
    assert error.item == "x" and error.reason_code == "code"
    assert "code" in str(error)
