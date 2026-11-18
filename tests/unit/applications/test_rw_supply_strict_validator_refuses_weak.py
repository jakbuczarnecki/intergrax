# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-B Part G: the strict-config validator refuses weak or disabled settings (NT-RW-SUP-09, F-D6-10).

NT-RW-SUP-08 (a production host factory composes the D6 controls, EX-D6-02 / F-D6-04) is red after
requalification and its fixer (G-09, the host factory) is outside the write set; it is held out of
the merge set. Secrets are generated at run time and never printed.
"""

from __future__ import annotations

import secrets
from pathlib import Path

import pytest
from intergrax.agent_distribution.trust import AgentPackageTrustPolicy
from intergrax.applications._shared.security_assembly_resolver import validate_strict_security_configuration
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.core.security.security_configuration import (
    PLUGIN_TRUST_POLICY_MISSING,
    SIGNING_SECRET_MISSING,
    SecurityConfigurationError,
)
from intergrax.runtime.events.store import resolve_runtime_event_persistence

pytestmark = [pytest.mark.unit]

_SECRET_ENV = "INTERGRAX_CRITICAL_ACTION_SIGNING_KEY"


def _strong_secret() -> str:
    return "tv-" + secrets.token_urlsafe(32)


def _evidence(tmp_path: Path) -> EvidencePersistencePort:
    port = resolve_runtime_event_persistence(db_path=tmp_path / "evidence.db")
    assert isinstance(port, EvidencePersistencePort)
    return port


def _items(issues: tuple[SecurityConfigurationError, ...]) -> set[str]:
    return {issue.item for issue in issues}


def test_sup_09a_short_signing_secret_refused(tmp_path: Path) -> None:
    for weak in (secrets.choice("abcdefgh"), secrets.token_hex(8)):
        issues = validate_strict_security_configuration(
            ApplicationEnvironmentProfile.product_defaults(),
            environ={_SECRET_ENV: weak},
            audit_evidence_persistence=_evidence(tmp_path),
            plugin_entry_points_enabled=False,
        )
        signing = [issue for issue in issues if issue.item == "critical_action_signing"]
        assert len(signing) == 1
        assert signing[0].reason_code == SIGNING_SECRET_MISSING
        if len(weak) > 1:
            echoed = weak in str(signing[0])
            assert not echoed, "the issue message echoes the configured value (value not shown)"

    strong = validate_strict_security_configuration(
        ApplicationEnvironmentProfile.product_defaults(),
        environ={_SECRET_ENV: _strong_secret()},
        audit_evidence_persistence=_evidence(tmp_path),
        plugin_entry_points_enabled=False,
    )
    assert strong == ()


def test_sup_09b_disabled_defenses_refused(tmp_path: Path) -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    env.identity_profile = env.identity_profile.model_copy(update={"critical_action_signing_enabled": False})
    env.security_profile = env.security_profile.model_copy(
        update={
            "tool_injection_defense_enabled": False,
            "retrieval_poisoning_defense_enabled": False,
            "immutable_audit_trail_enabled": False,
        }
    )
    issues = validate_strict_security_configuration(
        env,
        environ={},
        audit_evidence_persistence=_evidence(tmp_path),
        plugin_entry_points_enabled=False,
    )
    assert {
        "critical_action_signing",
        "tool_injection_defense",
        "retrieval_poisoning_defense",
        "security_audit_trail",
    } <= _items(issues)


def test_sup_09c_lab_profile_in_strict_mode_validated(tmp_path: Path) -> None:
    lab = ApplicationEnvironmentProfile.lab_defaults()
    assert validate_strict_security_configuration(lab, environ={}, plugin_entry_points_enabled=False) == ()
    lab.execution_mode = ExecutionMode.STRICT
    issues = validate_strict_security_configuration(
        lab,
        environ={},
        audit_evidence_persistence=_evidence(tmp_path),
        plugin_entry_points_enabled=False,
    )
    # The lab defaults disable signing and the immutable audit trail: refused like a product profile.
    assert {"critical_action_signing", "security_audit_trail"} <= _items(issues)


def test_sup_09d_plugin_discovery_without_policy_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_DISCOVER_PLUGINS", "1")
    monkeypatch.setenv(_SECRET_ENV, _strong_secret())
    # The caller supplies neither the discovery flag nor ``environ``: the process environment is read.
    issues = validate_strict_security_configuration(
        ApplicationEnvironmentProfile.product_defaults(),
        audit_evidence_persistence=_evidence(tmp_path),
    )
    assert [issue.reason_code for issue in issues] == [PLUGIN_TRUST_POLICY_MISSING]
    accepted = validate_strict_security_configuration(
        ApplicationEnvironmentProfile.product_defaults(),
        audit_evidence_persistence=_evidence(tmp_path),
        plugin_trust_policy=AgentPackageTrustPolicy(),
    )
    assert accepted == ()


def test_sup_09e_discovery_flag_read_from_the_passed_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review L-1: an explicit target ``environ`` that enables discovery is validated as such."""
    monkeypatch.delenv("INTERGRAX_DISCOVER_PLUGINS", raising=False)
    issues = validate_strict_security_configuration(
        ApplicationEnvironmentProfile.product_defaults(),
        environ={_SECRET_ENV: _strong_secret(), "INTERGRAX_DISCOVER_PLUGINS": "1"},
        audit_evidence_persistence=_evidence(tmp_path),
    )
    assert [issue.reason_code for issue in issues] == [PLUGIN_TRUST_POLICY_MISSING]
