# © Artur Czarnecki. All rights reserved.

"""PDC-D6-R006 / R007: late secret resolution, static raw-secret gate, log and trace leak scan."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import pytest
from intergrax.applications._shared.critical_action_signing_wiring import (
    resolve_critical_action_signing_wiring,
)
from intergrax.applications._shared.security_audit_trail_wiring import (
    resolve_security_audit_trail_wiring,
)
from intergrax.core.security.raw_secret_gate import (
    RULE_CREDENTIAL_FORMAT,
    RULE_ENV_FALLBACK,
    RULE_KNOWN_FORBIDDEN,
    RULE_SECRET_NAMED_LITERAL,
    scan_source_for_raw_secrets,
    scan_tree_for_raw_secrets,
)
from intergrax.integrations.contracts.credential import (
    CredentialRef,
    CredentialScopeAdmissionDecision,
    CredentialUseGrant,
    CredentialUseScope,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.credentials.secrets_store_resolver import (
    SecretsStoreCredentialResolver,
)
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist
from intergrax.utils.time_provider import TimeProvider

from ._support import (
    REPO_ROOT,
    CountingSecretsStore,
    generated_secret,
    open_evidence_plane,
    product_env,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PRODUCT_ROOTS = ("intergrax", "agents", "applications")
_FORBIDDEN_LITERAL = "-".join(("harness", "dev", "signing", "key"))


# ---------------------------------------------------------------------------
# Static raw-secret gate (R007)
# ---------------------------------------------------------------------------


def test_d6_r007_product_source_has_no_raw_secret_findings() -> None:
    findings = scan_tree_for_raw_secrets([REPO_ROOT / root for root in _PRODUCT_ROOTS], relative_to=REPO_ROOT)
    assert findings == (), [finding.key() for finding in findings]


def test_d6_r007_gate_catches_the_original_signing_key_fallback() -> None:
    legacy = (
        "import os\n"
        "def resolve(identity):\n"
        f'    secret = os.environ.get(identity.critical_action_signing_secret_env, "{_FORBIDDEN_LITERAL}")\n'
        "    return secret\n"
    )
    rules = {finding.rule for finding in scan_source_for_raw_secrets(legacy, path="legacy.py")}
    assert RULE_ENV_FALLBACK in rules
    assert RULE_KNOWN_FORBIDDEN in rules


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ('import os\nx = os.getenv("VENDOR_API_KEY", "fallbackvalue")\n', RULE_ENV_FALLBACK),
        ('import os\nx = os.environ.get("DB_PASSWORD", "hunter2hunter2")\n', RULE_ENV_FALLBACK),
        ('db_password = "Zx9fQ2mLp8Rt5Vw3Yb7Nc1Kd"\n', RULE_SECRET_NAMED_LITERAL),
        ('client(api_key="Zx9fQ2mLp8Rt5Vw3Yb7Nc1Kd")\n', RULE_SECRET_NAMED_LITERAL),
        ('def f(auth_token="Zx9fQ2mLp8Rt5Vw3Yb7Nc1Kd"):\n    return auth_token\n', RULE_SECRET_NAMED_LITERAL),
        ('KEY = "-----BEGIN RSA PRIVATE KEY-----\\nabc"\n', RULE_CREDENTIAL_FORMAT),
        ('t = "ghp_' + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8" + '"\n', RULE_CREDENTIAL_FORMAT),
        ('t = "AKIA' + "ABCDEFGHIJKLMNOP" + '"\n', RULE_CREDENTIAL_FORMAT),
    ],
)
def test_d6_r007_gate_rejects_synthetic_raw_secrets(source: str, rule: str) -> None:
    findings = scan_source_for_raw_secrets(source, path="synthetic.py")
    assert rule in {finding.rule for finding in findings}


@pytest.mark.parametrize(
    "source",
    [
        'import os\nx = os.environ.get("VENDOR_API_KEY")\n',
        'import os\nx = os.environ.get("REGION", "eu-central-1")\n',
        'secret_env: str = "INTERGRAX_CRITICAL_ACTION_SIGNING_KEY"\n',
        'REASON_TOKEN_COUNT_FAILED = "token_count_failed"\n',
        'CREDENTIAL_UNAVAILABLE = "credential_unavailable"\n',
        'MAX_TOKENS = os_value("N", "4096")\n',
    ],
)
def test_d6_r007_gate_accepts_benign_source(source: str) -> None:
    assert scan_source_for_raw_secrets(source, path="benign.py") == ()


def test_d6_r007_findings_never_carry_the_literal() -> None:
    secret = generated_secret() + "Zx9"
    findings = scan_source_for_raw_secrets(f'db_password = "{secret}Aa1Bb2Cc3"\n', path="x.py")
    assert findings
    for finding in findings:
        assert secret not in repr(finding)


# ---------------------------------------------------------------------------
# Use-site resolution (R006)
# ---------------------------------------------------------------------------

_SECRET_PATH = "secrets/tenant-a/vendor-api"
_TENANT = "tenant-a"
_PROVIDER = "vendor_api"
_INTEGRATION = "vendor_api:issue_tracker"
_OPERATION = "vendor.orders.read"
_EXECUTION = "exec-a"


class _FixedTime(TimeProvider):
    fixed: ClassVar[datetime] = datetime(2026, 11, 12, 12, 0, 0, tzinfo=UTC)

    @classmethod
    def utc_now(cls) -> datetime:
        return cls.fixed


class _Admission:
    def __init__(self, decision: CredentialScopeAdmissionDecision) -> None:
        self.decision = decision

    def admit(self, grant: CredentialUseGrant, scope: CredentialUseScope):
        return self.decision


def _scope(hosts: tuple[str, ...] = ("https://api.vendor.com:443",), **overrides: str) -> CredentialUseScope:
    base = {
        "tenant_id": _TENANT,
        "provider_id": _PROVIDER,
        "integration_id": _INTEGRATION,
        "operation": _OPERATION,
        "execution_id": _EXECUTION,
    }
    base.update(overrides)
    return CredentialUseScope(target_scope=canonicalize_network_egress_allowlist(hosts), **base)


def _grant(hosts: tuple[str, ...] = ("https://api.vendor.com:443",)) -> CredentialUseGrant:
    return CredentialUseGrant(
        grant_id="grant-1",
        credential_ref=CredentialRef.from_secret_path(
            provider_id=_PROVIDER, secret_path=_SECRET_PATH, tenant_id=_TENANT
        ),
        tenant_id=_TENANT,
        provider_id=_PROVIDER,
        integration_id=_INTEGRATION,
        operation=_OPERATION,
        execution_id=_EXECUTION,
        target_scope=canonicalize_network_egress_allowlist(hosts),
        expires_at=_FixedTime.fixed + timedelta(minutes=5),
    )


def _broker(store: CountingSecretsStore, decision=CredentialScopeAdmissionDecision.ALLOW):
    return ScopedCredentialBroker(
        resolver=SecretsStoreCredentialResolver(store),
        admission=_Admission(decision),
        time_provider=_FixedTime,
    )


def test_d6_r006_secret_is_resolved_only_at_the_use_site() -> None:
    value = generated_secret()
    store = CountingSecretsStore({_SECRET_PATH: value})
    broker = _broker(store)
    grant, scope = _grant(), _scope()
    # building grants, scopes and the broker never touches the secret store
    assert store.reads == []
    result = broker.resolve_scoped(grant, scope)
    assert store.reads == [_SECRET_PATH]
    assert result.resolved_credential.value == value
    # the safe evidence, the grant and the scope never carry the value
    for safe in (result.use_evidence, grant, scope, grant.credential_ref):
        assert value not in repr(safe)
    assert value not in repr(result) and value not in str(result.resolved_credential)


@pytest.mark.parametrize(
    "decision",
    [CredentialScopeAdmissionDecision.DENY, CredentialScopeAdmissionDecision.UNAVAILABLE],
)
def test_d6_r006_denied_admission_never_reads_the_secret(
    decision: CredentialScopeAdmissionDecision,
) -> None:
    store = CountingSecretsStore({_SECRET_PATH: generated_secret()})
    with pytest.raises(Exception):
        _broker(store, decision).resolve_scoped(_grant(), _scope())
    assert store.reads == []


def test_d6_r006_scope_mismatch_never_reads_the_secret() -> None:
    store = CountingSecretsStore({_SECRET_PATH: generated_secret()})
    with pytest.raises(Exception):
        _broker(store).resolve_scoped(_grant(), _scope(tenant_id="tenant-b"))
    assert store.reads == []


# ---------------------------------------------------------------------------
# caplog / trace leak scan (R007)
# ---------------------------------------------------------------------------


def test_d6_r007_secret_never_reaches_logs_reprs_or_evidence(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    signing_secret = generated_secret()
    credential_value = generated_secret()
    db_path = tmp_path / "evidence.db"
    store = CountingSecretsStore({_SECRET_PATH: credential_value})

    with caplog.at_level(logging.DEBUG):
        signing = resolve_critical_action_signing_wiring(
            product_env(), environ={"INTERGRAX_CRITICAL_ACTION_SIGNING_KEY": signing_secret}
        )
        audit = resolve_security_audit_trail_wiring(product_env(), evidence_persistence=open_evidence_plane(db_path))
        assert audit.trail is not None
        audit.trail.append(
            tenant_id="tenant-a",
            action="credential.used",
            actor_id="svc",
            resource=_SECRET_PATH,
            metadata={"grant_id": "grant-1"},
        )
        result = _broker(store).resolve_scoped(_grant(), _scope())

    transcript = "\n".join([caplog.text, repr(signing), repr(audit), repr(result), repr(result.use_evidence)])
    for secret in (signing_secret, credential_value):
        assert secret not in transcript
    # evidence plane (the trace/audit store) holds no secret bytes either
    raw = db_path.read_bytes()
    for secret in (signing_secret, credential_value):
        assert secret.encode("utf-8") not in raw
    connection = sqlite3.connect(db_path)
    try:
        dump = "\n".join(connection.iterdump())
    finally:
        connection.close()
    for secret in (signing_secret, credential_value):
        assert secret not in dump


def test_d6_r007_failed_signing_resolution_does_not_log_the_environment(
    caplog: pytest.LogCaptureFixture,
) -> None:
    other_secret = generated_secret()
    with caplog.at_level(logging.DEBUG), pytest.raises(Exception):
        resolve_critical_action_signing_wiring(product_env(), environ={"OTHER_SECRET": other_secret})
    assert other_secret not in caplog.text
