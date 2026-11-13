# © Artur Czarnecki. All rights reserved.

"""PDC-D6-R009 (xd_06): a credential revoked in the middle of a retry loop is denied."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import ClassVar

import pytest
from intergrax.integrations.contracts.credential import (
    CredentialNotFoundError,
    CredentialRef,
    CredentialScopeAdmissionDecision,
    CredentialScopeAdmissionDeniedError,
    CredentialUseGrant,
    CredentialUseGrantExpiredError,
    CredentialUseScope,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.credentials.secrets_store_resolver import (
    SecretsStoreCredentialResolver,
)
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist
from intergrax.utils.time_provider import TimeProvider

from ._support import CountingSecretsStore, generated_secret

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SECRET_PATH = "secrets/tenant-a/vendor-api"
_HOSTS = ("https://api.vendor.com:443",)


class _Clock(TimeProvider):
    now: ClassVar[datetime] = datetime(2026, 11, 12, 12, 0, 0, tzinfo=UTC)

    @classmethod
    def utc_now(cls) -> datetime:
        return cls.now


class _RevocableAdmission:
    """Admission port backed by a revocation list that can change between attempts."""

    def __init__(self) -> None:
        self.revoked_grants: set[str] = set()
        self.calls = 0

    def admit(self, grant: CredentialUseGrant, scope: CredentialUseScope):
        self.calls += 1
        if grant.grant_id in self.revoked_grants:
            return CredentialScopeAdmissionDecision.DENY
        return CredentialScopeAdmissionDecision.ALLOW


def _grant(lifetime: timedelta = timedelta(minutes=5)) -> CredentialUseGrant:
    return CredentialUseGrant(
        grant_id="grant-1",
        credential_ref=CredentialRef.from_secret_path(
            provider_id="vendor_api", secret_path=_SECRET_PATH, tenant_id="tenant-a"
        ),
        tenant_id="tenant-a",
        provider_id="vendor_api",
        integration_id="vendor_api:issue_tracker",
        operation="vendor.orders.read",
        execution_id="exec-a",
        target_scope=canonicalize_network_egress_allowlist(_HOSTS),
        expires_at=_Clock.now + lifetime,
    )


def _scope() -> CredentialUseScope:
    return CredentialUseScope(
        tenant_id="tenant-a",
        provider_id="vendor_api",
        integration_id="vendor_api:issue_tracker",
        operation="vendor.orders.read",
        execution_id="exec-a",
        target_scope=canonicalize_network_egress_allowlist(_HOSTS),
    )


def _broker(store: CountingSecretsStore, admission: _RevocableAdmission) -> ScopedCredentialBroker:
    return ScopedCredentialBroker(
        resolver=SecretsStoreCredentialResolver(store),
        admission=admission,
        time_provider=_Clock,
    )


def test_xd06_grant_revoked_between_retry_attempts_is_denied() -> None:
    store = CountingSecretsStore({_SECRET_PATH: generated_secret()})
    admission = _RevocableAdmission()
    broker = _broker(store, admission)
    grant, scope = _grant(), _scope()

    first = broker.resolve_scoped(grant, scope)  # attempt 1 succeeds, upstream call fails
    assert first.resolved_credential.value
    admission.revoked_grants.add(grant.grant_id)  # revoked while the caller is backing off

    with pytest.raises(CredentialScopeAdmissionDeniedError):
        broker.resolve_scoped(grant, scope)  # retry attempt 2
    assert len(store.reads) == 1, "the revoked retry must not read the secret store again"


def test_xd06_retry_loop_stops_at_the_revocation_and_never_reuses_a_cached_value() -> None:
    store = CountingSecretsStore({_SECRET_PATH: generated_secret()})
    admission = _RevocableAdmission()
    broker = _broker(store, admission)
    grant, scope = _grant(), _scope()

    outcomes: list[str] = []
    for attempt in range(4):
        if attempt == 2:
            admission.revoked_grants.add(grant.grant_id)
        try:
            broker.resolve_scoped(grant, scope)
            outcomes.append("resolved")
        except CredentialScopeAdmissionDeniedError:
            outcomes.append("denied")
    assert outcomes == ["resolved", "resolved", "denied", "denied"]
    assert len(store.reads) == 2
    assert admission.calls == 4, "every retry is re-admitted; no admission result is cached"


def test_xd06_secret_removed_from_the_store_between_attempts_is_denied() -> None:
    store = CountingSecretsStore({_SECRET_PATH: generated_secret()})
    broker = _broker(store, _RevocableAdmission())
    grant, scope = _grant(), _scope()
    broker.resolve_scoped(grant, scope)
    del store.values[_SECRET_PATH]  # secret revoked at the provider
    with pytest.raises(CredentialNotFoundError):
        broker.resolve_scoped(grant, scope)


def test_xd06_grant_expiring_between_attempts_is_denied() -> None:
    store = CountingSecretsStore({_SECRET_PATH: generated_secret()})
    broker = _broker(store, _RevocableAdmission())
    grant, scope = _grant(lifetime=timedelta(seconds=30)), _scope()
    broker.resolve_scoped(grant, scope)
    original = _Clock.now
    try:
        _Clock.now = original + timedelta(seconds=31)
        with pytest.raises(CredentialUseGrantExpiredError):
            broker.resolve_scoped(grant, scope)
    finally:
        _Clock.now = original
    assert len(store.reads) == 1


def test_xd06_unavailable_admission_mid_retry_fails_closed() -> None:
    store = CountingSecretsStore({_SECRET_PATH: generated_secret()})

    class _Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def admit(self, grant: CredentialUseGrant, scope: CredentialUseScope):
            self.calls += 1
            if self.calls == 1:
                return CredentialScopeAdmissionDecision.ALLOW
            return CredentialScopeAdmissionDecision.UNAVAILABLE

    broker = ScopedCredentialBroker(
        resolver=SecretsStoreCredentialResolver(store), admission=_Flaky(), time_provider=_Clock
    )
    grant, scope = _grant(), _scope()
    broker.resolve_scoped(grant, scope)
    with pytest.raises(CredentialScopeAdmissionDeniedError):
        broker.resolve_scoped(grant, scope)
    assert len(store.reads) == 1
