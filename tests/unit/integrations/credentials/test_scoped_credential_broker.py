# © Artur Czarnecki. All rights reserved.

"""AW-7C P0-2 — purpose-scoped credential brokering."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import ClassVar

import pytest

from intergrax.integrations.contracts.credential import (
    CredentialNotFoundError,
    CredentialProviderUnavailableError,
    CredentialRef,
    CredentialResolutionContext,
    CredentialScopeAdmissionDecision,
    CredentialScopeAdmissionDeniedError,
    CredentialScopeMismatchError,
    CredentialUseGrant,
    CredentialUseGrantExpiredError,
    CredentialUseScope,
    ScopedCredentialResolutionResult,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.credentials.scope_validation import (
    requested_target_scope_within_grant,
    validate_credential_use_grant,
    validate_scoped_use_scope,
)
from intergrax.integrations.credentials.secrets_store_resolver import (
    SecretsStoreCredentialResolver,
)
from intergrax.runtime.sandbox.network_egress import (
    NetworkEgressAllowlist,
    canonicalize_network_egress_allowlist,
    parse_network_egress_host,
)
from intergrax.utils.time_provider import TimeProvider

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SENTINEL = "SUPER_SECRET_P0_2_SENTINEL"
_SECRET_PATH = "secrets/tenant-a/vendor-api"
_TENANT = "tenant-a"
_PROVIDER = "vendor_api"
_INTEGRATION = "vendor_api:issue_tracker"
_OPERATION_READ = "vendor.orders.read"
_OPERATION_WRITE = "vendor.orders.delete"
_EXECUTION_A = "exec-a"
_EXECUTION_B = "exec-b"
_GRANT_ID = "grant-001"


class _FixedTimeProvider(TimeProvider):
    fixed_now: ClassVar[datetime] = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)

    @classmethod
    def utc_now(cls) -> datetime:
        return cls.fixed_now


class _CountingSecretsStore:
    def __init__(self, values: Mapping[str, str] | None = None) -> None:
        self.values = dict(values or {})
        self.get_secret_calls: list[tuple[str, str | None]] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        self.get_secret_calls.append((path.strip(), version))
        return self.values[path.strip()]


class _RecordingAdmission:
    def __init__(self, decision: CredentialScopeAdmissionDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[CredentialUseGrant, CredentialUseScope]] = []

    def admit(
        self,
        grant: CredentialUseGrant,
        scope: CredentialUseScope,
    ) -> CredentialScopeAdmissionDecision:
        self.calls.append((grant, scope))
        return self.decision


def _host_allowlist(*hosts: str) -> NetworkEgressAllowlist:
    return canonicalize_network_egress_allowlist(hosts)


def _credential_ref(*, tenant_id: str = _TENANT) -> CredentialRef:
    return CredentialRef.from_secret_path(
        provider_id=_PROVIDER,
        secret_path=_SECRET_PATH,
        tenant_id=tenant_id,
    )


def _grant(
    *,
    tenant_id: str = _TENANT,
    execution_id: str = _EXECUTION_A,
    operation: str = _OPERATION_READ,
    integration_id: str = _INTEGRATION,
    target_hosts: tuple[str, ...] = ("https://api.vendor.com:443",),
    expires_at: datetime | None = None,
    credential_ref: CredentialRef | None = None,
) -> CredentialUseGrant:
    return CredentialUseGrant(
        grant_id=_GRANT_ID,
        credential_ref=credential_ref or _credential_ref(tenant_id=tenant_id),
        tenant_id=tenant_id,
        provider_id=_PROVIDER,
        integration_id=integration_id,
        operation=operation,
        execution_id=execution_id,
        target_scope=_host_allowlist(*target_hosts),
        expires_at=expires_at or (_FixedTimeProvider.utc_now() + timedelta(minutes=5)),
    )


def _scope(
    *,
    tenant_id: str = _TENANT,
    execution_id: str = _EXECUTION_A,
    operation: str = _OPERATION_READ,
    integration_id: str = _INTEGRATION,
    target_hosts: tuple[str, ...] = ("https://api.vendor.com:443",),
) -> CredentialUseScope:
    return CredentialUseScope(
        tenant_id=tenant_id,
        provider_id=_PROVIDER,
        integration_id=integration_id,
        operation=operation,
        execution_id=execution_id,
        target_scope=_host_allowlist(*target_hosts),
    )


def _broker(
    store: _CountingSecretsStore,
    admission: _RecordingAdmission,
) -> ScopedCredentialBroker:
    return ScopedCredentialBroker(
        resolver=SecretsStoreCredentialResolver(store),
        admission=admission,
        time_provider=_FixedTimeProvider,
    )


def _assert_sentinel_absent(payload: object) -> None:
    text = json.dumps(payload, default=str) if not isinstance(payload, str) else payload
    assert _SENTINEL not in text


def test_scoped_resolution_happy_path_checks_all_dimensions_before_lookup() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    result = broker.resolve_scoped(_grant(), _scope())
    assert isinstance(result, ScopedCredentialResolutionResult)
    assert result.resolved_credential.value == _SENTINEL
    assert store.get_secret_calls == [(_SECRET_PATH, None)]
    assert len(admission.calls) == 1
    assert result.use_evidence.grant_id == _GRANT_ID
    assert result.use_evidence.execution_id == _EXECUTION_A
    assert result.use_evidence.operation == _OPERATION_READ
    assert result.use_evidence.integration_id == _INTEGRATION


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (lambda: _scope(tenant_id=""), "tenant"),
        (lambda: _scope(operation=""), "operation"),
        (lambda: _scope(execution_id=""), "execution"),
        (lambda: _scope(integration_id=""), "integration"),
        (lambda: _scope(target_hosts=()), "target scope"),
    ),
)
def test_scoped_scope_contract_rejects_blank_required_fields(
    factory: object,
    message: str,
) -> None:
    with pytest.raises(CredentialScopeMismatchError, match=message):
        validate_scoped_use_scope(factory())  # type: ignore[operator]


def test_scoped_grant_contract_rejects_blank_required_fields() -> None:
    grant = _grant()
    broken = CredentialUseGrant(
        grant_id="",
        credential_ref=grant.credential_ref,
        tenant_id=grant.tenant_id,
        provider_id=grant.provider_id,
        integration_id=grant.integration_id,
        operation=grant.operation,
        execution_id=grant.execution_id,
        target_scope=grant.target_scope,
        expires_at=grant.expires_at,
    )
    with pytest.raises(CredentialScopeMismatchError, match="grant identity"):
        validate_credential_use_grant(broken)


def test_scoped_flow_requires_execution_id() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    with pytest.raises(CredentialScopeMismatchError, match="execution"):
        broker.resolve_scoped(_grant(), _scope(execution_id=""))
    assert store.get_secret_calls == []


def test_matching_tenant_passes() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    broker.resolve_scoped(_grant(tenant_id=_TENANT), _scope(tenant_id=_TENANT))
    assert store.get_secret_calls == [(_SECRET_PATH, None)]


@pytest.mark.parametrize(
    ("grant_tenant", "scope_tenant", "ref_tenant"),
    (
        ("tenant-a", "tenant-b", "tenant-a"),
        ("tenant-a", "tenant-a", "tenant-b"),
        ("tenant-a", "tenant-b", "tenant-b"),
    ),
)
def test_tenant_mismatch_denies(
    grant_tenant: str,
    scope_tenant: str,
    ref_tenant: str,
) -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    ref = _credential_ref(tenant_id=ref_tenant)
    with pytest.raises(CredentialScopeMismatchError, match="tenant"):
        broker.resolve_scoped(
            _grant(tenant_id=grant_tenant, credential_ref=ref),
            _scope(tenant_id=scope_tenant),
        )
    assert store.get_secret_calls == []


def test_execution_binding_mismatch_denies() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    with pytest.raises(CredentialScopeMismatchError, match="execution"):
        broker.resolve_scoped(_grant(execution_id=_EXECUTION_A), _scope(execution_id=_EXECUTION_B))
    assert store.get_secret_calls == []


def test_operation_binding_mismatch_denies() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    with pytest.raises(CredentialScopeMismatchError, match="operation"):
        broker.resolve_scoped(
            _grant(operation=_OPERATION_READ),
            _scope(operation=_OPERATION_WRITE),
        )
    assert store.get_secret_calls == []


def test_integration_binding_mismatch_denies() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    with pytest.raises(CredentialScopeMismatchError, match="integration"):
        broker.resolve_scoped(
            _grant(integration_id="vendor_api:issue_tracker"),
            _scope(integration_id="vendor_api:crm"),
        )
    assert store.get_secret_calls == []


def test_target_host_grant_match_passes() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    broker.resolve_scoped(
        _grant(target_hosts=("https://api.vendor.com:443",)),
        _scope(target_hosts=("https://api.vendor.com:443",)),
    )
    assert store.get_secret_calls == [(_SECRET_PATH, None)]


def test_target_host_mismatch_denies() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    with pytest.raises(CredentialScopeMismatchError, match="target scope"):
        broker.resolve_scoped(
            _grant(target_hosts=("https://api.vendor.com:443",)),
            _scope(target_hosts=("https://evil.example:443",)),
        )
    assert store.get_secret_calls == []


def test_requested_scope_broader_than_grant_denies() -> None:
    grant_scope = _host_allowlist("https://api.vendor.com:443")
    requested_scope = _host_allowlist(
        "https://api.vendor.com:443",
        "https://evil.example:443",
    )
    assert requested_target_scope_within_grant(requested_scope, grant_scope) is False
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    with pytest.raises(CredentialScopeMismatchError, match="target scope"):
        broker.resolve_scoped(
            _grant(target_hosts=("https://api.vendor.com:443",)),
            CredentialUseScope(
                tenant_id=_TENANT,
                provider_id=_PROVIDER,
                integration_id=_INTEGRATION,
                operation=_OPERATION_READ,
                execution_id=_EXECUTION_A,
                target_scope=requested_scope,
            ),
        )
    assert store.get_secret_calls == []


def test_requested_subset_of_grant_passes() -> None:
    grant_scope = _host_allowlist(
        "https://api.vendor.com:443",
        "https://backup.vendor.com:443",
    )
    requested_scope = _host_allowlist("https://api.vendor.com:443")
    assert requested_target_scope_within_grant(requested_scope, grant_scope) is True


def test_future_grant_passes_at_boundary() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    expires_at = _FixedTimeProvider.utc_now() + timedelta(seconds=1)
    broker.resolve_scoped(_grant(expires_at=expires_at), _scope())
    assert store.get_secret_calls == [(_SECRET_PATH, None)]


def test_expired_grant_denies_without_lookup() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    expired_at = _FixedTimeProvider.utc_now() - timedelta(seconds=1)
    with pytest.raises(CredentialUseGrantExpiredError):
        broker.resolve_scoped(_grant(expires_at=expired_at), _scope())
    assert store.get_secret_calls == []


def test_policy_allow_invokes_resolver_once() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    broker.resolve_scoped(_grant(), _scope())
    assert len(store.get_secret_calls) == 1


def test_policy_deny_skips_resolver() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.DENY)
    broker = _broker(store, admission)
    with pytest.raises(CredentialScopeAdmissionDeniedError, match="denied"):
        broker.resolve_scoped(_grant(), _scope())
    assert store.get_secret_calls == []


def test_policy_unavailable_skips_resolver() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.UNAVAILABLE)
    broker = _broker(store, admission)
    with pytest.raises(CredentialScopeAdmissionDeniedError, match="unavailable"):
        broker.resolve_scoped(_grant(), _scope())
    assert store.get_secret_calls == []


def test_secret_safety_repr_and_evidence() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    result = broker.resolve_scoped(_grant(), _scope())
    _assert_sentinel_absent(repr(result))
    _assert_sentinel_absent(repr(result.use_evidence))
    _assert_sentinel_absent(repr(result.resolved_credential))


def test_secret_safety_exception_messages() -> None:
    store = _CountingSecretsStore()
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = _broker(store, admission)
    with pytest.raises(CredentialNotFoundError) as exc_info:
        broker.resolve_scoped(_grant(), _scope())
    _assert_sentinel_absent(str(exc_info.value))


def test_legacy_generic_resolver_still_works_with_tenant_only_context() -> None:
    store = _CountingSecretsStore({_SECRET_PATH: _SENTINEL})
    resolver = SecretsStoreCredentialResolver(store)
    ref = _credential_ref()
    resolved = resolver.resolve(ref, context=CredentialResolutionContext(tenant_id=_TENANT))
    assert resolved.value == _SENTINEL


def test_provider_unavailable_preserved_without_env_fallback() -> None:
    class _BrokenStore:
        def get_secret(self, path: str, *, version: str | None = None) -> str:
            raise RuntimeError("backend down")

    store = _BrokenStore()
    admission = _RecordingAdmission(CredentialScopeAdmissionDecision.ALLOW)
    broker = ScopedCredentialBroker(
        resolver=SecretsStoreCredentialResolver(store),  # type: ignore[arg-type]
        admission=admission,
        time_provider=_FixedTimeProvider,
    )
    with pytest.raises(CredentialProviderUnavailableError):
        broker.resolve_scoped(_grant(), _scope())


def test_naive_expiry_rejected_at_validation() -> None:
    naive = datetime(2026, 9, 7, 12, 0, 0)
    grant = CredentialUseGrant(
        grant_id=_GRANT_ID,
        credential_ref=_credential_ref(),
        tenant_id=_TENANT,
        provider_id=_PROVIDER,
        integration_id=_INTEGRATION,
        operation=_OPERATION_READ,
        execution_id=_EXECUTION_A,
        target_scope=_host_allowlist("https://api.vendor.com:443"),
        expires_at=naive,
    )
    with pytest.raises(CredentialScopeMismatchError, match="timezone-aware"):
        validate_credential_use_grant(grant)


def test_parse_network_egress_host_used_for_target_scope() -> None:
    host = parse_network_egress_host("api.vendor.com:443")
    assert host.canonical_form() == "https://api.vendor.com"
    assert host.port == 443
