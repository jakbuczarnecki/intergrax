# © Artur Czarnecki. All rights reserved.

"""Focused tests for tenant connection product orchestration (PRODUCT-5B)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Literal
from unittest.mock import patch

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.tenant_connection_auth import (
    TenantConnectionAuthBeginResult,
    TenantConnectionAuthExchangeResult,
    TenantConnectionAuthManualBindResult,
    TenantConnectionAuthMode,
    TenantConnectionAuthProviderDescriptor,
    TenantConnectionAuthProviderRegistry,
    TenantConnectionAuthQualification,
    generate_correlation_state,
    generate_pkce_pair,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionRehydrator,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnectionAdministrativeStatus,
)
from intergrax.runtime.vendor_knowledge.models import JsonValue
from local_workspace_application.workspaces.tenant_connection_authorization_transaction import (
    TenantConnectionAuthorizationCompletionState,
    TenantConnectionAuthorizationTransaction,
    TenantConnectionAuthorizationTransactionRepository,
    build_state_parameter,
    parse_state_parameter,
    verifier_secret_path,
)
from local_workspace_application.workspaces.tenant_connection_product_errors import (
    TenantConnectionProductError,
)
from local_workspace_application.workspaces.tenant_connection_product_orchestration import (
    TenantConnectionProductOrchestrationConfig,
    TenantConnectionProductOrchestrationService,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_REDIRECT = "https://app.example.com/oauth/callback"
_PROVIDER = "test.oauth"
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _SecretsStore:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.deleted: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        _ = version
        if path not in self.store:
            raise KeyError(path)
        return self.store[path]

    def put_secret(self, path: str, value: str) -> None:
        self.store[path] = value

    def delete_secret(self, path: str) -> None:
        self.deleted.append(path)
        self.store.pop(path, None)


class _FakeIntegration:
    provider_id = _PROVIDER
    integration_kind = IntegrationCategory.COLLABORATION_SUITE.value


class _FakeFactory:
    def create_integration(self, **kwargs: object) -> _FakeIntegration:
        _ = kwargs
        return _FakeIntegration()


class _FakeOAuthProvider:
    provider_id = _PROVIDER
    integration_kind = IntegrationCategory.COLLABORATION_SUITE
    auth_mode = TenantConnectionAuthMode.OAUTH_DELEGATED
    qualification = TenantConnectionAuthQualification.QUALIFIED
    exchange_calls = 0

    def describe(self) -> TenantConnectionAuthProviderDescriptor:
        return TenantConnectionAuthProviderDescriptor(
            provider_id=self.provider_id,
            integration_kind=self.integration_kind,
            auth_mode=self.auth_mode,
            safe_display_name="Test OAuth",
            supported_scopes_summary="test",
        )

    def begin_authorization(
        self,
        *,
        tenant_id: str,
        redirect_uri: str,
        reconnect_connection_ref: str | None,
    ) -> TenantConnectionAuthBeginResult:
        _ = tenant_id, reconnect_connection_ref
        verifier, _ = generate_pkce_pair()
        correlation = generate_correlation_state()
        return TenantConnectionAuthBeginResult(
            authorization_url=f"https://oauth.example/authorize?redirect_uri={redirect_uri}",
            code_verifier=verifier,
            correlation_state=correlation,
            required_user_action="redirect",
        )

    def exchange_authorization_code(
        self,
        *,
        tenant_id: str,
        redirect_uri: str,
        authorization_code: str,
        code_verifier: str,
        correlation_state: str,
    ) -> TenantConnectionAuthExchangeResult:
        _ = tenant_id, redirect_uri, authorization_code, code_verifier, correlation_state
        self.exchange_calls += 1
        return TenantConnectionAuthExchangeResult(
            credential_bundle_json=json.dumps({"access_token": "token-value"}),
            connected_principal_ref="principal-1",
        )

    def bind_manual_credentials(
        self,
        *,
        tenant_id: str,
        credential_payload: Mapping[str, JsonValue],
    ) -> TenantConnectionAuthManualBindResult:
        _ = tenant_id, credential_payload
        raise ValueError("not manual")

    def build_secret_free_config(
        self,
        *,
        tenant_id: str,
        reconnect_connection: object | None,
    ) -> Mapping[str, JsonValue]:
        _ = tenant_id, reconnect_connection
        return {}

    def revoke_remote_credentials(
        self,
        *,
        tenant_id: str,
        credential_bundle_json: str,
    ) -> None:
        _ = tenant_id, credential_bundle_json


def _build_service(
    *,
    secrets: _SecretsStore | None = None,
    provider: _FakeOAuthProvider | None = None,
    store: InMemoryDocumentStore | None = None,
) -> tuple[TenantConnectionProductOrchestrationService, _SecretsStore, _FakeOAuthProvider]:
    document_store = store or InMemoryDocumentStore()
    secrets_store = secrets or _SecretsStore()
    connection_repository = DocumentStoreTenantConnectionRepository(document_store)
    transaction_repository = TenantConnectionAuthorizationTransactionRepository(document_store)
    registry = TenantConnectionAuthProviderRegistry()
    fake_provider = provider or _FakeOAuthProvider()
    registry.register(fake_provider)
    connection_registry = KnowledgeConnectionRegistry()
    rehydrator = TenantConnectionRehydrator(
        repository=connection_repository,
        secrets_store=secrets_store,
        integration_factory=_FakeFactory(),
        connection_registry=connection_registry,
    )
    service = TenantConnectionProductOrchestrationService(
        tenant_id=_TENANT,
        connection_repository=connection_repository,
        transaction_repository=transaction_repository,
        secrets_store=secrets_store,
        auth_provider_registry=registry,
        rehydrator=rehydrator,
        connection_registry=connection_registry,
        config=TenantConnectionProductOrchestrationConfig(
            redirect_allowlist=frozenset({_REDIRECT}),
            transaction_ttl_seconds=900,
            completion_claim_ttl_seconds=120,
        ),
    )
    return service, secrets_store, fake_provider


def test_provider_list_safe() -> None:
    service, _, _ = _build_service()
    providers = service.list_supported_connection_providers()
    assert providers
    assert "credential_ref" not in json.dumps(providers)


def test_begin_stores_verifier_not_in_transaction() -> None:
    service, secrets, _ = _build_service()
    begin = service.begin_connection_authorization(
        provider_id=_PROVIDER,
        redirect_uri=_REDIRECT,
    )
    transaction = TenantConnectionAuthorizationTransactionRepository(
        InMemoryDocumentStore(),
    )
    # repository uses same store - get via internal
    repo = service._transaction_repository
    stored = repo.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=begin.authorization_transaction_ref,
    )
    assert stored is not None
    assert stored.verifier_secret_ref is not None
    assert "pkce-verifier" in stored.verifier_secret_ref
    dumped = stored.model_dump()
    assert "code_verifier" not in dumped
    verifier_path = verifier_secret_path(
        tenant_id=_TENANT,
        authorization_transaction_ref=begin.authorization_transaction_ref,
    )
    assert secrets.store[verifier_path]


def test_oauth_complete_happy_path() -> None:
    service, secrets, provider = _build_service()
    begin = service.begin_connection_authorization(
        provider_id=_PROVIDER,
        redirect_uri=_REDIRECT,
    )
    stored = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=begin.authorization_transaction_ref,
    )
    assert stored is not None
    state = build_state_parameter(
        tenant_id=_TENANT,
        authorization_transaction_ref=begin.authorization_transaction_ref,
        correlation_state=stored.correlation_state,
    )
    result = service.complete_connection_authorization(
        authorization_code="auth-code-1",
        state=state,
    )
    assert result.disposition == "created"
    assert "credential_ref" not in result.connection.model_dump()
    assert provider.exchange_calls == 1
    connection = service.get_connection(result.connection.connection_ref)
    assert connection.connection_ref == result.connection.connection_ref


def test_concurrent_exchange_only_once() -> None:
    service, _, provider = _build_service()
    begin = service.begin_connection_authorization(
        provider_id=_PROVIDER,
        redirect_uri=_REDIRECT,
    )
    stored = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=begin.authorization_transaction_ref,
    )
    state = build_state_parameter(
        tenant_id=_TENANT,
        authorization_transaction_ref=begin.authorization_transaction_ref,
        correlation_state=stored.correlation_state,
    )
    service.complete_connection_authorization(authorization_code="code-1", state=state)
    with pytest.raises(TenantConnectionProductError) as exc:
        service.complete_connection_authorization(authorization_code="code-1", state=state)
    assert exc.value.error_code == "authorization_callback_replay"
    assert provider.exchange_calls == 1


def test_crash_resume_with_staged_credentials() -> None:
    secrets = _SecretsStore()
    provider = _FakeOAuthProvider()
    service, _, _ = _build_service(secrets=secrets, provider=provider)
    begin = service.begin_connection_authorization(
        provider_id=_PROVIDER,
        redirect_uri=_REDIRECT,
    )
    stored = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=begin.authorization_transaction_ref,
    )
    assert stored is not None
    staging = stored.credential_staging_ref
    assert staging is not None
    secrets.put_secret(staging, json.dumps({"access_token": "staged"}))
    exchanging = TenantConnectionAuthorizationTransaction(
        authorization_transaction_ref=stored.authorization_transaction_ref,
        tenant_id=stored.tenant_id,
        provider_id=stored.provider_id,
        correlation_state=stored.correlation_state,
        redirect_uri=stored.redirect_uri,
        connection_ref=stored.connection_ref,
        verifier_secret_ref=stored.verifier_secret_ref,
        credential_staging_ref=staging,
        created_at=stored.created_at,
        expires_at=stored.expires_at,
        completion_state=TenantConnectionAuthorizationCompletionState.EXCHANGING,
        version=stored.version + 1,
    )
    service._transaction_repository.replace_if_match(stored, exchanging)
    state = build_state_parameter(
        tenant_id=_TENANT,
        authorization_transaction_ref=begin.authorization_transaction_ref,
        correlation_state=stored.correlation_state,
    )
    result = service.complete_connection_authorization(authorization_code="code-2", state=state)
    assert result.disposition == "created"
    assert provider.exchange_calls == 0


def test_revoke_idempotent() -> None:
    service, _, _ = _build_service()
    begin = service.begin_connection_authorization(
        provider_id=_PROVIDER,
        redirect_uri=_REDIRECT,
    )
    stored = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=begin.authorization_transaction_ref,
    )
    state = build_state_parameter(
        tenant_id=_TENANT,
        authorization_transaction_ref=begin.authorization_transaction_ref,
        correlation_state=stored.correlation_state,
    )
    created = service.complete_connection_authorization(authorization_code="code", state=state)
    revoked = service.revoke_connection(connection_ref=created.connection.connection_ref)
    assert revoked.administrative_status is TenantConnectionAdministrativeStatus.REVOKED
    again = service.revoke_connection(connection_ref=created.connection.connection_ref)
    assert again.administrative_status is TenantConnectionAdministrativeStatus.REVOKED


def test_redirect_not_allowlisted() -> None:
    service, _, _ = _build_service()
    with pytest.raises(TenantConnectionProductError) as exc:
        service.begin_connection_authorization(
            provider_id=_PROVIDER,
            redirect_uri="https://evil.example/callback",
        )
    assert exc.value.error_code == "authorization_redirect_not_allowed"


def test_state_tenant_mismatch() -> None:
    service, _, _ = _build_service()
    begin = service.begin_connection_authorization(
        provider_id=_PROVIDER,
        redirect_uri=_REDIRECT,
    )
    stored = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=begin.authorization_transaction_ref,
    )
    bad_state = build_state_parameter(
        tenant_id="other-tenant",
        authorization_transaction_ref=begin.authorization_transaction_ref,
        correlation_state=stored.correlation_state,
    )
    with pytest.raises(TenantConnectionProductError) as exc:
        service.complete_connection_authorization(authorization_code="code", state=bad_state)
    assert exc.value.error_code == "tenant_mismatch"


def test_parse_state_parameter() -> None:
    state = build_state_parameter(
        tenant_id=_TENANT,
        authorization_transaction_ref="auth.ref",
        correlation_state="nonce",
    )
    tenant, ref, nonce = parse_state_parameter(state)
    assert tenant == _TENANT
    assert ref == "auth.ref"
    assert nonce == "nonce"


def _replace_transaction(
    service: TenantConnectionProductOrchestrationService,
    replacement: TenantConnectionAuthorizationTransaction,
) -> TenantConnectionAuthorizationTransaction:
    current = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=replacement.authorization_transaction_ref,
    )
    assert current is not None
    service._transaction_repository.replace_if_match(current, replacement)
    return replacement


def _expired_transaction(
    stored: TenantConnectionAuthorizationTransaction,
    *,
    completion_state: TenantConnectionAuthorizationCompletionState,
    exchange_started_at: datetime | None = None,
) -> TenantConnectionAuthorizationTransaction:
    return TenantConnectionAuthorizationTransaction(
        authorization_transaction_ref=stored.authorization_transaction_ref,
        tenant_id=stored.tenant_id,
        provider_id=stored.provider_id,
        correlation_state=stored.correlation_state,
        redirect_uri=stored.redirect_uri,
        connection_ref=stored.connection_ref,
        verifier_secret_ref=stored.verifier_secret_ref,
        credential_staging_ref=stored.credential_staging_ref,
        created_at=stored.created_at,
        expires_at=stored.created_at,
        completion_state=completion_state,
        completion_claim_expires_at=(
            _NOW - timedelta(minutes=1)
            if completion_state is TenantConnectionAuthorizationCompletionState.CLAIMED
            else None
        ),
        exchange_started_at=exchange_started_at,
        version=stored.version + 1,
    )


def _begin_expired_pending(
    service: TenantConnectionProductOrchestrationService,
    secrets: _SecretsStore,
) -> tuple[TenantConnectionAuthorizationTransaction, str]:
    begin = service.begin_connection_authorization(
        provider_id=_PROVIDER,
        redirect_uri=_REDIRECT,
    )
    stored = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=begin.authorization_transaction_ref,
    )
    assert stored is not None
    verifier_path = verifier_secret_path(
        tenant_id=_TENANT,
        authorization_transaction_ref=begin.authorization_transaction_ref,
    )
    pending = _replace_transaction(
        service,
        _expired_transaction(stored, completion_state=TenantConnectionAuthorizationCompletionState.PENDING),
    )
    return pending, verifier_path


@patch(
    "local_workspace_application.workspaces.tenant_connection_product_orchestration._utcnow",
    return_value=_NOW,
)
def test_cleanup_expired_pending_deletes_verifier(mock_now) -> None:
    _ = mock_now
    service, secrets, _ = _build_service()
    pending, verifier_path = _begin_expired_pending(service, secrets)
    assert secrets.store[verifier_path]

    cleaned = service.cleanup_expired_authorization_transactions()
    assert cleaned == 1

    stored = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=pending.authorization_transaction_ref,
    )
    assert stored is not None
    assert stored.completion_state is TenantConnectionAuthorizationCompletionState.EXPIRED
    assert verifier_path in secrets.deleted
    assert verifier_path not in secrets.store


@patch(
    "local_workspace_application.workspaces.tenant_connection_product_orchestration._utcnow",
    return_value=_NOW,
)
def test_cleanup_expired_claimed_deletes_verifier(mock_now) -> None:
    _ = mock_now
    service, secrets, _ = _build_service()
    pending, verifier_path = _begin_expired_pending(service, secrets)
    claimed = _replace_transaction(
        service,
        TenantConnectionAuthorizationTransaction(
            **{
                **pending.model_dump(),
                "completion_state": TenantConnectionAuthorizationCompletionState.CLAIMED,
                "completion_claim_expires_at": _NOW + timedelta(minutes=2),
                "version": pending.version + 1,
            }
        ),
    )

    cleaned = service.cleanup_expired_authorization_transactions()
    assert cleaned == 1

    stored = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=claimed.authorization_transaction_ref,
    )
    assert stored is not None
    assert stored.completion_state is TenantConnectionAuthorizationCompletionState.EXPIRED
    assert verifier_path in secrets.deleted


@patch(
    "local_workspace_application.workspaces.tenant_connection_product_orchestration._utcnow",
    return_value=_NOW,
)
def test_exchanging_is_never_reset_to_pending(mock_now) -> None:
    _ = mock_now
    service, secrets, _ = _build_service()
    pending, _ = _begin_expired_pending(service, secrets)
    exchanging = _replace_transaction(
        service,
        TenantConnectionAuthorizationTransaction(
            **{
                **pending.model_dump(),
                "completion_state": TenantConnectionAuthorizationCompletionState.EXCHANGING,
                "completion_claim_expires_at": None,
                "exchange_started_at": _NOW - timedelta(seconds=30),
                "version": pending.version + 1,
            }
        ),
    )

    with pytest.raises(TenantConnectionProductError) as exc:
        service.complete_connection_authorization(
            authorization_code="code-should-not-retry",
            authorization_transaction_ref=exchanging.authorization_transaction_ref,
        )
    assert exc.value.error_code == "authorization_already_in_progress"

    stored = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=exchanging.authorization_transaction_ref,
    )
    assert stored is not None
    assert stored.completion_state is TenantConnectionAuthorizationCompletionState.EXCHANGING


@patch(
    "local_workspace_application.workspaces.tenant_connection_product_orchestration._utcnow",
    return_value=_NOW,
)
def test_cleanup_exchanging_with_staged_credentials_completes(mock_now) -> None:
    _ = mock_now
    service, secrets, provider = _build_service()
    pending, _ = _begin_expired_pending(service, secrets)
    staging = pending.credential_staging_ref
    assert staging is not None
    secrets.put_secret(staging, json.dumps({"access_token": "staged"}))
    _replace_transaction(
        service,
        TenantConnectionAuthorizationTransaction(
            **{
                **pending.model_dump(),
                "completion_state": TenantConnectionAuthorizationCompletionState.EXCHANGING,
                "exchange_started_at": _NOW - timedelta(minutes=5),
                "version": pending.version + 1,
            }
        ),
    )

    cleaned = service.cleanup_expired_authorization_transactions()
    assert cleaned == 1
    assert provider.exchange_calls == 0

    stored = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=pending.authorization_transaction_ref,
    )
    assert stored is not None
    assert stored.completion_state is TenantConnectionAuthorizationCompletionState.COMPLETED


@patch(
    "local_workspace_application.workspaces.tenant_connection_product_orchestration._utcnow",
    return_value=_NOW,
)
def test_abandoned_exchanging_without_staged_credentials_becomes_unknown(mock_now) -> None:
    _ = mock_now
    service, secrets, provider = _build_service()
    pending, verifier_path = _begin_expired_pending(service, secrets)
    _replace_transaction(
        service,
        TenantConnectionAuthorizationTransaction(
            **{
                **pending.model_dump(),
                "completion_state": TenantConnectionAuthorizationCompletionState.EXCHANGING,
                "exchange_started_at": _NOW - timedelta(minutes=5),
                "version": pending.version + 1,
            }
        ),
    )

    cleaned = service.cleanup_expired_authorization_transactions()
    assert cleaned == 1
    assert provider.exchange_calls == 0

    stored = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=pending.authorization_transaction_ref,
    )
    assert stored is not None
    assert (
        stored.completion_state
        is TenantConnectionAuthorizationCompletionState.EXCHANGE_OUTCOME_UNKNOWN
    )
    assert verifier_path in secrets.deleted


@patch(
    "local_workspace_application.workspaces.tenant_connection_product_orchestration._utcnow",
    return_value=_NOW,
)
def test_recovery_never_re_exchanges_authorization_code(mock_now) -> None:
    _ = mock_now
    service, secrets, provider = _build_service()
    pending, _ = _begin_expired_pending(service, secrets)
    staging = pending.credential_staging_ref
    assert staging is not None
    secrets.put_secret(staging, json.dumps({"access_token": "staged"}))
    exchanging = _replace_transaction(
        service,
        TenantConnectionAuthorizationTransaction(
            **{
                **pending.model_dump(),
                "completion_state": TenantConnectionAuthorizationCompletionState.EXCHANGING,
                "exchange_started_at": _NOW - timedelta(minutes=5),
                "version": pending.version + 1,
            }
        ),
    )
    state = build_state_parameter(
        tenant_id=_TENANT,
        authorization_transaction_ref=exchanging.authorization_transaction_ref,
        correlation_state=exchanging.correlation_state,
    )

    service.cleanup_expired_authorization_transactions()
    assert provider.exchange_calls == 0

    with pytest.raises(TenantConnectionProductError) as exc:
        service.complete_connection_authorization(authorization_code="code-again", state=state)
    assert exc.value.error_code == "authorization_callback_replay"
    assert provider.exchange_calls == 0


@patch(
    "local_workspace_application.workspaces.tenant_connection_product_orchestration._utcnow",
    return_value=_NOW,
)
def test_credentials_obtained_survives_auth_ttl(mock_now) -> None:
    _ = mock_now
    service, secrets, provider = _build_service()
    pending, _ = _begin_expired_pending(service, secrets)
    staging = pending.credential_staging_ref
    assert staging is not None
    secrets.put_secret(staging, json.dumps({"access_token": "staged"}))
    _replace_transaction(
        service,
        TenantConnectionAuthorizationTransaction(
            **{
                **pending.model_dump(),
                "completion_state": TenantConnectionAuthorizationCompletionState.CREDENTIALS_OBTAINED,
                "version": pending.version + 1,
            }
        ),
    )

    cleaned = service.cleanup_expired_authorization_transactions()
    assert cleaned == 1
    assert provider.exchange_calls == 0
    assert staging in secrets.deleted

    stored = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=pending.authorization_transaction_ref,
    )
    assert stored is not None
    assert stored.completion_state is TenantConnectionAuthorizationCompletionState.COMPLETED


@patch(
    "local_workspace_application.workspaces.tenant_connection_product_orchestration._utcnow",
    return_value=_NOW,
)
def test_cleanup_does_not_delete_staged_credentials_before_finalize(mock_now) -> None:
    _ = mock_now
    service, secrets, _ = _build_service()
    pending, _ = _begin_expired_pending(service, secrets)
    staging = pending.credential_staging_ref
    assert staging is not None
    secrets.put_secret(staging, json.dumps({"access_token": "staged"}))
    _replace_transaction(
        service,
        TenantConnectionAuthorizationTransaction(
            **{
                **pending.model_dump(),
                "completion_state": TenantConnectionAuthorizationCompletionState.CREDENTIALS_OBTAINED,
                "version": pending.version + 1,
            }
        ),
    )

    with patch.object(
        service,
        "_cleanup_transaction_secrets",
        wraps=service._cleanup_transaction_secrets,
    ) as cleanup_secrets:
        service.cleanup_expired_authorization_transactions()
        cleanup_secrets.assert_not_called()


@patch(
    "local_workspace_application.workspaces.tenant_connection_product_orchestration._utcnow",
    return_value=_NOW,
)
def test_credentials_obtained_resumes_after_simulated_restart(mock_now) -> None:
    _ = mock_now
    document_store = InMemoryDocumentStore()
    secrets = _SecretsStore()
    provider = _FakeOAuthProvider()
    service, _, _ = _build_service(secrets=secrets, provider=provider, store=document_store)
    pending, _ = _begin_expired_pending(service, secrets)
    staging = pending.credential_staging_ref
    assert staging is not None
    secrets.put_secret(staging, json.dumps({"access_token": "staged"}))
    obtained = _replace_transaction(
        service,
        TenantConnectionAuthorizationTransaction(
            **{
                **pending.model_dump(),
                "completion_state": TenantConnectionAuthorizationCompletionState.CREDENTIALS_OBTAINED,
                "version": pending.version + 1,
            }
        ),
    )

    restarted_service, _, restarted_provider = _build_service(
        secrets=secrets,
        provider=provider,
        store=document_store,
    )
    result = restarted_service.complete_connection_authorization(
        authorization_transaction_ref=obtained.authorization_transaction_ref,
    )
    assert result.disposition == "created"
    assert restarted_provider.exchange_calls == 0


@patch(
    "local_workspace_application.workspaces.tenant_connection_product_orchestration._utcnow",
    return_value=_NOW,
)
def test_terminal_state_cleanup_is_idempotent(mock_now) -> None:
    _ = mock_now
    for terminal in (
        TenantConnectionAuthorizationCompletionState.COMPLETED,
        TenantConnectionAuthorizationCompletionState.EXCHANGE_OUTCOME_UNKNOWN,
        TenantConnectionAuthorizationCompletionState.EXPIRED,
    ):
        service, secrets, _ = _build_service()
        pending, _ = _begin_expired_pending(service, secrets)
        txn = _replace_transaction(
            service,
            TenantConnectionAuthorizationTransaction(
                **{
                    **pending.model_dump(),
                    "completion_state": terminal,
                    "consumed_at": (
                        _NOW
                        if terminal is TenantConnectionAuthorizationCompletionState.COMPLETED
                        else None
                    ),
                    "version": pending.version + 1,
                }
            ),
        )
        assert service.cleanup_expired_authorization_transactions() == 0
        stored = service._transaction_repository.get(
            tenant_id=_TENANT,
            authorization_transaction_ref=txn.authorization_transaction_ref,
        )
        assert stored is not None
        assert stored.completion_state is terminal
        assert service.cleanup_expired_authorization_transactions() == 0


@patch(
    "local_workspace_application.workspaces.tenant_connection_product_orchestration._utcnow",
    return_value=_NOW,
)
def test_concurrent_credentials_obtained_recovery_is_cas_safe(mock_now) -> None:
    _ = mock_now
    service, secrets, provider = _build_service()
    pending, _ = _begin_expired_pending(service, secrets)
    staging = pending.credential_staging_ref
    assert staging is not None
    secrets.put_secret(staging, json.dumps({"access_token": "staged"}))
    obtained = _replace_transaction(
        service,
        TenantConnectionAuthorizationTransaction(
            **{
                **pending.model_dump(),
                "completion_state": TenantConnectionAuthorizationCompletionState.CREDENTIALS_OBTAINED,
                "version": pending.version + 1,
            }
        ),
    )

    first = service.cleanup_expired_authorization_transactions()
    second = service.cleanup_expired_authorization_transactions()
    assert first == 1
    assert second == 0
    assert provider.exchange_calls == 0

    stored = service._transaction_repository.get(
        tenant_id=_TENANT,
        authorization_transaction_ref=obtained.authorization_transaction_ref,
    )
    assert stored is not None
    assert stored.completion_state is TenantConnectionAuthorizationCompletionState.COMPLETED
