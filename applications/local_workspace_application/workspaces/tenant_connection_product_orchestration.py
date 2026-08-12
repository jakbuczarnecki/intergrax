# © Artur Czarnecki. All rights reserved.

"""Reusable tenant connection product orchestration (PRODUCT-5B)."""

from __future__ import annotations

import secrets
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.tenant_connection_auth import (
    TenantConnectionAuthMode,
    TenantConnectionAuthProviderRegistry,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import TenantConnectionRehydrator
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnection,
    TenantConnectionAdministrativeStatus,
    TenantConnectionAlreadyExists,
    TenantConnectionNotFound,
    TenantConnectionService,
    TenantConnectionVersionConflict,
    to_safe_tenant_connection,
)
from intergrax.runtime.vendor_knowledge.models import JsonValue
from local_workspace_application.workspaces.tenant_connection_authorization_transaction import (
    TenantConnectionAuthorizationCompletionState,
    TenantConnectionAuthorizationTransaction,
    TenantConnectionAuthorizationTransactionRepository,
    build_state_parameter,
    connection_credential_secret_path,
    credential_staging_secret_path,
    parse_state_parameter,
    verifier_secret_path,
)
from local_workspace_application.workspaces.tenant_connection_product_errors import (
    TenantConnectionProductError,
    tenant_connection_product_error,
)


@dataclass(frozen=True, slots=True)
class ConnectionAuthBeginResult:
    authorization_transaction_ref: str
    authorization_url: str | None
    expires_at: datetime
    required_user_action: Literal["redirect", "present_manual_instructions"]
    manual_instructions: str | None = None


@dataclass(frozen=True, slots=True)
class ConnectionAuthCompleteResult:
    connection: SafeTenantConnectionV1
    disposition: Literal["created", "reconnected", "already_exists"]


@dataclass(frozen=True, slots=True)
class TenantConnectionProductOrchestrationConfig:
    redirect_allowlist: frozenset[str]
    transaction_ttl_seconds: int = 900
    completion_claim_ttl_seconds: int = 120


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_transaction_ref(provider_id: str) -> str:
    slug = provider_id.replace(".", "-").replace("_", "-")[:24]
    return f"auth.{slug}.{secrets.token_urlsafe(10)}"


def _new_connection_ref(provider_id: str) -> str:
    slug = provider_id.replace(".", "-").replace("_", "-")[:24]
    return f"conn.{slug}.{secrets.token_urlsafe(8)}"


def _embed_oauth_state(authorization_url: str, state_value: str) -> str:
    parsed = urllib.parse.urlparse(authorization_url)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    query["state"] = [state_value]
    encoded = urllib.parse.urlencode(query, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=encoded))


class TenantConnectionProductOrchestrationService:
    """Tenant-scoped product orchestration for connection authorization."""

    def __init__(
        self,
        *,
        tenant_id: str,
        connection_repository: DocumentStoreTenantConnectionRepository,
        transaction_repository: TenantConnectionAuthorizationTransactionRepository,
        secrets_store: SecretsStore,
        auth_provider_registry: TenantConnectionAuthProviderRegistry,
        rehydrator: TenantConnectionRehydrator,
        connection_registry: KnowledgeConnectionRegistry,
        config: TenantConnectionProductOrchestrationConfig,
    ) -> None:
        cleaned_tenant = tenant_id.strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        self._tenant_id = cleaned_tenant
        self._connection_repository = connection_repository
        self._transaction_repository = transaction_repository
        self._secrets_store = secrets_store
        self._auth_providers = auth_provider_registry
        self._rehydrator = rehydrator
        self._connection_registry = connection_registry
        self._config = config
        self._connection_service = TenantConnectionService(
            tenant_id=cleaned_tenant,
            repository=connection_repository,
        )

    def list_supported_connection_providers(
        self,
    ) -> tuple[Mapping[str, JsonValue], ...]:
        return tuple(
            {
                "provider_id": descriptor.provider_id,
                "integration_kind": descriptor.integration_kind.value,
                "auth_mode": descriptor.auth_mode.value,
                "safe_display_name": descriptor.safe_display_name,
                "supported_scopes_summary": descriptor.supported_scopes_summary,
                "qualification": descriptor.qualification.value,
            }
            for descriptor in self._auth_providers.list_descriptors()
        )

    def begin_connection_authorization(
        self,
        *,
        provider_id: str,
        redirect_uri: str | None = None,
        safe_display_name: str | None = None,
        connection_ref: str | None = None,
    ) -> ConnectionAuthBeginResult:
        provider = self._resolve_provider(provider_id)
        cleaned_redirect = (redirect_uri or "").strip()
        if provider.auth_mode is TenantConnectionAuthMode.OAUTH_DELEGATED:
            if not cleaned_redirect:
                raise tenant_connection_product_error("authorization_redirect_not_allowed")
            if cleaned_redirect not in self._config.redirect_allowlist:
                raise tenant_connection_product_error("authorization_redirect_not_allowed")
        reconnect_ref = connection_ref.strip() if connection_ref else None
        if reconnect_ref:
            existing = self._connection_repository.get(
                tenant_id=self._tenant_id,
                connection_ref=reconnect_ref,
            )
            if existing is None:
                raise tenant_connection_product_error("connection_not_found")
            if existing.administrative_status is TenantConnectionAdministrativeStatus.REVOKED:
                raise tenant_connection_product_error("connection_revoked")
            if existing.provider_id != provider.provider_id:
                raise tenant_connection_product_error("authorization_state_invalid")

        now = _utcnow()
        transaction_ref = _new_transaction_ref(provider.provider_id)
        try:
            begin = provider.begin_authorization(
                tenant_id=self._tenant_id,
                redirect_uri=cleaned_redirect,
                reconnect_connection_ref=reconnect_ref,
            )
        except ValueError as exc:
            if str(exc) == "connection_provider_misconfigured":
                raise tenant_connection_product_error("connection_provider_misconfigured") from exc
            raise
        verifier_ref: str | None = None
        if begin.code_verifier is not None:
            verifier_ref = verifier_secret_path(
                tenant_id=self._tenant_id,
                authorization_transaction_ref=transaction_ref,
            )
            self._secrets_store.put_secret(verifier_ref, begin.code_verifier)

        staging_ref = credential_staging_secret_path(
            tenant_id=self._tenant_id,
            authorization_transaction_ref=transaction_ref,
        )
        expires_at = now + timedelta(seconds=self._config.transaction_ttl_seconds)
        transaction = TenantConnectionAuthorizationTransaction(
            authorization_transaction_ref=transaction_ref,
            tenant_id=self._tenant_id,
            provider_id=provider.provider_id,
            correlation_state=begin.correlation_state,
            redirect_uri=cleaned_redirect or "",
            connection_ref=reconnect_ref,
            verifier_secret_ref=verifier_ref,
            credential_staging_ref=staging_ref,
            created_at=now,
            expires_at=expires_at,
            completion_state=TenantConnectionAuthorizationCompletionState.PENDING,
            version=1,
        )
        self._transaction_repository.create(transaction)

        authorization_url = begin.authorization_url
        if authorization_url is not None:
            state_value = build_state_parameter(
                tenant_id=self._tenant_id,
                authorization_transaction_ref=transaction_ref,
                correlation_state=begin.correlation_state,
            )
            authorization_url = _embed_oauth_state(authorization_url, state_value)

        _ = safe_display_name
        return ConnectionAuthBeginResult(
            authorization_transaction_ref=transaction_ref,
            authorization_url=authorization_url,
            expires_at=expires_at,
            required_user_action=begin.required_user_action,
            manual_instructions=begin.manual_instructions,
        )

    def complete_connection_authorization(
        self,
        *,
        authorization_transaction_ref: str | None = None,
        authorization_code: str | None = None,
        state: str | None = None,
        credential_payload: Mapping[str, JsonValue] | None = None,
    ) -> ConnectionAuthCompleteResult:
        resolved_ref = authorization_transaction_ref
        if resolved_ref is None and state:
            _, resolved_ref, _ = parse_state_parameter(state)
        if not resolved_ref or not resolved_ref.strip():
            raise tenant_connection_product_error("authorization_transaction_not_found")

        transaction = self._transaction_repository.get(
            tenant_id=self._tenant_id,
            authorization_transaction_ref=resolved_ref.strip(),
        )
        if transaction is None:
            raise tenant_connection_product_error("authorization_transaction_not_found")

        provider = self._resolve_provider(transaction.provider_id)
        now = _utcnow()

        if state is not None:
            state_tenant, parsed_ref, nonce = parse_state_parameter(state)
            if state_tenant != self._tenant_id:
                raise tenant_connection_product_error("tenant_mismatch")
            if parsed_ref != transaction.authorization_transaction_ref:
                raise tenant_connection_product_error("authorization_state_invalid")
            if nonce != transaction.correlation_state:
                raise tenant_connection_product_error("authorization_state_invalid")

        if transaction.completion_state is TenantConnectionAuthorizationCompletionState.COMPLETED:
            raise tenant_connection_product_error("authorization_callback_replay")
        if transaction.completion_state is TenantConnectionAuthorizationCompletionState.EXCHANGE_OUTCOME_UNKNOWN:
            raise tenant_connection_product_error("authorization_exchange_outcome_unknown")
        if transaction.completion_state is TenantConnectionAuthorizationCompletionState.EXPIRED:
            raise tenant_connection_product_error("authorization_transaction_expired")

        if transaction.expires_at < now and transaction.completion_state in {
            TenantConnectionAuthorizationCompletionState.PENDING,
            TenantConnectionAuthorizationCompletionState.CLAIMED,
        }:
            self._cleanup_transaction_secrets(transaction)
            self._terminalize_transaction(
                transaction,
                TenantConnectionAuthorizationCompletionState.EXPIRED,
            )
            raise tenant_connection_product_error("authorization_transaction_expired")

        if transaction.completion_state is TenantConnectionAuthorizationCompletionState.EXCHANGING:
            staged = self._read_secret(transaction.credential_staging_ref or "")
            if staged is not None:
                obtained = self._transition_transaction(
                    transaction,
                    completion_state=TenantConnectionAuthorizationCompletionState.CREDENTIALS_OBTAINED,
                    completion_claim_expires_at=None,
                )
                provider = self._resolve_provider(transaction.provider_id)
                return self._finalize_from_staged_credentials(obtained, provider)
            raise tenant_connection_product_error("authorization_already_in_progress")

        if transaction.completion_state is TenantConnectionAuthorizationCompletionState.CREDENTIALS_OBTAINED:
            return self._finalize_from_staged_credentials(transaction, provider)

        if provider.auth_mode is TenantConnectionAuthMode.MANUAL_CREDENTIAL_BINDING:
            if credential_payload is None:
                raise tenant_connection_product_error("credential_binding_invalid")
            return self._complete_manual_binding(transaction, provider, credential_payload)

        if not authorization_code or not authorization_code.strip():
            raise tenant_connection_product_error("authorization_state_invalid")
        return self._complete_oauth_exchange(
            transaction,
            provider,
            authorization_code=authorization_code.strip(),
        )

    def get_connection(self, connection_ref: str) -> SafeTenantConnectionV1:
        try:
            return self._connection_service.get_safe(connection_ref)
        except TenantConnectionNotFound:
            raise tenant_connection_product_error("connection_not_found") from None

    def list_connections(
        self,
        *,
        administrative_status: TenantConnectionAdministrativeStatus | None = None,
        limit: int = 100,
    ) -> tuple[SafeTenantConnectionV1, ...]:
        return self._connection_service.list_safe(
            limit=limit,
            administrative_status=administrative_status,
        )

    def reconnect_connection(
        self,
        *,
        connection_ref: str,
        redirect_uri: str,
    ) -> ConnectionAuthBeginResult:
        connection = self._connection_repository.get(
            tenant_id=self._tenant_id,
            connection_ref=connection_ref.strip(),
        )
        if connection is None:
            raise tenant_connection_product_error("connection_not_found")
        if connection.administrative_status is TenantConnectionAdministrativeStatus.REVOKED:
            raise tenant_connection_product_error("connection_revoked")
        if connection.administrative_status not in {
            TenantConnectionAdministrativeStatus.ACTIVE,
            TenantConnectionAdministrativeStatus.DISABLED,
        }:
            raise tenant_connection_product_error("connection_not_active")
        return self.begin_connection_authorization(
            provider_id=connection.provider_id,
            redirect_uri=redirect_uri,
            connection_ref=connection.connection_ref,
            safe_display_name=connection.safe_display_name,
        )

    def revoke_connection(
        self,
        *,
        connection_ref: str,
        idempotency_key: str | None = None,
    ) -> SafeTenantConnectionV1:
        _ = idempotency_key
        connection = self._connection_repository.get(
            tenant_id=self._tenant_id,
            connection_ref=connection_ref.strip(),
        )
        if connection is None:
            raise tenant_connection_product_error("connection_not_found")

        if connection.administrative_status is TenantConnectionAdministrativeStatus.REVOKED:
            return to_safe_tenant_connection(connection)

        provider = self._auth_providers.get(connection.provider_id)
        credential = self._read_secret(connection.credential_ref)
        if provider is not None and credential is not None:
            try:
                provider.revoke_remote_credentials(
                    tenant_id=self._tenant_id,
                    credential_bundle_json=credential,
                )
            except Exception:
                pass

        self._delete_secret(connection.credential_ref)
        self._connection_registry.deregister(
            tenant_id=self._tenant_id,
            connection_ref=connection.connection_ref,
        )

        now = _utcnow()
        if now <= connection.updated_at:
            now = connection.updated_at + timedelta(microseconds=1)
        revoked = TenantConnection(
            connection_ref=connection.connection_ref,
            tenant_id=connection.tenant_id,
            provider_id=connection.provider_id,
            integration_kind=connection.integration_kind,
            safe_display_name=connection.safe_display_name,
            administrative_status=TenantConnectionAdministrativeStatus.REVOKED,
            credential_ref=connection.credential_ref,
            validated_secret_free_config=connection.validated_secret_free_config,
            configuration_version=connection.configuration_version + 1,
            created_at=connection.created_at,
            updated_at=now,
            connected_principal_ref=connection.connected_principal_ref,
        )
        try:
            self._connection_service.update(
                revoked,
                expected_configuration_version=connection.configuration_version,
            )
        except TenantConnectionVersionConflict:
            raise tenant_connection_product_error("connection_version_conflict") from None
        return to_safe_tenant_connection(revoked)

    def cleanup_expired_authorization_transactions(
        self,
        *,
        limit: int = 100,
    ) -> int:
        now = _utcnow()
        expired = self._transaction_repository.list_expired(
            tenant_id=self._tenant_id,
            now=now,
            limit=limit,
        )
        cleaned = 0
        for transaction in expired:
            if transaction.completion_state in {
                TenantConnectionAuthorizationCompletionState.COMPLETED,
                TenantConnectionAuthorizationCompletionState.EXCHANGE_OUTCOME_UNKNOWN,
                TenantConnectionAuthorizationCompletionState.EXPIRED,
            }:
                continue
            self._cleanup_transaction_secrets(transaction)
            self._terminalize_transaction(
                transaction,
                TenantConnectionAuthorizationCompletionState.EXPIRED,
            )
            cleaned += 1
        return cleaned

    def _resolve_provider(self, provider_id: str) -> object:
        provider = self._auth_providers.get(provider_id)
        if provider is None:
            raise tenant_connection_product_error("connection_provider_unsupported")
        return provider

    def _complete_manual_binding(
        self,
        transaction: TenantConnectionAuthorizationTransaction,
        provider: object,
        credential_payload: Mapping[str, JsonValue],
    ) -> ConnectionAuthCompleteResult:
        claimed = self._claim_transaction(transaction)
        try:
            bind_result = provider.bind_manual_credentials(
                tenant_id=self._tenant_id,
                credential_payload=credential_payload,
            )
        except ValueError:
            self._release_claim_if_possible(claimed)
            raise tenant_connection_product_error("credential_binding_invalid") from None

        staging_ref = transaction.credential_staging_ref
        if staging_ref is None:
            raise tenant_connection_product_error("authorization_state_invalid")
        self._secrets_store.put_secret(staging_ref, bind_result.credential_bundle_json)
        obtained = self._transition_transaction(
            claimed,
            completion_state=TenantConnectionAuthorizationCompletionState.CREDENTIALS_OBTAINED,
            completion_claim_expires_at=None,
        )
        return self._finalize_from_staged_credentials(
            obtained,
            provider,
            connected_principal_ref=bind_result.connected_principal_ref,
        )

    def _complete_oauth_exchange(
        self,
        transaction: TenantConnectionAuthorizationTransaction,
        provider: object,
        *,
        authorization_code: str,
    ) -> ConnectionAuthCompleteResult:
        if transaction.completion_state is TenantConnectionAuthorizationCompletionState.CREDENTIALS_OBTAINED:
            return self._finalize_from_staged_credentials(transaction, provider)

        claimed = self._claim_transaction(transaction)
        if claimed.completion_state is TenantConnectionAuthorizationCompletionState.CREDENTIALS_OBTAINED:
            return self._finalize_from_staged_credentials(claimed, provider)

        staged = self._read_secret(claimed.credential_staging_ref or "")
        if staged is not None:
            obtained = self._transition_transaction(
                claimed,
                completion_state=TenantConnectionAuthorizationCompletionState.CREDENTIALS_OBTAINED,
                completion_claim_expires_at=None,
            )
            return self._finalize_from_staged_credentials(obtained, provider)

        exchanging = self._transition_transaction(
            claimed,
            completion_state=TenantConnectionAuthorizationCompletionState.EXCHANGING,
            completion_claim_expires_at=None,
        )

        verifier = self._read_verifier(exchanging)
        if verifier is None:
            self._mark_exchange_outcome_unknown(exchanging)
            raise tenant_connection_product_error("authorization_exchange_outcome_unknown")

        try:
            exchange = provider.exchange_authorization_code(
                tenant_id=self._tenant_id,
                redirect_uri=exchanging.redirect_uri,
                authorization_code=authorization_code,
                code_verifier=verifier,
                correlation_state=exchanging.correlation_state,
            )
        except Exception:
            self._mark_exchange_outcome_unknown(exchanging)
            raise tenant_connection_product_error(
                "authorization_exchange_outcome_unknown"
            ) from None

        staging_ref = exchanging.credential_staging_ref
        if staging_ref is None:
            self._mark_exchange_outcome_unknown(exchanging)
            raise tenant_connection_product_error("authorization_exchange_outcome_unknown")

        self._secrets_store.put_secret(staging_ref, exchange.credential_bundle_json)
        self._delete_verifier(exchanging)

        obtained = self._reload_transaction(exchanging.authorization_transaction_ref)
        if obtained is None:
            raise tenant_connection_product_error("authorization_transaction_not_found")
        if obtained.completion_state is not TenantConnectionAuthorizationCompletionState.EXCHANGING:
            raise tenant_connection_product_error("authorization_already_in_progress")

        obtained_transition = self._transition_transaction(
            obtained,
            completion_state=TenantConnectionAuthorizationCompletionState.CREDENTIALS_OBTAINED,
            completion_claim_expires_at=None,
        )
        return self._finalize_from_staged_credentials(
            obtained_transition,
            provider,
            connected_principal_ref=exchange.connected_principal_ref,
        )

    def _finalize_from_staged_credentials(
        self,
        transaction: TenantConnectionAuthorizationTransaction,
        provider: object,
        *,
        connected_principal_ref: str | None = None,
    ) -> ConnectionAuthCompleteResult:
        staging_ref = transaction.credential_staging_ref
        if staging_ref is None:
            raise tenant_connection_product_error("authorization_state_invalid")
        bundle = self._read_secret(staging_ref)
        if bundle is None:
            if transaction.completion_state is TenantConnectionAuthorizationCompletionState.EXCHANGING:
                self._mark_exchange_outcome_unknown(transaction)
            raise tenant_connection_product_error("authorization_exchange_outcome_unknown")

        principal = connected_principal_ref
        reconnect_ref = transaction.connection_ref
        disposition: Literal["created", "reconnected", "already_exists"] = "created"
        now = _utcnow()

        if reconnect_ref:
            existing = self._connection_repository.get(
                tenant_id=self._tenant_id,
                connection_ref=reconnect_ref,
            )
            if existing is None:
                raise tenant_connection_product_error("connection_not_found")
            credential_ref = connection_credential_secret_path(
                tenant_id=self._tenant_id,
                connection_ref=reconnect_ref,
            )
            self._secrets_store.put_secret(credential_ref, bundle)
            self._delete_secret(staging_ref)
            secret_free = provider.build_secret_free_config(
                tenant_id=self._tenant_id,
                reconnect_connection=existing,
            )
            updated = TenantConnection(
                connection_ref=existing.connection_ref,
                tenant_id=existing.tenant_id,
                provider_id=existing.provider_id,
                integration_kind=existing.integration_kind,
                safe_display_name=existing.safe_display_name,
                administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
                credential_ref=credential_ref,
                validated_secret_free_config=dict(secret_free),
                configuration_version=existing.configuration_version + 1,
                created_at=existing.created_at,
                updated_at=now,
                connected_principal_ref=principal or existing.connected_principal_ref,
            )
            try:
                self._connection_service.update(
                    updated,
                    expected_configuration_version=existing.configuration_version,
                )
            except TenantConnectionVersionConflict:
                raise tenant_connection_product_error("connection_version_conflict") from None
            safe = self._rehydrate_or_fail(updated)
            completed = self._transition_transaction(
                transaction,
                completion_state=TenantConnectionAuthorizationCompletionState.COMPLETED,
                completion_claim_expires_at=None,
                consumed_at=now,
            )
            _ = completed
            return ConnectionAuthCompleteResult(connection=safe, disposition="reconnected")

        duplicate = self._find_duplicate_active_connection(
            provider_id=transaction.provider_id,
            connected_principal_ref=principal,
        )
        if duplicate is not None:
            completed = self._transition_transaction(
                transaction,
                completion_state=TenantConnectionAuthorizationCompletionState.COMPLETED,
                completion_claim_expires_at=None,
                consumed_at=now,
            )
            _ = completed
            self._delete_secret(staging_ref)
            return ConnectionAuthCompleteResult(
                connection=to_safe_tenant_connection(duplicate),
                disposition="already_exists",
            )

        connection_ref = _new_connection_ref(transaction.provider_id)
        credential_ref = connection_credential_secret_path(
            tenant_id=self._tenant_id,
            connection_ref=connection_ref,
        )
        self._secrets_store.put_secret(credential_ref, bundle)
        self._delete_secret(staging_ref)
        secret_free = provider.build_secret_free_config(
            tenant_id=self._tenant_id,
            reconnect_connection=None,
        )
        connection = TenantConnection(
            connection_ref=connection_ref,
            tenant_id=self._tenant_id,
            provider_id=transaction.provider_id,
            integration_kind=provider.integration_kind,
            safe_display_name=transaction.provider_id,
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
            credential_ref=credential_ref,
            validated_secret_free_config=dict(secret_free),
            configuration_version=1,
            created_at=now,
            updated_at=now,
            connected_principal_ref=principal,
        )
        try:
            self._connection_service.create(connection)
        except TenantConnectionAlreadyExists:
            raise tenant_connection_product_error("connection_already_exists") from None

        safe = self._rehydrate_or_fail(connection)
        completed = self._transition_transaction(
            transaction,
            completion_state=TenantConnectionAuthorizationCompletionState.COMPLETED,
            completion_claim_expires_at=None,
            consumed_at=now,
        )
        _ = completed
        return ConnectionAuthCompleteResult(connection=safe, disposition=disposition)

    def _rehydrate_or_fail(self, connection: TenantConnection) -> SafeTenantConnectionV1:
        results = self._rehydrator.rehydrate_tenant(tenant_id=self._tenant_id)
        target = next(
            (
                result
                for result in results
                if result.connection.connection_ref == connection.connection_ref
            ),
            None,
        )
        if target is None or target.status.value != "registered":
            raise tenant_connection_product_error("connection_runtime_unavailable")
        return target.connection

    def _find_duplicate_active_connection(
        self,
        *,
        provider_id: str,
        connected_principal_ref: str | None,
    ) -> TenantConnection | None:
        for connection in self._connection_repository.list(
            tenant_id=self._tenant_id,
            limit=1000,
            administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
        ):
            if connection.provider_id != provider_id:
                continue
            if connected_principal_ref is None:
                if connection.connected_principal_ref is None:
                    return connection
            elif connection.connected_principal_ref == connected_principal_ref:
                return connection
        return None

    def _claim_transaction(
        self,
        transaction: TenantConnectionAuthorizationTransaction,
    ) -> TenantConnectionAuthorizationTransaction:
        now = _utcnow()
        if transaction.completion_state is TenantConnectionAuthorizationCompletionState.CLAIMED:
            if (
                transaction.completion_claim_expires_at is not None
                and transaction.completion_claim_expires_at < now
            ):
                released = self._transition_transaction(
                    transaction,
                    completion_state=TenantConnectionAuthorizationCompletionState.PENDING,
                    completion_claim_expires_at=None,
                )
                transaction = released
            else:
                return transaction

        if transaction.completion_state is not TenantConnectionAuthorizationCompletionState.PENDING:
            if transaction.completion_state is TenantConnectionAuthorizationCompletionState.CREDENTIALS_OBTAINED:
                return transaction
            raise tenant_connection_product_error("authorization_already_in_progress")

        claim_expires = now + timedelta(seconds=self._config.completion_claim_ttl_seconds)
        return self._transition_transaction(
            transaction,
            completion_state=TenantConnectionAuthorizationCompletionState.CLAIMED,
            completion_claim_expires_at=claim_expires,
        )

    def _release_claim_if_possible(
        self,
        transaction: TenantConnectionAuthorizationTransaction,
    ) -> None:
        if transaction.completion_state is not TenantConnectionAuthorizationCompletionState.CLAIMED:
            return
        try:
            self._transition_transaction(
                transaction,
                completion_state=TenantConnectionAuthorizationCompletionState.PENDING,
                completion_claim_expires_at=None,
            )
        except TenantConnectionProductError:
            return

    def _transition_transaction(
        self,
        transaction: TenantConnectionAuthorizationTransaction,
        *,
        completion_state: TenantConnectionAuthorizationCompletionState,
        completion_claim_expires_at: datetime | None,
        consumed_at: datetime | None = None,
    ) -> TenantConnectionAuthorizationTransaction:
        replacement = TenantConnectionAuthorizationTransaction(
            authorization_transaction_ref=transaction.authorization_transaction_ref,
            tenant_id=transaction.tenant_id,
            provider_id=transaction.provider_id,
            correlation_state=transaction.correlation_state,
            redirect_uri=transaction.redirect_uri,
            connection_ref=transaction.connection_ref,
            verifier_secret_ref=transaction.verifier_secret_ref,
            credential_staging_ref=transaction.credential_staging_ref,
            created_at=transaction.created_at,
            expires_at=transaction.expires_at,
            completion_state=completion_state,
            completion_claim_expires_at=completion_claim_expires_at,
            consumed_at=consumed_at if consumed_at is not None else transaction.consumed_at,
            version=transaction.version + 1,
        )
        if not self._transaction_repository.replace_if_match(transaction, replacement):
            current = self._reload_transaction(transaction.authorization_transaction_ref)
            if current is None:
                raise tenant_connection_product_error("authorization_transaction_not_found")
            raise tenant_connection_product_error("authorization_already_in_progress")
        return replacement

    def _terminalize_transaction(
        self,
        transaction: TenantConnectionAuthorizationTransaction,
        terminal_state: TenantConnectionAuthorizationCompletionState,
    ) -> None:
        try:
            self._transition_transaction(
                transaction,
                completion_state=terminal_state,
                completion_claim_expires_at=None,
            )
        except TenantConnectionProductError:
            return

    def _mark_exchange_outcome_unknown(
        self,
        transaction: TenantConnectionAuthorizationTransaction,
    ) -> None:
        self._delete_verifier(transaction)
        if transaction.credential_staging_ref:
            self._delete_secret(transaction.credential_staging_ref)
        self._terminalize_transaction(
            transaction,
            TenantConnectionAuthorizationCompletionState.EXCHANGE_OUTCOME_UNKNOWN,
        )

    def _reload_transaction(
        self,
        authorization_transaction_ref: str,
    ) -> TenantConnectionAuthorizationTransaction | None:
        return self._transaction_repository.get(
            tenant_id=self._tenant_id,
            authorization_transaction_ref=authorization_transaction_ref,
        )

    def _read_verifier(
        self,
        transaction: TenantConnectionAuthorizationTransaction,
    ) -> str | None:
        ref = transaction.verifier_secret_ref
        if ref is None:
            return None
        return self._read_secret(ref)

    def _delete_verifier(
        self,
        transaction: TenantConnectionAuthorizationTransaction,
    ) -> None:
        ref = transaction.verifier_secret_ref
        if ref is None:
            return
        self._delete_secret(ref)

    def _cleanup_transaction_secrets(
        self,
        transaction: TenantConnectionAuthorizationTransaction,
    ) -> None:
        self._delete_verifier(transaction)
        if transaction.credential_staging_ref:
            self._delete_secret(transaction.credential_staging_ref)

    def _read_secret(self, path: str) -> str | None:
        if not path or not path.strip():
            return None
        try:
            value = self._secrets_store.get_secret(path.strip())
        except Exception:
            return None
        if not isinstance(value, str) or not value.strip():
            return None
        return value

    def _delete_secret(self, path: str) -> None:
        if not path or not path.strip():
            return
        try:
            self._secrets_store.delete_secret(path.strip())
        except Exception:
            return


__all__ = [
    "ConnectionAuthBeginResult",
    "ConnectionAuthCompleteResult",
    "TenantConnectionProductOrchestrationConfig",
    "TenantConnectionProductOrchestrationFactory",
    "TenantConnectionProductOrchestrationService",
]


class TenantConnectionProductOrchestrationFactory:
    """Build tenant-scoped orchestration services and handle public OAuth callbacks."""

    def __init__(
        self,
        *,
        connection_repository: DocumentStoreTenantConnectionRepository,
        transaction_repository: TenantConnectionAuthorizationTransactionRepository,
        secrets_store: SecretsStore,
        auth_provider_registry: TenantConnectionAuthProviderRegistry,
        rehydrator: TenantConnectionRehydrator,
        connection_registry: KnowledgeConnectionRegistry,
        config: TenantConnectionProductOrchestrationConfig,
    ) -> None:
        self._connection_repository = connection_repository
        self._transaction_repository = transaction_repository
        self._secrets_store = secrets_store
        self._auth_providers = auth_provider_registry
        self._rehydrator = rehydrator
        self._connection_registry = connection_registry
        self._config = config

    def for_tenant(self, tenant_id: str) -> TenantConnectionProductOrchestrationService:
        return TenantConnectionProductOrchestrationService(
            tenant_id=tenant_id,
            connection_repository=self._connection_repository,
            transaction_repository=self._transaction_repository,
            secrets_store=self._secrets_store,
            auth_provider_registry=self._auth_providers,
            rehydrator=self._rehydrator,
            connection_registry=self._connection_registry,
            config=self._config,
        )

    def complete_oauth_callback(
        self,
        *,
        provider_id: str,
        authorization_code: str,
        state: str,
    ) -> ConnectionAuthCompleteResult:
        tenant_id, transaction_ref, _ = parse_state_parameter(state)
        transaction = self._transaction_repository.get(
            tenant_id=tenant_id,
            authorization_transaction_ref=transaction_ref,
        )
        if transaction is None:
            raise tenant_connection_product_error("authorization_transaction_not_found")
        if transaction.provider_id != provider_id.strip():
            raise tenant_connection_product_error("authorization_state_invalid")
        service = self.for_tenant(tenant_id)
        return service.complete_connection_authorization(
            authorization_transaction_ref=transaction_ref,
            authorization_code=authorization_code,
            state=state,
        )
