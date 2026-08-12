# © Artur Czarnecki. All rights reserved.

"""Durable authorization transaction metadata for tenant connection auth (PRODUCT-5B)."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)

_PARTITION_PREFIX = "vendor_knowledge_auth_transactions"
_ROW_PREFIX = "auth_transaction"


class TenantConnectionAuthorizationCompletionState(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    EXCHANGING = "exchanging"
    CREDENTIALS_OBTAINED = "credentials_obtained"
    COMPLETED = "completed"
    EXCHANGE_OUTCOME_UNKNOWN = "exchange_outcome_unknown"
    EXPIRED = "expired"


class TenantConnectionAuthorizationTransactionError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


class TenantConnectionAuthorizationTransaction(BaseModel):
    """Metadata-only durable OAuth/manual authorization session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    authorization_transaction_ref: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    provider_id: str = Field(min_length=1, max_length=64)
    correlation_state: str = Field(min_length=1, max_length=256)
    redirect_uri: str = Field(min_length=1, max_length=512)
    connection_ref: str | None = Field(default=None, max_length=128)
    verifier_secret_ref: str | None = Field(default=None, max_length=512)
    credential_staging_ref: str | None = Field(default=None, max_length=512)
    created_at: datetime
    expires_at: datetime
    completion_state: TenantConnectionAuthorizationCompletionState
    completion_claim_expires_at: datetime | None = None
    exchange_started_at: datetime | None = None
    consumed_at: datetime | None = None
    version: int = Field(ge=1)

    @field_validator(
        "created_at",
        "expires_at",
        "completion_claim_expires_at",
        "exchange_started_at",
        "consumed_at",
    )
    @classmethod
    def _utc_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware UTC")
        if value.utcoffset() != timedelta(0):
            raise ValueError("timestamp must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def _validate_state_invariants(self) -> TenantConnectionAuthorizationTransaction:
        if self.expires_at < self.created_at:
            raise ValueError("expires_at must be greater than or equal to created_at")
        if self.completion_state is TenantConnectionAuthorizationCompletionState.CLAIMED:
            if self.completion_claim_expires_at is None:
                raise ValueError("claimed transaction requires completion_claim_expires_at")
        if self.completion_state is TenantConnectionAuthorizationCompletionState.COMPLETED:
            if self.consumed_at is None:
                raise ValueError("completed transaction requires consumed_at")
        return self


def auth_transaction_partition_key(tenant_id: str) -> str:
    cleaned = tenant_id.strip()
    if not cleaned:
        raise ValueError("tenant_id must be a non-empty string")
    return f"{_PARTITION_PREFIX}:{cleaned}"


def auth_transaction_row_key(authorization_transaction_ref: str) -> str:
    cleaned = authorization_transaction_ref.strip()
    if not cleaned:
        raise ValueError("authorization_transaction_ref must be a non-empty string")
    return f"{_ROW_PREFIX}:{cleaned}"


def build_state_parameter(
    *,
    tenant_id: str,
    authorization_transaction_ref: str,
    correlation_state: str,
) -> str:
    return ":".join(
        (
            tenant_id.strip(),
            authorization_transaction_ref.strip(),
            correlation_state.strip(),
        )
    )


def parse_state_parameter(state: str) -> tuple[str, str, str]:
    cleaned = state.strip()
    parts = cleaned.split(":", 2)
    if len(parts) != 3:
        raise ValueError("authorization state format is invalid")
    tenant_id, ref, nonce = parts
    if not tenant_id.strip() or not ref.strip() or not nonce.strip():
        raise ValueError("authorization state format is invalid")
    return tenant_id.strip(), ref.strip(), nonce.strip()


def verifier_secret_path(
    *,
    tenant_id: str,
    authorization_transaction_ref: str,
) -> str:
    return (
        f"secrets/{tenant_id.strip()}/auth-transactions/"
        f"{authorization_transaction_ref.strip()}/pkce-verifier"
    )


def credential_staging_secret_path(
    *,
    tenant_id: str,
    authorization_transaction_ref: str,
) -> str:
    return (
        f"secrets/{tenant_id.strip()}/auth-transactions/"
        f"{authorization_transaction_ref.strip()}/credential-staging"
    )


def connection_credential_secret_path(
    *,
    tenant_id: str,
    connection_ref: str,
) -> str:
    return f"secrets/{tenant_id.strip()}/connections/{connection_ref.strip()}"


def _transaction_to_document(
    transaction: TenantConnectionAuthorizationTransaction,
) -> DocumentRecord:
    return DocumentRecord(
        partition_key=auth_transaction_partition_key(transaction.tenant_id),
        row_key=auth_transaction_row_key(transaction.authorization_transaction_ref),
        data=transaction.model_dump(mode="json"),
        ttl_seconds=None,
    )


def _transaction_from_document(document: DocumentRecord) -> TenantConnectionAuthorizationTransaction:
    try:
        return TenantConnectionAuthorizationTransaction.model_validate(dict(document.data))
    except Exception as exc:
        raise TenantConnectionAuthorizationTransactionError(
            "authorization_transaction_malformed"
        ) from exc


class TenantConnectionAuthorizationTransactionRepository:
    """CAS-backed durable store for authorization transaction metadata."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store

    def _conditional(self) -> ConditionalDocumentStore:
        if not isinstance(self._store, ConditionalDocumentStore):
            raise TenantConnectionAuthorizationTransactionError(
                "authorization_transaction_conditional_store_required"
            )
        return self._store

    def create(self, transaction: TenantConnectionAuthorizationTransaction) -> None:
        if not self._conditional().put_if_absent(_transaction_to_document(transaction)):
            raise TenantConnectionAuthorizationTransactionError(
                "authorization_transaction_already_exists"
            )

    def get(
        self,
        *,
        tenant_id: str,
        authorization_transaction_ref: str,
    ) -> TenantConnectionAuthorizationTransaction | None:
        document = self._store.get(
            auth_transaction_partition_key(tenant_id),
            auth_transaction_row_key(authorization_transaction_ref),
        )
        if document is None:
            return None
        transaction = _transaction_from_document(document)
        if transaction.tenant_id != tenant_id.strip():
            raise TenantConnectionAuthorizationTransactionError("tenant_mismatch")
        return transaction

    def replace_if_match(
        self,
        current: TenantConnectionAuthorizationTransaction,
        replacement: TenantConnectionAuthorizationTransaction,
    ) -> bool:
        if current.authorization_transaction_ref != replacement.authorization_transaction_ref:
            raise ValueError("authorization transaction identity mismatch")
        if current.tenant_id != replacement.tenant_id:
            raise ValueError("authorization transaction tenant mismatch")
        return self._conditional().replace_if_match(
            expected=_transaction_to_document(current),
            replacement=_transaction_to_document(replacement),
        )

    def list_expired(
        self,
        *,
        tenant_id: str,
        now: datetime,
        limit: int = 100,
    ) -> tuple[TenantConnectionAuthorizationTransaction, ...]:
        partition = auth_transaction_partition_key(tenant_id)
        prefix = f"{_ROW_PREFIX}:"
        documents: list[DocumentRecord] = []
        cursor: str | None = None
        while len(documents) < limit:
            page = (
                self._store.query(partition, limit=limit, row_key_prefix=prefix, cursor=cursor)
                if cursor
                else self._store.query(partition, limit=limit, row_key_prefix=prefix)
            )
            documents.extend(page.documents)
            next_cursor = getattr(page, "next_cursor", None)
            if next_cursor is None:
                break
            cursor = next_cursor
        results: list[TenantConnectionAuthorizationTransaction] = []
        for document in documents:
            transaction = _transaction_from_document(document)
            if transaction.expires_at <= now:
                results.append(transaction)
            if len(results) >= limit:
                break
        return tuple(results)




__all__ = [
    "TenantConnectionAuthorizationCompletionState",
    "TenantConnectionAuthorizationTransaction",
    "TenantConnectionAuthorizationTransactionError",
    "TenantConnectionAuthorizationTransactionRepository",
    "auth_transaction_partition_key",
    "auth_transaction_row_key",
    "build_state_parameter",
    "connection_credential_secret_path",
    "credential_staging_secret_path",
    "parse_state_parameter",
    "verifier_secret_path",
]
