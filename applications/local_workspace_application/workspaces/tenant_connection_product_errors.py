# © Artur Czarnecki. All rights reserved.

"""Stable product errors for tenant connection orchestration (PRODUCT-5B)."""

from __future__ import annotations


class TenantConnectionProductError(RuntimeError):
    def __init__(self, error_code: str, *, retryable: bool = False) -> None:
        self.error_code = error_code
        self.retryable = retryable
        super().__init__(error_code)


_RETRYABLE: frozenset[str] = frozenset(
    {
        "authorization_transaction_expired",
        "authorization_exchange_outcome_unknown",
        "credential_binding_invalid",
        "connection_version_conflict",
        "connection_runtime_unavailable",
    }
)


def tenant_connection_product_error(error_code: str) -> TenantConnectionProductError:
    return TenantConnectionProductError(
        error_code,
        retryable=error_code in _RETRYABLE,
    )


__all__ = ["TenantConnectionProductError", "tenant_connection_product_error"]
