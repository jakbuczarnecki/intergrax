# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed external operation failure taxonomy (DIAG integration R2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

SCHEMA_EXTERNAL_OPERATION_FAILURE_KIND_V1: Final = "external_operation_failure_kind.v1"


class ExternalOperationFailureKind(StrEnum):
    """Namespaced failure classes — never raw exception types."""

    TIMEOUT = "external_operation.timeout"
    AUTHENTICATION_FAILED = "external_operation.authentication_failed"
    AUTHORIZATION_FAILED = "external_operation.authorization_failed"
    RATE_LIMITED = "external_operation.rate_limited"
    INVALID_REQUEST = "external_operation.invalid_request"
    PROVIDER_UNAVAILABLE = "external_operation.provider_unavailable"
    REMOTE_FAILURE = "external_operation.remote_failure"
    UNKNOWN = "external_operation.unknown"


__all__ = ["ExternalOperationFailureKind", "SCHEMA_EXTERNAL_OPERATION_FAILURE_KIND_V1"]
