# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed runtime inspection read errors (INSPECT-01-A)."""

from __future__ import annotations

from enum import StrEnum

from intergrax.contracts.execution_identity import ExecutionId


class RuntimeInspectionErrorCode(StrEnum):
    NOT_FOUND = "not_found"
    TENANT_BOUNDARY = "tenant_boundary"
    EXECUTION_FACTS_UNAVAILABLE = "execution_facts_unavailable"
    SOURCE_INTEGRITY = "source_integrity"
    CONFIGURATION = "configuration"


class RuntimeInspectionError(Exception):
    """Base typed inspection read failure — no secret-bearing exception strings as ABI."""

    def __init__(
        self,
        code: RuntimeInspectionErrorCode,
        message: str,
        *,
        execution_id: ExecutionId | None = None,
        source_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.execution_id = execution_id
        self.source_id = source_id


class RuntimeInspectionNotFoundError(RuntimeInspectionError):
    def __init__(self, *, execution_id: ExecutionId) -> None:
        super().__init__(
            RuntimeInspectionErrorCode.NOT_FOUND,
            "execution inspection scope not found",
            execution_id=execution_id,
        )


class RuntimeInspectionTenantBoundaryError(RuntimeInspectionError):
    def __init__(self, *, execution_id: ExecutionId) -> None:
        super().__init__(
            RuntimeInspectionErrorCode.TENANT_BOUNDARY,
            "execution inspection denied for tenant scope",
            execution_id=execution_id,
        )


__all__ = [
    "RuntimeInspectionError",
    "RuntimeInspectionErrorCode",
    "RuntimeInspectionNotFoundError",
    "RuntimeInspectionTenantBoundaryError",
]
