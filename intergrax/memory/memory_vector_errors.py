# © Artur Czarnecki. All rights reserved.

"""Memory vector wiring errors (Phase MEM-VEC-1.4)."""


class MemoryVectorBackendUnavailableError(RuntimeError):
    """Raised when memory vector flags are enabled but no vector backend is wired."""

    def __init__(self, *, reason: str = "vector_backend_unavailable") -> None:
        super().__init__(reason)
        self.reason = reason


class MemoryTenantScopeViolationError(RuntimeError):
    """Raised when a caller attempts to override bound tenant vector scope."""

    def __init__(self, *, expected_tenant_id: str, requested_tenant_id: str) -> None:
        super().__init__(
            f"tenant scope violation: bound={expected_tenant_id}, requested={requested_tenant_id}"
        )
        self.expected_tenant_id = expected_tenant_id
        self.requested_tenant_id = requested_tenant_id
