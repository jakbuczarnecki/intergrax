# © Artur Czarnecki. All rights reserved.

"""Memory vector wiring errors (Phase MEM-VEC-1.4)."""


class MemoryVectorBackendUnavailableError(RuntimeError):
    """Raised when memory vector flags are enabled but no vector backend is wired."""

    def __init__(self, *, reason: str = "vector_backend_unavailable") -> None:
        super().__init__(reason)
        self.reason = reason
