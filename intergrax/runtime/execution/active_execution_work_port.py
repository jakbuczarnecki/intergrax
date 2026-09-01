# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-scoped active work submission binding (DS-NEXUS-01)."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class ActiveExecutionWorkPortBinding(Generic[InputT, OutputT, ResultT]):
    """Typed execution-scoped access token for one work port capability."""

    port: ExecutionWorkPort[InputT, OutputT, ResultT]

    @classmethod
    def for_port(
        cls,
        port: ExecutionWorkPort[InputT, OutputT, ResultT],
    ) -> ActiveExecutionWorkPortBinding[InputT, OutputT, ResultT]:
        """Anchor work port types before execution-scoped access."""
        return cls(port)

    def get_active(self) -> ExecutionWorkPort[InputT, OutputT, ResultT] | None:
        binding = _active_execution_work_port.get()
        if binding is None:
            return None
        if binding.port is not self.port:
            return None
        return self.port

    def require_active(self) -> ExecutionWorkPort[InputT, OutputT, ResultT]:
        binding = _active_execution_work_port.get()
        if binding is None:
            raise RuntimeError("active execution work port required")
        if binding.port is not self.port:
            raise RuntimeError(
                "active execution work port does not match this binding",
            )
        return self.port


_active_execution_work_port: ContextVar[
    ActiveExecutionWorkPortBinding | None
] = ContextVar(
    "active_execution_work_port",
    default=None,
)


def bind_active_execution_work_port(
    binding: ActiveExecutionWorkPortBinding[InputT, OutputT, ResultT],
) -> Token:
    return _active_execution_work_port.set(binding)


def reset_active_execution_work_port(token: Token) -> None:
    _active_execution_work_port.reset(token)


def is_execution_work_port_active() -> bool:
    return _active_execution_work_port.get() is not None
