# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-scoped active work submission binding (DS-NEXUS-01)."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TypeVar

from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ResultT = TypeVar("ResultT")

_active_execution_work_port: ContextVar[ExecutionWorkPort | None] = ContextVar(
    "active_execution_work_port",
    default=None,
)


def bind_active_execution_work_port(
    port: ExecutionWorkPort[InputT, OutputT, ResultT],
) -> Token:
    return _active_execution_work_port.set(port)


def reset_active_execution_work_port(token: Token) -> None:
    _active_execution_work_port.reset(token)


def get_active_execution_work_port() -> ExecutionWorkPort | None:
    return _active_execution_work_port.get()


def require_active_execution_work_port() -> ExecutionWorkPort:
    port = get_active_execution_work_port()
    if port is None:
        raise RuntimeError("active execution work port required")
    return port
