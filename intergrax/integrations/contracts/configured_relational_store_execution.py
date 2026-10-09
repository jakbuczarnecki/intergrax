# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integrations-owned configured relational store execution contract (TRACE-X-P5-R2-P3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.sql_scalar import SqlScalar


@dataclass(frozen=True, slots=True)
class RelationalQueryRequest:
    sql: str
    params: tuple[SqlScalar, ...] = ()


@dataclass(frozen=True, slots=True)
class RelationalQueryResult:
    rows: tuple[dict[str, SqlScalar], ...]
    row_count: int


@dataclass(frozen=True, slots=True)
class RelationalExecuteRequest:
    sql: str
    params: tuple[SqlScalar, ...] = ()


@dataclass(frozen=True, slots=True)
class RelationalExecuteResult:
    executed: bool = True


@runtime_checkable
class ConfiguredRelationalStoreExecutionPort(Protocol):
    """Category-specific relational operations — no universal operation dispatcher."""

    def query(self, request: RelationalQueryRequest) -> RelationalQueryResult: ...

    def execute(self, request: RelationalExecuteRequest) -> RelationalExecuteResult: ...


__all__ = [
    "ConfiguredRelationalStoreExecutionPort",
    "RelationalExecuteRequest",
    "RelationalExecuteResult",
    "RelationalQueryRequest",
    "RelationalQueryResult",
]
