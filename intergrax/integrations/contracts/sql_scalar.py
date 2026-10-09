# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Closed SQL scalar domain for configured relational execution (TRACE-X-P5-R2-P3)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

SqlScalar: TypeAlias = str | int | float | bool | bytes | None


class SqlScalarBoundaryError(ValueError):
    """Provider returned a value outside the closed SqlScalar domain."""


def sql_scalar_tuple_from_sequence(values: Sequence[object]) -> tuple[SqlScalar, ...]:
    return tuple(validate_sql_scalar(value) for value in values)


def sql_scalar_row_from_mapping(row: Mapping[str, object]) -> dict[str, SqlScalar]:
    return {key: validate_sql_scalar(value) for key, value in row.items()}


def validate_sql_scalar(value: object) -> SqlScalar:
    if value is None:
        return None
    if type(value) is str:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        return value
    if type(value) is bool:
        return value
    if type(value) is bytes:
        return value
    raise SqlScalarBoundaryError(
        f"unsupported sql scalar runtime type: {type(value).__name__}",
    )


__all__ = [
    "SqlScalar",
    "SqlScalarBoundaryError",
    "sql_scalar_row_from_mapping",
    "sql_scalar_tuple_from_sequence",
    "validate_sql_scalar",
]
