"""Strict typed Parquet scalar extraction for Data Pack bootstrap reading."""

from __future__ import annotations

import math
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.errors import (
    DataPackReaderSchemaError,
)


def _value_category(value: object) -> str:
    if value is None:
        return "null"
    return type(value).__name__


def _schema_error(
    *,
    shard_path: Path,
    row_index: int,
    column: str,
    expected: str,
    value: object,
) -> DataPackReaderSchemaError:
    return DataPackReaderSchemaError(
        f"{shard_path}: row {row_index} column {column}: expected {expected}, "
        f"got {_value_category(value)}"
    )


def require_string(
    value: object,
    *,
    shard_path: Path,
    row_index: int,
    column: str,
) -> str:
    if not isinstance(value, str):
        raise _schema_error(
            shard_path=shard_path,
            row_index=row_index,
            column=column,
            expected="str",
            value=value,
        )
    return value


def require_optional_string(
    value: object,
    *,
    shard_path: Path,
    row_index: int,
    column: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _schema_error(
            shard_path=shard_path,
            row_index=row_index,
            column=column,
            expected="str or null",
            value=value,
        )
    return value


def require_int(
    value: object,
    *,
    shard_path: Path,
    row_index: int,
    column: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _schema_error(
            shard_path=shard_path,
            row_index=row_index,
            column=column,
            expected="int",
            value=value,
        )
    if value < minimum:
        raise _schema_error(
            shard_path=shard_path,
            row_index=row_index,
            column=column,
            expected=f"int >= {minimum}",
            value=value,
        )
    return value


def require_bool(
    value: object,
    *,
    shard_path: Path,
    row_index: int,
    column: str,
) -> bool:
    if not isinstance(value, bool):
        raise _schema_error(
            shard_path=shard_path,
            row_index=row_index,
            column=column,
            expected="bool",
            value=value,
        )
    return value


def require_float_vector(
    value: object,
    *,
    shard_path: Path,
    row_index: int,
    column: str,
    expected_length: int,
) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise _schema_error(
            shard_path=shard_path,
            row_index=row_index,
            column=column,
            expected="list",
            value=value,
        )
    if len(value) != expected_length:
        raise _schema_error(
            shard_path=shard_path,
            row_index=row_index,
            column=column,
            expected=f"list length {expected_length}",
            value=f"length {len(value)}",
        )
    elements: list[float] = []
    for element_index, element in enumerate(value):
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            raise DataPackReaderSchemaError(
                f"{shard_path}: row {row_index} column {column}[{element_index}]: "
                f"expected numeric, got {_value_category(element)}"
            )
        numeric = float(element)
        if not math.isfinite(numeric):
            raise DataPackReaderSchemaError(
                f"{shard_path}: row {row_index} column {column}[{element_index}]: "
                f"expected finite numeric, got {numeric}"
            )
        elements.append(numeric)
    return tuple(elements)
