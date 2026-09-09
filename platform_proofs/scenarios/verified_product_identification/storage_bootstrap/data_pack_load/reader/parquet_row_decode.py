"""Strict typed Parquet scalar extraction for Data Pack bootstrap reading."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypeAlias

import pyarrow as pa

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.errors import (
    DataPackReaderSchemaError,
)

ParquetNumericScalar: TypeAlias = int | float
ParquetStringScalar: TypeAlias = str | None
ParquetIntegerScalar: TypeAlias = int | None
ParquetBooleanScalar: TypeAlias = bool | None
ParquetVectorScalar: TypeAlias = list[ParquetNumericScalar] | None
ParquetScalar: TypeAlias = (
    str | int | float | bool | list[ParquetNumericScalar] | None
)


def _value_category(value: ParquetScalar) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def _schema_error(
    *,
    shard_path: Path,
    row_index: int,
    column: str,
    expected: str,
    value: ParquetScalar,
    got: str | None = None,
) -> DataPackReaderSchemaError:
    observed = got if got is not None else _value_category(value)
    return DataPackReaderSchemaError(
        f"{shard_path}: row {row_index} column {column}: expected {expected}, "
        f"got {observed}"
    )


def extract_parquet_scalar(
    cell: pa.Scalar,
    *,
    shard_path: Path,
    row_index: int,
    column: str,
) -> ParquetScalar:
    """Isolate PyArrow ``as_py()`` and return only canonical scalar shapes."""
    raw = cell.as_py()
    match raw:
        case None:
            return None
        case bool() as value:
            return value
        case str() as value:
            return value
        case int() as value:
            return value
        case float() as value:
            return value
        case list(elements):
            numeric_elements: list[ParquetNumericScalar] = []
            for element_index, element in enumerate(elements):
                if isinstance(element, bool) or not isinstance(element, (int, float)):
                    raise DataPackReaderSchemaError(
                        f"{shard_path}: row {row_index} column {column}[{element_index}]: "
                        f"expected numeric, got {type(element).__name__}"
                    )
                numeric_elements.append(element)
            return numeric_elements
        case _:
            raise _schema_error(
                shard_path=shard_path,
                row_index=row_index,
                column=column,
                expected="canonical Parquet scalar",
                value=None,
                got=type(raw).__name__,
            )


def require_string(
    value: ParquetStringScalar,
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
    value: ParquetStringScalar,
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
    value: ParquetIntegerScalar,
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
    value: ParquetBooleanScalar,
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
    value: ParquetVectorScalar,
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
            value=value,
            got=f"length {len(value)}",
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
