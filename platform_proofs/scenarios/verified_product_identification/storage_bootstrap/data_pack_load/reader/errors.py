"""Narrow scenario-owned errors for Data Pack bootstrap reading."""

from __future__ import annotations


class DataPackReaderError(Exception):
    """Base error for Data Pack bootstrap reader failures."""


class DataPackReaderIntegrityError(DataPackReaderError):
    """Shard metadata, pairing, or record-count integrity failure."""


class DataPackReaderSchemaError(DataPackReaderError):
    """Parquet schema or column type contract violation."""


class DataPackReaderOrderingError(DataPackReaderError):
    """Canonical global_row_index ordering violation."""
