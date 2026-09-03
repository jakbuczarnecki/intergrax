"""Deterministic artifact shard path helpers."""

from __future__ import annotations

from pathlib import Path

MANIFEST_FILE_NAME = "manifest.json"
SHARD_FILE_PREFIX = "part-"
SHARD_FILE_SUFFIX = ".parquet"
TEMP_SHARD_SUFFIX = ".parquet.tmp"


def shard_file_name(shard_ordinal: int) -> str:
    return f"{SHARD_FILE_PREFIX}{shard_ordinal:06d}{SHARD_FILE_SUFFIX}"


def temp_shard_file_name(shard_ordinal: int) -> str:
    return f"{shard_file_name(shard_ordinal)}{TEMP_SHARD_SUFFIX}"


def shard_path(artifact_dir: Path, shard_ordinal: int) -> Path:
    return artifact_dir / shard_file_name(shard_ordinal)


def temp_shard_path(artifact_dir: Path, shard_ordinal: int) -> Path:
    return artifact_dir / temp_shard_file_name(shard_ordinal)


def manifest_path(artifact_dir: Path) -> Path:
    return artifact_dir / MANIFEST_FILE_NAME
