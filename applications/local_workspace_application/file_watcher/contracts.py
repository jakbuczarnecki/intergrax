# © Artur Czarnecki. All rights reserved.

"""Filesystem snapshot and incremental change contracts (LKW.7A).

Version identity is metadata-based only:
canonical path + size_bytes + modified_time_ns.
This is not cryptographic content-change detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FileChangeKind = Literal["created", "modified", "deleted"]


def normalize_watch_path_key(path: str) -> str:
    """Deterministic sort/dedupe key for canonical filesystem paths."""
    return str(Path(path))


class FileSnapshot(BaseModel):
    """Allowlisted regular-file metadata snapshot (no content)."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["lkw.file_snapshot.v1"] = "lkw.file_snapshot.v1"
    path: str
    size_bytes: int = Field(ge=0)
    modified_time_ns: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("path must be non-blank")
        candidate = Path(stripped)
        if not candidate.is_absolute():
            raise ValueError("path must be absolute")
        return str(candidate)


class FileChange(BaseModel):
    """Single-path filesystem change derived from metadata snapshots."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["lkw.file_change.v1"] = "lkw.file_change.v1"
    kind: FileChangeKind
    path: str
    previous: FileSnapshot | None = None
    current: FileSnapshot | None = None

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("path must be non-blank")
        candidate = Path(stripped)
        if not candidate.is_absolute():
            raise ValueError("path must be absolute")
        return str(candidate)

    @model_validator(mode="after")
    def _validate_kind_invariants(self) -> FileChange:
        if self.kind == "created":
            if self.previous is not None:
                raise ValueError("created change must not include previous snapshot")
            if self.current is None:
                raise ValueError("created change requires current snapshot")
            if self.path != self.current.path:
                raise ValueError("created change path must equal current.path")
            return self
        if self.kind == "modified":
            if self.previous is None or self.current is None:
                raise ValueError(
                    "modified change requires previous and current snapshots"
                )
            if not (self.previous.path == self.current.path == self.path):
                raise ValueError("modified change paths must agree")
            if (
                self.previous.size_bytes == self.current.size_bytes
                and self.previous.modified_time_ns == self.current.modified_time_ns
            ):
                raise ValueError(
                    "modified change requires size_bytes or modified_time_ns difference"
                )
            return self
        if self.kind == "deleted":
            if self.current is not None:
                raise ValueError("deleted change must not include current snapshot")
            if self.previous is None:
                raise ValueError("deleted change requires previous snapshot")
            if self.path != self.previous.path:
                raise ValueError("deleted change path must equal previous.path")
            return self
        raise ValueError(f"unsupported change kind: {self.kind}")


class IncrementalFileChangeBatch(BaseModel):
    """Coalesced actionable snapshots plus diagnostic deletions (LKW.7A)."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["lkw.file_change_batch.v1"] = "lkw.file_change_batch.v1"
    source_snapshots: tuple[FileSnapshot, ...] = ()
    deleted_paths: tuple[str, ...] = ()
    change_token: str | None = None

    @property
    def source_paths(self) -> tuple[str, ...]:
        return tuple(snapshot.path for snapshot in self.source_snapshots)

    @model_validator(mode="after")
    def _validate_batch_invariants(self) -> IncrementalFileChangeBatch:
        source_paths = [snapshot.path for snapshot in self.source_snapshots]
        source_keys = [normalize_watch_path_key(path) for path in source_paths]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("source_snapshots must contain unique paths")
        if source_keys != sorted(source_keys):
            raise ValueError("source_snapshots must be sorted by normalized path")

        deleted_keys = [normalize_watch_path_key(path) for path in self.deleted_paths]
        if len(deleted_keys) != len(set(deleted_keys)):
            raise ValueError("deleted_paths must contain unique paths")
        if deleted_keys != sorted(deleted_keys):
            raise ValueError("deleted_paths must be sorted by normalized path")
        for path in self.deleted_paths:
            candidate = Path(path)
            if not path.strip() or not candidate.is_absolute():
                raise ValueError("deleted_paths must be absolute non-blank paths")

        overlap = set(source_keys) & set(deleted_keys)
        if overlap:
            raise ValueError(
                "a path cannot exist in both source_snapshots and deleted_paths"
            )

        if self.source_snapshots and self.change_token is None:
            raise ValueError(
                "change_token is required when source_snapshots is non-empty"
            )
        if not self.source_snapshots and self.change_token is not None:
            raise ValueError("change_token must be None when source_snapshots is empty")
        return self
