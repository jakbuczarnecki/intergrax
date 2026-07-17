# © Artur Czarnecki. All rights reserved.

"""Durable file-watcher checkpoint contracts and JSON store (LKW.7B2A).

Persists baseline snapshots and final pending FileChange values only.
No process loop, sleep, signals, or automatic save lifecycle lives here.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

from local_workspace_application.file_watcher.contracts import (
    FileChange,
    FileSnapshot,
    normalize_watch_path_key,
)

if TYPE_CHECKING:
    from local_workspace_application.file_watcher.runtime import FileWatcherRuntime

LKW_FILE_WATCHER_CHECKPOINT_SCHEMA_VERSION = "lkw.file_watcher_checkpoint.v1"


def _canonical_checkpoint_root(path: str) -> str:
    """Return a resolved absolute root path without requiring it to exist."""
    stripped = path.strip()
    if not stripped:
        raise ValueError("allowed root must be non-blank")
    candidate = Path(stripped).expanduser()
    if not candidate.is_absolute():
        raise ValueError("allowed root must be absolute")
    try:
        return str(candidate.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        raise ValueError("allowed root could not be canonicalized") from None


class FileWatcherCheckpoint(BaseModel):
    """Versioned durable watcher baseline + pending final changes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["lkw.file_watcher_checkpoint.v1"] = (
        LKW_FILE_WATCHER_CHECKPOINT_SCHEMA_VERSION
    )

    tenant_id: str
    workspace_id: str
    collection_id: str

    allowed_roots: tuple[str, ...]

    baseline_snapshots: tuple[FileSnapshot, ...] = ()
    pending_changes: tuple[FileChange, ...] = ()

    @field_validator("tenant_id", "workspace_id", "collection_id")
    @classmethod
    def _require_stripped_non_blank(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-blank string")
        return value.strip()

    @field_validator("allowed_roots", mode="before")
    @classmethod
    def _canonicalize_allowed_roots(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("allowed_roots must be a list or tuple")
        return tuple(_canonical_checkpoint_root(str(item)) for item in value)

    @model_validator(mode="after")
    def _validate_checkpoint_invariants(self) -> FileWatcherCheckpoint:
        root_keys = [normalize_watch_path_key(root) for root in self.allowed_roots]
        if not root_keys:
            raise ValueError("allowed_roots must be non-empty")
        if len(root_keys) != len(set(root_keys)):
            raise ValueError("allowed_roots must contain unique paths")
        if root_keys != sorted(root_keys):
            raise ValueError("allowed_roots must be sorted by normalized path")

        baseline_keys = [
            normalize_watch_path_key(snapshot.path)
            for snapshot in self.baseline_snapshots
        ]
        if len(baseline_keys) != len(set(baseline_keys)):
            raise ValueError("baseline_snapshots must contain unique paths")
        if baseline_keys != sorted(baseline_keys):
            raise ValueError("baseline_snapshots must be sorted by normalized path")

        pending_keys = [
            normalize_watch_path_key(change.path) for change in self.pending_changes
        ]
        if len(pending_keys) != len(set(pending_keys)):
            raise ValueError("pending_changes must contain unique paths")
        if pending_keys != sorted(pending_keys):
            raise ValueError("pending_changes must be sorted by normalized path")

        baseline_by_key = {
            normalize_watch_path_key(snapshot.path): snapshot
            for snapshot in self.baseline_snapshots
        }
        pending_by_key = {
            normalize_watch_path_key(change.path): change
            for change in self.pending_changes
        }

        for key, change in pending_by_key.items():
            if change.kind in ("created", "modified"):
                if change.current is None:
                    raise ValueError(
                        "pending created/modified change requires current snapshot"
                    )
                baseline_snapshot = baseline_by_key.get(key)
                if baseline_snapshot is None:
                    raise ValueError(
                        "pending created/modified path must exist in baseline"
                    )
                if baseline_snapshot != change.current:
                    raise ValueError(
                        "pending created/modified current must match baseline snapshot"
                    )
            elif change.kind == "deleted":
                if key in baseline_by_key:
                    raise ValueError(
                        "pending deleted path must be absent from baseline"
                    )
            else:
                raise ValueError(f"unsupported pending change kind: {change.kind}")

        return self


def build_file_watcher_checkpoint(
    *,
    tenant_id: str,
    workspace_id: str,
    collection_id: str,
    allowed_roots: frozenset[str],
    baseline_snapshots: tuple[FileSnapshot, ...],
    pending_changes: tuple[FileChange, ...],
) -> FileWatcherCheckpoint:
    """Build a validated checkpoint with deterministic ordering."""
    roots_by_key: dict[str, str] = {}
    for root in allowed_roots:
        canonical = _canonical_checkpoint_root(str(root))
        roots_by_key[normalize_watch_path_key(canonical)] = canonical
    ordered_roots = tuple(roots_by_key[key] for key in sorted(roots_by_key.keys()))

    baseline_by_key: dict[str, FileSnapshot] = {}
    for snapshot in baseline_snapshots:
        baseline_by_key[normalize_watch_path_key(snapshot.path)] = snapshot
    ordered_baseline = tuple(
        baseline_by_key[key] for key in sorted(baseline_by_key.keys())
    )

    pending_by_key: dict[str, FileChange] = {}
    for change in pending_changes:
        pending_by_key[normalize_watch_path_key(change.path)] = change
    ordered_pending = tuple(
        pending_by_key[key] for key in sorted(pending_by_key.keys())
    )

    return FileWatcherCheckpoint(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        collection_id=collection_id,
        allowed_roots=ordered_roots,
        baseline_snapshots=ordered_baseline,
        pending_changes=ordered_pending,
    )


def encode_file_watcher_checkpoint(checkpoint: FileWatcherCheckpoint) -> bytes:
    """Encode a checkpoint as deterministic UTF-8 JSON with one trailing newline."""
    payload = json.dumps(
        checkpoint.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (payload + "\n").encode("utf-8")


def decode_file_watcher_checkpoint(payload: bytes) -> FileWatcherCheckpoint:
    """Decode and validate checkpoint bytes; fail closed on any defect."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise RuntimeError("checkpoint_invalid_encoding") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError("checkpoint_invalid_json") from None
    if not isinstance(data, dict):
        raise RuntimeError("checkpoint_invalid")
    try:
        return FileWatcherCheckpoint.model_validate(data)
    except ValidationError:
        raise RuntimeError("checkpoint_invalid") from None


def file_watcher_checkpoint_path(data_home: str | Path) -> Path:
    """Resolve ``<data_home>/data/file_watcher/checkpoint.json``."""
    if isinstance(data_home, Path):
        raw = str(data_home)
    else:
        raw = data_home
    stripped = raw.strip()
    if not stripped:
        raise RuntimeError("data_home_must_be_non_blank")
    candidate = Path(stripped).expanduser()
    if not candidate.is_absolute():
        raise RuntimeError("data_home_must_be_absolute")
    return candidate.resolve(strict=False) / "data" / "file_watcher" / "checkpoint.json"


class JsonFileWatcherCheckpointStore:
    """Atomic JSON checkpoint persistence under an absolute path."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        candidate = Path(checkpoint_path).expanduser()
        if not candidate.is_absolute():
            raise RuntimeError("checkpoint_path_must_be_absolute")
        self._checkpoint_path = candidate.resolve(strict=False)

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    def load(self) -> FileWatcherCheckpoint | None:
        path = self._checkpoint_path
        if not path.exists():
            return None
        if not path.is_file():
            raise RuntimeError("checkpoint_not_file")
        try:
            payload = path.read_bytes()
        except OSError:
            raise RuntimeError("checkpoint_read_failed") from None
        return decode_file_watcher_checkpoint(payload)

    def save(self, checkpoint: FileWatcherCheckpoint) -> None:
        try:
            payload = encode_file_watcher_checkpoint(checkpoint)
        except Exception:
            raise RuntimeError("checkpoint_write_failed") from None

        parent = self._checkpoint_path.parent
        temp_path: Path | None = None
        try:
            parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(
                prefix="lkw-file-watcher-checkpoint.",
                suffix=".tmp",
                dir=str(parent),
            )
            temp_path = Path(temp_name)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._checkpoint_path)
            temp_path = None
        except RuntimeError:
            raise
        except Exception:
            raise RuntimeError("checkpoint_write_failed") from None
        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass


def restore_file_watcher_runtime(
    *,
    runtime: FileWatcherRuntime,
    store: JsonFileWatcherCheckpointStore,
    now_monotonic: float,
) -> bool:
    """Load a checkpoint and restore runtime state; False when none exists."""
    checkpoint = store.load()
    if checkpoint is None:
        return False
    runtime.restore_checkpoint(checkpoint, now_monotonic=now_monotonic)
    return True
