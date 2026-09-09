"""Single-host filesystem durable checkpoint store for storage bootstrap."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.codec import (
    decode_checkpoint,
    encode_checkpoint,
    run_identity_digest,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.contracts import (
    BootstrapCheckpointState,
    BootstrapRunIdentity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.errors import (
    CheckpointAlreadyExists,
    CheckpointConcurrentModification,
    CheckpointCorrupt,
    CheckpointNotFound,
    CheckpointPersistenceError,
)


@dataclass(slots=True)
class FilesystemBootstrapCheckpointStore:
    """Local filesystem checkpoint store — safe for single-host use only."""

    checkpoint_root: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def load(self, run_identity: BootstrapRunIdentity) -> BootstrapCheckpointState | None:
        path = self._state_path(run_identity)
        if not path.is_file():
            return None
        try:
            return decode_checkpoint(path.read_bytes())
        except CheckpointCorrupt as exc:
            raise CheckpointCorrupt(f"{path}: {exc}") from exc
        except OSError as exc:
            raise CheckpointPersistenceError(f"failed to read checkpoint: {path}") from exc

    def initialize(
        self,
        run_identity: BootstrapRunIdentity,
        state: BootstrapCheckpointState,
    ) -> BootstrapCheckpointState:
        path = self._state_path(run_identity)
        with self._lock:
            if path.is_file():
                raise CheckpointAlreadyExists(f"checkpoint already exists: {path}")
            self._write_atomic(path, state)
            return state

    def commit_batch(
        self,
        *,
        run_identity: BootstrapRunIdentity,
        expected_revision: int,
        checkpoint: BootstrapCheckpointState,
    ) -> BootstrapCheckpointState:
        path = self._state_path(run_identity)
        with self._lock:
            if not path.is_file():
                raise CheckpointNotFound(f"checkpoint missing: {path}")
            try:
                current = decode_checkpoint(path.read_bytes())
            except CheckpointCorrupt as exc:
                raise CheckpointCorrupt(f"{path}: {exc}") from exc
            except OSError as exc:
                raise CheckpointPersistenceError(f"failed to read checkpoint: {path}") from exc
            if current.state_revision != expected_revision:
                raise CheckpointConcurrentModification(
                    f"expected revision {expected_revision}, found {current.state_revision}"
                )
            if checkpoint.state_revision != expected_revision + 1:
                raise CheckpointConcurrentModification(
                    f"next revision must be {expected_revision + 1}, got {checkpoint.state_revision}"
                )
            self._write_atomic(path, checkpoint)
            return checkpoint

    def _state_path(self, run_identity: BootstrapRunIdentity) -> Path:
        digest = run_identity_digest(run_identity)
        return self.checkpoint_root / digest / "state.json"

    def _write_atomic(self, path: Path, state: BootstrapCheckpointState) -> None:
        encoded = encode_checkpoint(state)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(encoded.bytes_payload)
            temp_path.replace(path)
        except OSError as exc:
            raise CheckpointPersistenceError(f"failed to persist checkpoint: {path}") from exc
