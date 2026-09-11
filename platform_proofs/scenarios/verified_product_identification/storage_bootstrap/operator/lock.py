"""Single-host operator lock for concurrent full storage load protection."""

from __future__ import annotations

import json
import os
import socket
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.errors import (
    StorageLoadOperatorLockError,
)

_LOCK_FILE_NAME = "operator.lock"


@dataclass(frozen=True, slots=True)
class OperatorLockMetadata:
    run_id: str
    pid: int
    host_identifier: str
    started_at: str
    artifact_content_identity: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "pid": self.pid,
                "host_identifier": self.host_identifier,
                "started_at": self.started_at,
                "artifact_content_identity": self.artifact_content_identity,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> OperatorLockMetadata:
        parsed = json.loads(payload)
        return cls(
            run_id=str(parsed["run_id"]),
            pid=int(parsed["pid"]),
            host_identifier=str(parsed["host_identifier"]),
            started_at=str(parsed["started_at"]),
            artifact_content_identity=str(parsed["artifact_content_identity"]),
        )


@dataclass(slots=True)
class OperatorLock:
    checkpoint_root: Path
    metadata: OperatorLockMetadata
    _path: Path | None = None

    @property
    def path(self) -> Path:
        return self.checkpoint_root / _LOCK_FILE_NAME

    def acquire(self) -> None:
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.path
        if lock_path.is_file():
            raise StorageLoadOperatorLockError(
                f"operator lock already held: {lock_path}"
            )
        try:
            with lock_path.open("x", encoding="utf-8") as handle:
                handle.write(self.metadata.to_json())
                handle.write("\n")
        except FileExistsError as exc:
            raise StorageLoadOperatorLockError(
                f"operator lock already held: {lock_path}"
            ) from exc
        except OSError as exc:
            raise StorageLoadOperatorLockError(
                f"failed to acquire operator lock: {lock_path}"
            ) from exc
        self._path = lock_path

    def release(self) -> None:
        if self._path is None:
            return
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            raise StorageLoadOperatorLockError(
                f"failed to release operator lock: {self._path}"
            ) from exc
        self._path = None


def build_operator_lock_metadata(
    *,
    artifact_content_identity: str,
    run_id: str | None = None,
) -> OperatorLockMetadata:
    return OperatorLockMetadata(
        run_id=run_id or uuid.uuid4().hex,
        pid=os.getpid(),
        host_identifier=socket.gethostname(),
        started_at=datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
        artifact_content_identity=artifact_content_identity,
    )
