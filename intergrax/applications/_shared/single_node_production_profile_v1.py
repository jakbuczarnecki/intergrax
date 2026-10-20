# © Artur Czarnecki. All rights reserved.

"""Operator-approved single-node Linux VM production profile v1 (PROD-Q OD-R2-03).

Typed configuration and CONFIGURED-level validation only. Target VM guarantees
(ext4, mount identity, systemd lifecycle) require operator TARGET-VERIFIED evidence.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from intergrax.applications._shared.settings_loader import EnvReader
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionProcessStorage,
)
from intergrax.hosting.instance._native_lock import NativeFileLock, NativeFileLockError

SINGLE_NODE_PRODUCTION_PROFILE_V1_ID: Final = "single_node_linux_vm.production_profile.v1"

_ENV_PREFIX: Final = "INTERGRAX_SINGLE_NODE_PRODUCTION_V1_"
_WRITER_LOCK_FILE_NAME: Final = ".intergrax-single-node-production-v1.writer.lock"

_DEFAULT_GCA_DB_NAME: Final = "gca_host.sqlite"
_DEFAULT_PIS_DB_NAME: Final = "provider_invocation.sqlite"
_DEFAULT_AD_DB_NAME: Final = "agent_distribution.sqlite"
_DEFAULT_CONTINUATION_NAME: Final = "execution_continuation.seal.json"

EphemeralPathKind = Literal["tmp_dir", "memory_fs", "qualification_locator", "none"]


class SingleNodeProductionProfileV1ConfigurationError(ValueError):
    """Fail-closed single-node production profile v1 configuration error."""


@dataclass(frozen=True, slots=True)
class SingleNodeProductionProfileV1CrossProcessWriterHold:
    """Advisory cross-process writer lock for profile v1 (claim 3 — not SQLite serialization)."""

    lock_path: Path | None
    _lock: NativeFileLock | None

    def release(self) -> None:
        if self._lock is not None:
            self._lock.close()
            object.__setattr__(self, "_lock", None)

    def __enter__(self) -> SingleNodeProductionProfileV1CrossProcessWriterHold:
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class SqliteEffectiveDurabilityPragmas:
    """Effective SQLite pragmas observed on an opened database file."""

    journal_mode: str
    synchronous: int


@dataclass(frozen=True, slots=True)
class LinuxExt4MountProbeResult:
    """Best-effort mount probe (TARGET-VERIFIED only on operator Linux VM)."""

    applicable: bool
    mount_point: str | None
    fstype: str | None
    is_ext4: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SingleNodeProductionProfileV1Configuration:
    """Durable storage and identity contract for one strict-production process."""

    profile_id: str
    durable_data_directory: Path
    service_runtime_user: str
    trusted_artifact_parent_roots: tuple[Path, ...]
    gca_host_sqlite_path: Path
    provider_invocation_sqlite_path: Path
    agent_distribution_sqlite_path: Path
    continuation_state_file: Path
    enforce_single_active_writer_process: bool = True

    def resolve_strict_production_process_storage(self) -> StrictProductionProcessStorage:
        return StrictProductionProcessStorage(
            agent_distribution_db=self.agent_distribution_sqlite_path,
            continuation_state_file=self.continuation_state_file,
            provider_invocation_db=self.provider_invocation_sqlite_path,
            governed_external_work_host_db=self.gca_host_sqlite_path,
            trusted_artifact_parent_roots=self.trusted_artifact_parent_roots,
        )


def _ephemeral_path_kind(path: Path) -> EphemeralPathKind:
    resolved = path.expanduser().resolve(strict=False)
    text = resolved.as_posix()
    tmp = Path(tempfile.gettempdir()).resolve()
    if resolved == tmp or str(resolved).startswith(f"{tmp}{os.sep}"):
        return "tmp_dir"
    if text.startswith("/dev/shm/") or text.startswith("/run/"):
        return "memory_fs"
    if text.startswith("test://"):
        return "qualification_locator"
    return "none"


def _reject_ephemeral_path(path: Path, *, field_name: str) -> None:
    kind = _ephemeral_path_kind(path)
    if kind != "none":
        raise SingleNodeProductionProfileV1ConfigurationError(
            f"{field_name} must not use ephemeral or qualification storage "
            f"({kind}): {path}",
        )


def _parse_roots(env: EnvReader, data_dir: Path) -> tuple[Path, ...]:
    raw = env.optional_str("TRUSTED_ARTIFACT_PARENT_ROOTS")
    if raw:
        roots = tuple(Path(part.strip()) for part in raw.split(os.pathsep) if part.strip())
        if not roots:
            raise SingleNodeProductionProfileV1ConfigurationError(
                "TRUSTED_ARTIFACT_PARENT_ROOTS must list at least one path when set",
            )
        return roots
    return (data_dir / "trusted_artifacts",)


def load_single_node_production_profile_v1_from_env(
    *,
    env: EnvReader | None = None,
) -> SingleNodeProductionProfileV1Configuration:
    """Load profile v1 from ``INTERGRAX_SINGLE_NODE_PRODUCTION_V1_*`` environment variables."""
    reader = env or EnvReader(_ENV_PREFIX)
    data_raw = reader.optional_str("DATA_DIR")
    if not data_raw:
        raise SingleNodeProductionProfileV1ConfigurationError(
            f"{_ENV_PREFIX}DATA_DIR is required for single-node production profile v1",
        )
    data_dir = Path(data_raw)
    service_user = reader.str("SERVICE_RUNTIME_USER", default="intergrax-gca")
    enforce_single = reader.bool("ENFORCE_SINGLE_ACTIVE_WRITER", default=True)
    roots = _parse_roots(reader, data_dir)
    gca = Path(
        reader.str("GCA_HOST_SQLITE_PATH", default=str(data_dir / _DEFAULT_GCA_DB_NAME)),
    )
    pis = Path(
        reader.str(
            "PROVIDER_INVOCATION_SQLITE_PATH",
            default=str(data_dir / _DEFAULT_PIS_DB_NAME),
        ),
    )
    ad = Path(
        reader.str(
            "AGENT_DISTRIBUTION_SQLITE_PATH",
            default=str(data_dir / _DEFAULT_AD_DB_NAME),
        ),
    )
    continuation = Path(
        reader.str(
            "CONTINUATION_STATE_FILE",
            default=str(data_dir / _DEFAULT_CONTINUATION_NAME),
        ),
    )
    return SingleNodeProductionProfileV1Configuration(
        profile_id=SINGLE_NODE_PRODUCTION_PROFILE_V1_ID,
        durable_data_directory=data_dir,
        service_runtime_user=service_user,
        trusted_artifact_parent_roots=roots,
        gca_host_sqlite_path=gca,
        provider_invocation_sqlite_path=pis,
        agent_distribution_sqlite_path=ad,
        continuation_state_file=continuation,
        enforce_single_active_writer_process=enforce_single,
    )


def validate_configured_single_node_production_profile_v1(
    config: SingleNodeProductionProfileV1Configuration,
) -> None:
    """CONFIGURED checks: typed paths, co-location, and ephemeral storage denial."""
    if config.profile_id != SINGLE_NODE_PRODUCTION_PROFILE_V1_ID:
        raise SingleNodeProductionProfileV1ConfigurationError(
            f"unsupported profile_id {config.profile_id!r}",
        )
    if not config.service_runtime_user.strip():
        raise SingleNodeProductionProfileV1ConfigurationError(
            "service_runtime_user must be non-empty",
        )
    if not config.trusted_artifact_parent_roots:
        raise SingleNodeProductionProfileV1ConfigurationError(
            "trusted_artifact_parent_roots must be non-empty",
        )
    for label, path in (
        ("durable_data_directory", config.durable_data_directory),
        ("gca_host_sqlite_path", config.gca_host_sqlite_path),
        ("provider_invocation_sqlite_path", config.provider_invocation_sqlite_path),
        ("agent_distribution_sqlite_path", config.agent_distribution_sqlite_path),
        ("continuation_state_file", config.continuation_state_file),
    ):
        _reject_ephemeral_path(path, field_name=label)
    for root in config.trusted_artifact_parent_roots:
        _reject_ephemeral_path(root, field_name="trusted_artifact_parent_root")
    data_resolved = config.durable_data_directory.expanduser().resolve(strict=False)
    for path in (
        config.gca_host_sqlite_path,
        config.provider_invocation_sqlite_path,
        config.agent_distribution_sqlite_path,
        config.continuation_state_file,
    ):
        parent = path.expanduser().resolve(strict=False).parent
        if data_resolved not in (parent, *parent.parents):
            raise SingleNodeProductionProfileV1ConfigurationError(
                "all durable SQLite and continuation paths must reside under DATA_DIR "
                f"({data_resolved}); got {path}",
            )
    data_path = config.durable_data_directory.expanduser()
    if not data_path.is_dir():
        raise SingleNodeProductionProfileV1ConfigurationError(
            f"durable_data_directory must exist and be a directory: {data_path}",
        )


def writer_lock_path_for_profile_v1(config: SingleNodeProductionProfileV1Configuration) -> Path:
    return config.durable_data_directory / _WRITER_LOCK_FILE_NAME


def acquire_cross_process_writer_admission_for_profile_v1(
    config: SingleNodeProductionProfileV1Configuration,
) -> SingleNodeProductionProfileV1CrossProcessWriterHold:
    """Acquire OS-level exclusive writer lock when enforcement is enabled (claim 3)."""
    if not config.enforce_single_active_writer_process:
        return SingleNodeProductionProfileV1CrossProcessWriterHold(lock_path=None, _lock=None)
    lock_path = writer_lock_path_for_profile_v1(config)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock = NativeFileLock(lock_path)
    except NativeFileLockError as exc:
        raise SingleNodeProductionProfileV1ConfigurationError(
            "another process holds the single-node production profile v1 writer lock "
            f"({lock_path}); concurrent GCA/PIS writers are not supported",
        ) from exc
    return SingleNodeProductionProfileV1CrossProcessWriterHold(
        lock_path=lock_path,
        _lock=lock,
    )


def load_validate_and_admit_single_node_production_profile_v1_from_env(
    *,
    env: EnvReader | None = None,
    acquire_cross_process_writer: bool = True,
) -> tuple[
    SingleNodeProductionProfileV1Configuration,
    StrictProductionProcessStorage,
    SingleNodeProductionProfileV1CrossProcessWriterHold,
]:
    """Load profile v1, run CONFIGURED validation, optionally acquire cross-process writer lock."""
    config = load_single_node_production_profile_v1_from_env(env=env)
    validate_configured_single_node_production_profile_v1(config)
    hold = (
        acquire_cross_process_writer_admission_for_profile_v1(config)
        if acquire_cross_process_writer
        else SingleNodeProductionProfileV1CrossProcessWriterHold(lock_path=None, _lock=None)
    )
    storage = config.resolve_strict_production_process_storage()
    return config, storage, hold


def assert_owned_connection_sqlite_wal_full_pragmas(
    pragmas: SqliteEffectiveDurabilityPragmas,
    *,
    db_path: Path,
) -> SqliteEffectiveDurabilityPragmas:
    """REPO-TESTED assertion on application-owned connection pragmas (not inspection-only)."""
    if pragmas.journal_mode != "wal":
        raise SingleNodeProductionProfileV1ConfigurationError(
            f"expected journal_mode=wal on owned connection, got {pragmas.journal_mode!r} "
            f"for {db_path}",
        )
    if pragmas.synchronous != 2:
        raise SingleNodeProductionProfileV1ConfigurationError(
            f"expected synchronous=FULL (2) on owned connection, got {pragmas.synchronous} "
            f"for {db_path}",
        )
    return pragmas


def read_sqlite_effective_durability_pragmas(db_path: Path) -> SqliteEffectiveDurabilityPragmas:
    """Read effective WAL / synchronous pragmas from an on-disk SQLite file."""
    if not db_path.exists():
        raise SingleNodeProductionProfileV1ConfigurationError(
            f"sqlite database does not exist: {db_path}",
        )
    try:
        conn = sqlite3.connect(db_path)
        try:
            journal = str(conn.execute("PRAGMA journal_mode;").fetchone()[0])
            synchronous = int(conn.execute("PRAGMA synchronous;").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise SingleNodeProductionProfileV1ConfigurationError(
            f"unable to read sqlite pragmas for {db_path}",
        ) from exc
    return SqliteEffectiveDurabilityPragmas(
        journal_mode=journal.lower(),
        synchronous=synchronous,
    )


def assert_repo_tested_sqlite_wal_full_pragmas(db_path: Path) -> SqliteEffectiveDurabilityPragmas:
    """REPO-TESTED assertion: journal_mode wal and synchronous FULL (2)."""
    pragmas = read_sqlite_effective_durability_pragmas(db_path)
    if pragmas.journal_mode != "wal":
        raise SingleNodeProductionProfileV1ConfigurationError(
            f"expected journal_mode=wal, got {pragmas.journal_mode!r} for {db_path}",
        )
    if pragmas.synchronous != 2:
        raise SingleNodeProductionProfileV1ConfigurationError(
            f"expected synchronous=FULL (2), got {pragmas.synchronous} for {db_path}",
        )
    return pragmas


def probe_linux_ext4_mount_for_path(path: Path) -> LinuxExt4MountProbeResult:
    """TARGET-VERIFIED helper: inspect /proc/mounts on Linux only."""
    if sys.platform != "linux":
        return LinuxExt4MountProbeResult(
            applicable=False,
            mount_point=None,
            fstype=None,
            is_ext4=False,
            detail="mount probe applies on Linux target VM only",
        )
    resolved = path.expanduser().resolve(strict=False)
    proc_mounts = Path("/proc/mounts")
    if not proc_mounts.is_file():
        return LinuxExt4MountProbeResult(
            applicable=True,
            mount_point=None,
            fstype=None,
            is_ext4=False,
            detail="/proc/mounts unavailable",
        )
    best_mount = ""
    best_fstype = ""
    for line in proc_mounts.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point = parts[1]
        fstype = parts[2]
        mount_path = Path(mount_point)
        if resolved == mount_path or mount_path in resolved.parents:
            if len(mount_point) >= len(best_mount):
                best_mount = mount_point
                best_fstype = fstype
    if not best_mount:
        return LinuxExt4MountProbeResult(
            applicable=True,
            mount_point=None,
            fstype=None,
            is_ext4=False,
            detail=f"no mount entry for {resolved}",
        )
    return LinuxExt4MountProbeResult(
        applicable=True,
        mount_point=best_mount,
        fstype=best_fstype,
        is_ext4=best_fstype == "ext4",
        detail="ok",
    )


__all__ = [
    "LinuxExt4MountProbeResult",
    "SingleNodeProductionProfileV1Configuration",
    "SingleNodeProductionProfileV1ConfigurationError",
    "SingleNodeProductionProfileV1CrossProcessWriterHold",
    "SINGLE_NODE_PRODUCTION_PROFILE_V1_ID",
    "SqliteEffectiveDurabilityPragmas",
    "acquire_cross_process_writer_admission_for_profile_v1",
    "assert_owned_connection_sqlite_wal_full_pragmas",
    "assert_repo_tested_sqlite_wal_full_pragmas",
    "load_single_node_production_profile_v1_from_env",
    "load_validate_and_admit_single_node_production_profile_v1_from_env",
    "probe_linux_ext4_mount_for_path",
    "read_sqlite_effective_durability_pragmas",
    "validate_configured_single_node_production_profile_v1",
    "writer_lock_path_for_profile_v1",
]
