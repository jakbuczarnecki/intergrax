# © Artur Czarnecki. All rights reserved.

"""File-backed registry for harness online/shadow evaluation observations (W-OPS.11).

Tenant isolation (PDC-D5-R015): observations may carry a ``tenant_id``. Tenant-scoped
reads return only that tenant's observations and a read of another tenant's
observation is denied. Observations recorded without a tenant (legacy callers) are
never visible to a tenant-scoped read.

Concurrency semantics (PDC-D5-R018, DX-07) of ``FileOnlineEvaluationRegistry``:

- every read-modify-write runs under one exclusive section: an in-process lock plus a
  cross-process lock file (``<registry>.lock``, created with ``O_EXCL``);
- a writer that cannot acquire the lock within ``lock_timeout_seconds`` is **rejected**
  with ``OnlineEvaluationRegistryBusyError``; nothing is written;
- a lock file older than ``stale_lock_seconds`` is treated as abandoned and removed;
- the registry file is replaced atomically (temp file + ``os.replace``), so a crashed
  writer never leaves a truncated registry.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.online_evaluation_models import (
    OnlineEvaluationObservation,
)


class OnlineEvaluationTenantBoundaryError(PermissionError):
    """A tenant tried to read or write another tenant's (or an untenanted) observation."""


class OnlineEvaluationRegistryBusyError(RuntimeError):
    """The registry is locked by another writer; the write was rejected, not applied."""


class TenantOnlineEvaluationObservation(OnlineEvaluationObservation):
    """Observation with an optional tenant (``None`` only for legacy untenanted records)."""

    tenant_id: str | None = None


class _TenantRegistryStore(BaseModel):
    schema_version: str = "1.1.0"
    observations: list[TenantOnlineEvaluationObservation] = Field(default_factory=list)


def _require_tenant(tenant_id: str) -> str:
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise ValueError("tenant_id must be a non-blank string")
    return tenant_id


def _stamp(observation: OnlineEvaluationObservation, tenant_id: str | None) -> TenantOnlineEvaluationObservation:
    existing = getattr(observation, "tenant_id", None)
    if tenant_id is not None and existing is not None and existing != tenant_id:
        raise OnlineEvaluationTenantBoundaryError("observation belongs to a different tenant than the registry scope")
    payload = observation.model_dump()
    payload["tenant_id"] = tenant_id if tenant_id is not None else existing
    return TenantOnlineEvaluationObservation.model_validate(payload)


class OnlineEvaluationRegistry(Protocol):
    """Append-only store for harness evaluation observations."""

    def append(self, observation: OnlineEvaluationObservation) -> None: ...

    def list_observations(self) -> list[OnlineEvaluationObservation]: ...

    def clear(self) -> None: ...


class InMemoryOnlineEvaluationRegistry:
    """In-process registry for unit tests (thread-safe)."""

    def __init__(self) -> None:
        self._observations: list[OnlineEvaluationObservation] = []
        self._lock = threading.RLock()

    def append(self, observation: OnlineEvaluationObservation) -> None:
        with self._lock:
            self._observations.append(observation)

    def list_observations(self) -> list[OnlineEvaluationObservation]:
        with self._lock:
            return list(self._observations)

    def clear(self) -> None:
        with self._lock:
            self._observations.clear()

    def append_for_tenant(self, observation: OnlineEvaluationObservation, tenant_id: str) -> None:
        stamped = _stamp(observation, _require_tenant(tenant_id))
        with self._lock:
            self._observations.append(stamped)

    def list_observations_for_tenant(self, tenant_id: str) -> list[OnlineEvaluationObservation]:
        _require_tenant(tenant_id)
        with self._lock:
            return [item for item in self._observations if getattr(item, "tenant_id", None) == tenant_id]

    def get_observation(self, observation_id: str, *, tenant_id: str) -> OnlineEvaluationObservation:
        _require_tenant(tenant_id)
        with self._lock:
            for item in self._observations:
                if item.observation_id == observation_id:
                    if getattr(item, "tenant_id", None) != tenant_id:
                        raise OnlineEvaluationTenantBoundaryError("cross-tenant observation read denied")
                    return item
        raise KeyError(observation_id)

    def clear_tenant(self, tenant_id: str) -> None:
        _require_tenant(tenant_id)
        with self._lock:
            self._observations = [item for item in self._observations if getattr(item, "tenant_id", None) != tenant_id]


class FileOnlineEvaluationRegistry:
    """Persist observations under ``build/architecture_hardening/`` (gitignored)."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        lock_timeout_seconds: float = 10.0,
        stale_lock_seconds: float = 60.0,
    ) -> None:
        self._path = path or default_registry_path()
        self._lock_path = self._path.with_name(self._path.name + ".lock")
        self._lock_timeout = lock_timeout_seconds
        self._stale_after = stale_lock_seconds
        self._thread_lock = threading.RLock()

    # -- legacy surface (unchanged signatures) ----------------------------------------

    def append(self, observation: OnlineEvaluationObservation) -> None:
        stamped = _stamp(observation, None)
        with self._exclusive():
            store = self._load()
            store.observations.append(stamped)
            self._save(store)

    def list_observations(self) -> list[OnlineEvaluationObservation]:
        with self._exclusive():
            return list(self._load().observations)

    def clear(self) -> None:
        with self._exclusive():
            self._save(_TenantRegistryStore())

    # -- tenant surface ---------------------------------------------------------------

    def append_for_tenant(self, observation: OnlineEvaluationObservation, tenant_id: str) -> None:
        stamped = _stamp(observation, _require_tenant(tenant_id))
        with self._exclusive():
            store = self._load()
            store.observations.append(stamped)
            self._save(store)

    def list_observations_for_tenant(self, tenant_id: str) -> list[OnlineEvaluationObservation]:
        _require_tenant(tenant_id)
        with self._exclusive():
            return [item for item in self._load().observations if item.tenant_id == tenant_id]

    def get_observation(self, observation_id: str, *, tenant_id: str) -> OnlineEvaluationObservation:
        _require_tenant(tenant_id)
        with self._exclusive():
            for item in self._load().observations:
                if item.observation_id == observation_id:
                    if item.tenant_id != tenant_id:
                        raise OnlineEvaluationTenantBoundaryError("cross-tenant observation read denied")
                    return item
        raise KeyError(observation_id)

    def clear_tenant(self, tenant_id: str) -> None:
        _require_tenant(tenant_id)
        with self._exclusive():
            store = self._load()
            store.observations = [item for item in store.observations if item.tenant_id != tenant_id]
            self._save(store)

    # -- concurrency ------------------------------------------------------------------

    @contextlib.contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._thread_lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._acquire_file_lock()
            try:
                yield
            finally:
                with contextlib.suppress(OSError):
                    self._lock_path.unlink()

    def _busy(self) -> OnlineEvaluationRegistryBusyError:
        return OnlineEvaluationRegistryBusyError(f"evaluation registry is locked by another writer: {self._path.name}")

    def _acquire_file_lock(self) -> None:
        deadline = time.monotonic() + self._lock_timeout
        while True:
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except (FileExistsError, PermissionError):
                # PermissionError: Windows reports a lock file that is being deleted.
                self._remove_if_stale()
                if time.monotonic() >= deadline:
                    raise self._busy() from None
                time.sleep(0.005)
                continue
            with contextlib.suppress(OSError):
                os.write(fd, str(os.getpid()).encode("ascii"))
            os.close(fd)
            return

    def _remove_if_stale(self) -> None:
        try:
            age = time.time() - self._lock_path.stat().st_mtime
        except OSError:
            return
        if age > self._stale_after:
            with contextlib.suppress(OSError):
                self._lock_path.unlink()

    def _load(self) -> _TenantRegistryStore:
        if not self._path.is_file():
            return _TenantRegistryStore()
        raw = self._path.read_text(encoding="utf-8")
        return _TenantRegistryStore.model_validate_json(raw)

    def _save(self, store: _TenantRegistryStore) -> None:
        temp = self._path.with_name(f"{self._path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temp.write_text(store.model_dump_json(indent=2), encoding="utf-8")
        last_error: OSError | None = None
        for _ in range(20):
            try:
                os.replace(temp, self._path)
                return
            except PermissionError as exc:  # Windows: target briefly held by AV/indexer
                last_error = exc
                time.sleep(0.01)
        with contextlib.suppress(OSError):
            temp.unlink()
        if last_error is not None:
            raise last_error


class TenantScopedOnlineEvaluationRegistry:
    """Registry view bound to one tenant: writes are stamped, reads are filtered.

    Satisfies ``OnlineEvaluationRegistry`` so existing producers keep working, but it
    can only ever see its own tenant's observations.
    """

    def __init__(
        self,
        registry: InMemoryOnlineEvaluationRegistry | FileOnlineEvaluationRegistry,
        tenant_id: str,
    ) -> None:
        self._registry = registry
        self._tenant_id = _require_tenant(tenant_id)

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    def append(self, observation: OnlineEvaluationObservation) -> None:
        self._registry.append_for_tenant(observation, self._tenant_id)

    def list_observations(self) -> list[OnlineEvaluationObservation]:
        return self._registry.list_observations_for_tenant(self._tenant_id)

    def get_observation(self, observation_id: str) -> OnlineEvaluationObservation:
        return self._registry.get_observation(observation_id, tenant_id=self._tenant_id)

    def clear(self) -> None:
        self._registry.clear_tenant(self._tenant_id)


def default_registry_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "build" / "architecture_hardening" / "online_evaluation_observations.json"


_default_file_registry: FileOnlineEvaluationRegistry | None = None


def default_online_evaluation_registry() -> FileOnlineEvaluationRegistry:
    global _default_file_registry
    if _default_file_registry is None:
        _default_file_registry = FileOnlineEvaluationRegistry()
    return _default_file_registry


def reset_default_online_evaluation_registry_for_tests() -> None:
    """Clear module singleton between tests."""
    global _default_file_registry
    _default_file_registry = None
