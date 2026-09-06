# © Artur Czarnecki. All rights reserved.

"""Active effective profile revision pointer stores (P1.6)."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

from intergrax.applications.contracts.profile_resolution.activation import (
    ActiveEffectiveProfileRevisionBinding,
    ActiveEffectiveProfileRevisionCasOutcome,
    ActiveEffectiveProfileRevisionCasResult,
)
from intergrax.applications.contracts.profile_resolution.errors import (
    EffectiveProfileActivationPersistenceError,
    EffectiveProfileRevisionError,
)
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore

_ACTIVE_KV_PREFIX = "effective_profile_active_revision"
_ACTIVE_SCHEMA_VERSION = 1


def _scope_key(scope: EffectiveProfileRevisionScope) -> tuple[str, str | None]:
    return (scope.application_id, scope.tenant_id)


def _active_kv_key(scope: EffectiveProfileRevisionScope) -> str:
    app_id, tenant_id = _scope_key(scope)
    tenant_suffix = tenant_id or "_global"
    return f"{_ACTIVE_KV_PREFIX}:{app_id}:{tenant_suffix}"


def _binding_storage_payload(binding: ActiveEffectiveProfileRevisionBinding) -> dict[str, Any]:
    payload = json.loads(binding.model_dump_json())
    payload["revision_id"] = binding.revision_id.value
    return payload


def encode_active_effective_profile_revision_binding(
    binding: ActiveEffectiveProfileRevisionBinding,
) -> bytes:
    payload: dict[str, Any] = {
        "schema_version": _ACTIVE_SCHEMA_VERSION,
        "binding": _binding_storage_payload(binding),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_active_effective_profile_revision_binding(
    raw: bytes,
) -> ActiveEffectiveProfileRevisionBinding:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EffectiveProfileRevisionError(
            "invalid active effective profile revision binding encoding",
        ) from exc
    if not isinstance(payload, dict):
        raise EffectiveProfileRevisionError("invalid active effective profile revision payload")
    if payload.get("schema_version") != _ACTIVE_SCHEMA_VERSION:
        raise EffectiveProfileRevisionError(
            "unsupported active effective profile revision schema version",
        )
    binding_raw = payload.get("binding")
    if not isinstance(binding_raw, dict):
        raise EffectiveProfileRevisionError("invalid active effective profile revision snapshot")
    return ActiveEffectiveProfileRevisionBinding.model_validate(binding_raw)


def _expected_matches(
    current: ActiveEffectiveProfileRevisionBinding | None,
    expected_revision_id: EffectiveProfileRevisionId | None,
) -> bool:
    if expected_revision_id is None:
        return current is None
    if current is None:
        return False
    return current.revision_id == expected_revision_id


def _cas_result(
    *,
    outcome: ActiveEffectiveProfileRevisionCasOutcome,
    current: ActiveEffectiveProfileRevisionBinding | None,
) -> ActiveEffectiveProfileRevisionCasResult:
    return ActiveEffectiveProfileRevisionCasResult(
        outcome=outcome,
        current_binding=current,
    )


class InMemoryActiveEffectiveProfileRevisionStore:
    """Thread-safe in-memory active revision pointer store."""

    def __init__(
        self,
        *,
        persist_hook: Callable[[], None] | None = None,
    ) -> None:
        self._bindings: dict[tuple[str, str | None], ActiveEffectiveProfileRevisionBinding] = {}
        self._lock = threading.RLock()
        self._persist_hook = persist_hook

    @property
    def is_durable(self) -> bool:
        return False

    def get_active(
        self,
        scope: EffectiveProfileRevisionScope,
    ) -> ActiveEffectiveProfileRevisionBinding | None:
        with self._lock:
            return self._bindings.get(_scope_key(scope))

    def compare_and_set_active(
        self,
        scope: EffectiveProfileRevisionScope,
        *,
        expected_revision_id: EffectiveProfileRevisionId | None,
        new_binding: ActiveEffectiveProfileRevisionBinding,
    ) -> ActiveEffectiveProfileRevisionCasResult:
        if new_binding.scope != scope:
            raise EffectiveProfileRevisionError("active binding scope must match CAS scope")
        with self._lock:
            key = _scope_key(scope)
            current = self._bindings.get(key)
            if not _expected_matches(current, expected_revision_id):
                return _cas_result(outcome=ActiveEffectiveProfileRevisionCasOutcome.CONFLICT, current=current)
            if current is not None and current == new_binding:
                return _cas_result(outcome=ActiveEffectiveProfileRevisionCasOutcome.UNCHANGED, current=current)
            if current is not None and self._persist_hook is not None:
                try:
                    self._persist_hook()
                except Exception as exc:
                    raise EffectiveProfileActivationPersistenceError(
                        "active effective profile revision persistence failed",
                    ) from exc
            self._bindings[key] = new_binding
            outcome = (
                ActiveEffectiveProfileRevisionCasOutcome.UNCHANGED
                if current == new_binding
                else ActiveEffectiveProfileRevisionCasOutcome.UPDATED
            )
            return _cas_result(outcome=outcome, current=new_binding)


class KvActiveEffectiveProfileRevisionStore:
    """DistributedKVStore-backed active revision pointer store."""

    def __init__(
        self,
        kv_store: DistributedKVStore,
        *,
        persist_hook: Callable[[], None] | None = None,
    ) -> None:
        self._kv_store = kv_store
        self._persist_hook = persist_hook

    @property
    def is_durable(self) -> bool:
        return True

    def get_active(
        self,
        scope: EffectiveProfileRevisionScope,
    ) -> ActiveEffectiveProfileRevisionBinding | None:
        tenant_id = scope.tenant_id or scope.application_id
        raw = self._kv_store.get(tenant_id=tenant_id, key=_active_kv_key(scope))
        if raw is None:
            return None
        binding = decode_active_effective_profile_revision_binding(raw)
        if binding.scope != scope:
            return None
        return binding

    def compare_and_set_active(
        self,
        scope: EffectiveProfileRevisionScope,
        *,
        expected_revision_id: EffectiveProfileRevisionId | None,
        new_binding: ActiveEffectiveProfileRevisionBinding,
    ) -> ActiveEffectiveProfileRevisionCasResult:
        if new_binding.scope != scope:
            raise EffectiveProfileRevisionError("active binding scope must match CAS scope")
        tenant_id = scope.tenant_id or scope.application_id
        key = _active_kv_key(scope)
        current = self.get_active(scope)
        if not _expected_matches(current, expected_revision_id):
            return _cas_result(outcome=ActiveEffectiveProfileRevisionCasOutcome.CONFLICT, current=current)
        if current is not None and current == new_binding:
            return _cas_result(outcome=ActiveEffectiveProfileRevisionCasOutcome.UNCHANGED, current=current)
        if current is not None and self._persist_hook is not None:
            try:
                self._persist_hook()
            except Exception as exc:
                raise EffectiveProfileActivationPersistenceError(
                    "active effective profile revision persistence failed",
                ) from exc
        encoded = encode_active_effective_profile_revision_binding(new_binding)
        expected_raw = (
            encode_active_effective_profile_revision_binding(current)
            if current is not None
            else None
        )
        if not self._kv_store.compare_and_set(
            tenant_id=tenant_id,
            key=key,
            expected=expected_raw,
            new_value=encoded,
        ):
            refreshed = self.get_active(scope)
            return _cas_result(
                outcome=ActiveEffectiveProfileRevisionCasOutcome.CONFLICT,
                current=refreshed,
            )
        return _cas_result(outcome=ActiveEffectiveProfileRevisionCasOutcome.UPDATED, current=new_binding)


def wire_active_effective_profile_revision_store(
    *,
    kv_store: DistributedKVStore | None = None,
    active_store: InMemoryActiveEffectiveProfileRevisionStore | None = None,
) -> InMemoryActiveEffectiveProfileRevisionStore | KvActiveEffectiveProfileRevisionStore:
    if active_store is not None:
        return active_store
    if kv_store is not None:
        return KvActiveEffectiveProfileRevisionStore(kv_store)
    return InMemoryActiveEffectiveProfileRevisionStore()
