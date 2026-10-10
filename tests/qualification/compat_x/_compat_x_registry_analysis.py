# © Artur Czarnecki. All rights reserved.

"""Registry overlap analysis for CONTRACT vs RUNTIME vs event payload identities."""

from __future__ import annotations

from functools import lru_cache

from intergrax.contracts.migrations.registry import CONTRACT_SCHEMA_REGISTRY
from intergrax.runtime.events.payload_registry import list_registered_payload_schema_ids
from intergrax.runtime.schema.registry import RUNTIME_SCHEMA_REGISTRY

from tests.qualification.compat_x._compat_x_types import RegistryOverlapMatrix


def _normalize_identity(value: str) -> str:
    return value.strip().lower()


def _contract_identities() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in CONTRACT_SCHEMA_REGISTRY:
        mapping[_normalize_identity(entry.contract_name)] = entry.current_version
    return mapping


def _runtime_identities() -> dict[str, str]:
    return {_normalize_identity(key): version for key, version in RUNTIME_SCHEMA_REGISTRY.items()}


def _event_identities() -> dict[str, str]:
    return {_normalize_identity(schema_id): schema_id for schema_id in list_registered_payload_schema_ids()}


@lru_cache(maxsize=1)
def analyze_registry_overlap() -> RegistryOverlapMatrix:
    contracts = _contract_identities()
    runtime = _runtime_identities()
    events = _event_identities()
    contract_keys = frozenset(contracts)
    runtime_keys = frozenset(runtime)
    event_keys = frozenset(events)
    overlap_cr = contract_keys & runtime_keys
    overlap_ce = contract_keys & event_keys
    overlap_re = runtime_keys & event_keys
    overlap_all = contract_keys & runtime_keys & event_keys
    only_contracts = contract_keys - runtime_keys - event_keys
    only_runtime = runtime_keys - contract_keys - event_keys
    only_event = event_keys - contract_keys - runtime_keys
    conflicts: list[tuple[str, str, str]] = []
    for key in overlap_cr:
        if contracts[key] != runtime[key]:
            conflicts.append((key, contracts[key], runtime[key]))
    return RegistryOverlapMatrix(
        only_contracts=only_contracts,
        only_runtime=only_runtime,
        only_event=only_event,
        overlap_contract_runtime=overlap_cr,
        overlap_contract_event=overlap_ce,
        overlap_runtime_event=overlap_re,
        overlap_all_three=overlap_all,
        version_conflicts=tuple(conflicts),
    )
