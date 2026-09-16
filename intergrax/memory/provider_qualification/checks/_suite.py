# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderCheckSeverity,
)

T = TypeVar("T")


def validate_canonical_check_suite(
    checks: tuple[T, ...],
    *,
    capability: MemoryProviderCapabilityKind,
    check_id_of: Callable[[T], str],
    capability_of: Callable[[T], MemoryProviderCapabilityKind],
    severity_of: Callable[[T], MemoryProviderCheckSeverity],
) -> None:
    seen: set[str] = set()
    for item in checks:
        check_id = check_id_of(item)
        if not check_id.strip():
            raise ValueError(f"check_id must be non-empty for {capability.value}")
        if check_id in seen:
            raise ValueError(
                f"duplicate check_id {check_id!r} in canonical suite for {capability.value}"
            )
        seen.add(check_id)
        if capability_of(item) is not capability:
            raise ValueError(
                f"check {check_id!r} capability mismatch for {capability.value}"
            )
        if severity_of(item) is not MemoryProviderCheckSeverity.REQUIRED:
            raise ValueError(f"canonical check {check_id!r} must be REQUIRED")


def merge_canonical_and_extra_checks(
    defaults: tuple[T, ...],
    extra: tuple[T, ...],
    *,
    id_of: Callable[[T], str],
) -> tuple[T, ...]:
    by_id: dict[str, T] = {id_of(item): item for item in defaults}
    for item in extra:
        check_id = id_of(item)
        if check_id in by_id:
            raise ValueError(
                f"custom qualification check cannot override canonical check_id {check_id!r}"
            )
        by_id[check_id] = item
    return tuple(by_id[key] for key in sorted(by_id))
