# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R4 closed-world registry and parity types."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, TypeVar

_SurfaceKeyT = TypeVar("_SurfaceKeyT", bound=tuple[str, ...])


class ModelCallSurfaceClassification(enum.StrEnum):
    CANONICAL_PRIMARY = "canonical_primary"
    CANONICAL_INTERNAL_OPTIMIZATION = "canonical_internal_optimization"
    NON_PRODUCTION = "non_production"
    NOT_LLM_ADAPTER_CALL = "not_llm_adapter_call"


class ContextSurfaceClassification(enum.StrEnum):
    CANONICAL_CONTEXT_ENGINE_ASSEMBLY = "canonical_context_engine_assembly"
    CANONICAL_CONTEXT_ASSEMBLED_RECORDER = "canonical_context_assembled_recorder"
    CANONICAL_UAEP_CONTEXT_EMITTER = "canonical_uaep_context_emitter"
    CANONICAL_ATTRIBUTION_BIND = "canonical_attribution_bind"
    PAYLOAD_CODEC_OR_DESERIALIZE = "payload_codec_or_deserialize"
    ENUM_OR_POLICY_REFERENCE = "enum_or_policy_reference"
    ATTRIBUTION_CONSUMER = "attribution_consumer"


@dataclass(frozen=True, slots=True)
class RegisteredModelCallSurface:
    path: str
    method: str
    classification: ModelCallSurfaceClassification
    semantic_reason: str
    canonical_owner: str
    canonical_contract: str
    evidence_nodeid: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.method)


@dataclass(frozen=True, slots=True)
class RegisteredContextSurface:
    path: str
    surface_kind: str
    classification: ContextSurfaceClassification
    semantic_reason: str
    canonical_owner: str
    canonical_contract: str
    evidence_nodeid: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.surface_kind)


@dataclass(frozen=True, slots=True)
class SurfaceParityResult:
    ok: bool
    unknown: frozenset[tuple[str, str]]
    orphan: frozenset[tuple[str, str]]
    duplicate_registry_keys: frozenset[tuple[str, str]]


class _RegistryRow(Protocol[_SurfaceKeyT]):
    @property
    def key(self) -> _SurfaceKeyT: ...


def compare_discovered_to_registry(
    discovered_keys: frozenset[tuple[str, str]],
    registry: tuple[_RegistryRow[tuple[str, str]], ...],
) -> SurfaceParityResult:
    registry_keys_list = [row.key for row in registry]
    duplicate_registry_keys: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    for key in registry_keys_list:
        if key in seen:
            duplicate_registry_keys.add(key)
        seen.add(key)
    registry_keys = frozenset(registry_keys_list)
    unknown = discovered_keys - registry_keys
    orphan = registry_keys - discovered_keys
    ok = not unknown and not orphan and not duplicate_registry_keys
    return SurfaceParityResult(
        ok=ok,
        unknown=frozenset(unknown),
        orphan=frozenset(orphan),
        duplicate_registry_keys=frozenset(duplicate_registry_keys),
    )


__all__ = [
    "ContextSurfaceClassification",
    "ModelCallSurfaceClassification",
    "RegisteredContextSurface",
    "RegisteredModelCallSurface",
    "SurfaceParityResult",
    "compare_discovered_to_registry",
]
