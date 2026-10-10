# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2 closed-world configured/effective provenance qualification types."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ConfiguredExecutionPathClass(enum.StrEnum):
    A_CANONICAL = "A"
    B_SANCTIONED_NON_CONFIGURED = "B"
    C_TEST_REFERENCE_ONLY = "C"
    D_LEGACY_DEAD = "D"
    E_PRODUCTION_BYPASS = "E"
    F_UNCLEAR = "F"


@dataclass(frozen=True, slots=True)
class RegisteredConfiguredExecutionPath:
    """One closed-world inventory row (file-level or surface-level)."""

    path: str
    surface_id: str
    classification: ConfiguredExecutionPathClass
    summary: str
    evidence_nodeid: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.surface_id)


@dataclass(frozen=True, slots=True)
class ClosedWorldParityResult:
    unknown: frozenset[tuple[str, str]]
    orphan: frozenset[tuple[str, str]]
    duplicate_registry_keys: frozenset[tuple[str, str]]
    production_bypass: frozenset[tuple[str, str]]
    unclassified: frozenset[tuple[str, str]]

    @property
    def ok(self) -> bool:
        return (
            not self.unknown
            and not self.orphan
            and not self.duplicate_registry_keys
            and not self.production_bypass
            and not self.unclassified
        )


@dataclass(frozen=True, slots=True)
class SemanticOwnerRow:
    concern: str
    owner: str
    owner_count: int
    justification: str | None = None


@dataclass(frozen=True, slots=True)
class AdversarialBundleRow:
    bundle_id: str
    scenario: str
    test_module: str
    test_id: str
    status: str
