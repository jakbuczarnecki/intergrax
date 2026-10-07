# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-P0 closed-world registry and provenance qualification types."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal

from tests.qualification.trace_x._trace_x_p4_registry_types import (
    SurfaceParityResult,
    compare_discovered_to_registry,
)

ProvenanceDomain = Literal["policy", "profile_revision", "configuration"]


class PolicyProvenanceSurfaceKind(enum.StrEnum):
    CONTRACT_DEFINITION = "contract_definition"
    EVIDENCE_COMPOSITION = "evidence_composition"
    GOVERNANCE_RECORDING = "governance_recording"
    INSPECTION_PROJECTION = "inspection_projection"
    OBLIGATION_DERIVATION = "obligation_derivation"


class ProfileRevisionSurfaceKind(enum.StrEnum):
    CONTRACT_DEFINITION = "contract_definition"
    MATERIALIZATION = "materialization"
    ACTIVATION = "activation"
    EXECUTION_PINNING = "execution_pinning"
    EXECUTION_ADMISSION = "execution_admission"
    PERSISTENCE = "persistence"
    INSPECTION_PROJECTION = "inspection_projection"


class ConfigurationProvenanceSurfaceKind(enum.StrEnum):
    CONTRACT_DEFINITION = "contract_definition"
    REALIZATION_CORE = "realization_core"
    STRATEGY_OUTPUT = "strategy_output"
    INVARIANT_VALIDATION = "invariant_validation"


class ProvenanceDisposition(enum.StrEnum):
    SUPPORTED_CURRENT_HEAD = "SUPPORTED_CURRENT_HEAD"
    PARTIAL_CURRENT_HEAD = "PARTIAL_CURRENT_HEAD"
    GAP_REQUIRES_CHILD = "GAP_REQUIRES_CHILD"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DiscoveryCandidateDisposition(enum.StrEnum):
    APPLICABLE = "APPLICABLE"
    NOT_PROVENANCE = "NOT_PROVENANCE"
    TEST_OR_DIAGNOSTIC_ONLY = "TEST_OR_DIAGNOSTIC_ONLY"
    LEGACY_NON_PRODUCTION = "LEGACY_NON_PRODUCTION"


@dataclass(frozen=True, slots=True)
class ProvenanceJoin:
    domain: ProvenanceDomain
    join_id: str
    execution_key: str
    provenance_key: str
    typed_contract: str
    semantic_owner: str
    heuristic: bool = False


@dataclass(frozen=True, slots=True)
class ProvenanceGap:
    gap_id: str
    domain: ProvenanceDomain
    frz_criteria: tuple[str, ...]
    classification: Literal[
        "IN-SCOPE BLOCKER",
        "TRACKED FREEZE DEBT",
        "ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED",
    ]
    summary: str
    proposed_child: str | None = None


@dataclass(frozen=True, slots=True)
class ClassifiedDiscoveryCandidate:
    path: str
    surface_id: str
    disposition: DiscoveryCandidateDisposition
    reason: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.surface_id)


@dataclass(frozen=True, slots=True)
class RegisteredPolicyProvenanceSurface:
    path: str
    surface_id: str
    kind: PolicyProvenanceSurfaceKind
    semantic_owner: str
    canonical_contract: str
    execution_join: str
    evidence_nodeid: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.surface_id)


@dataclass(frozen=True, slots=True)
class RegisteredProfileRevisionSurface:
    path: str
    surface_id: str
    kind: ProfileRevisionSurfaceKind
    semantic_owner: str
    canonical_contract: str
    execution_join: str
    evidence_nodeid: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.surface_id)


@dataclass(frozen=True, slots=True)
class RegisteredConfigurationProvenanceSurface:
    path: str
    surface_id: str
    kind: ConfigurationProvenanceSurfaceKind
    semantic_owner: str
    canonical_contract: str
    execution_join: str
    evidence_nodeid: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.surface_id)


__all__ = [
    "ClassifiedDiscoveryCandidate",
    "ConfigurationProvenanceSurfaceKind",
    "DiscoveryCandidateDisposition",
    "PolicyProvenanceSurfaceKind",
    "ProfileRevisionSurfaceKind",
    "ProvenanceDisposition",
    "ProvenanceDomain",
    "ProvenanceGap",
    "ProvenanceJoin",
    "RegisteredConfigurationProvenanceSurface",
    "RegisteredPolicyProvenanceSurface",
    "RegisteredProfileRevisionSurface",
    "SurfaceParityResult",
    "compare_discovered_to_registry",
]
