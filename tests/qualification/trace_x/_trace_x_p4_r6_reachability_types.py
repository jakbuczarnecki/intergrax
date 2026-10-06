# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R6 mechanical reachability qualification types."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ReachabilityVerdict(enum.StrEnum):
    PRODUCTION_REACHABLE = "production_reachable"
    NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT = "not_reachable_from_sanctioned_production_root"
    TEST_OR_QUALIFICATION_ONLY = "test_or_qualification_only"
    DEVELOPMENT_LAB_ONLY = "development_lab_only"
    UNRESOLVED = "unresolved"


class ReachabilityReason(enum.StrEnum):
    TIER2_AGENT_MODULE = "tier2_agent_module"
    TIER3_APPLICATION_MODULE = "tier3_application_module"
    MODULE_OUTSIDE_SANCTIONED_IMPORT_CLOSURE = "module_outside_sanctioned_import_closure"
    CONSUMER_NOT_INSTANTIATED_FROM_SANCTIONED_WIRING = "consumer_not_instantiated_from_sanctioned_wiring"
    INFERENCE_EXECUTOR_NO_PRODUCTION_CALLER = "inference_executor_no_production_caller"
    INFERENCE_EXECUTOR_PRODUCTION_WIRED = "inference_executor_production_wired"
    WRAPPER_STREAM_NOT_INVOKED_FROM_SANCTIONED_COMPOSITION = (
        "wrapper_stream_not_invoked_from_sanctioned_composition"
    )
    SYNTHETIC_PRODUCTION_COMPOSITION_EDGE = "synthetic_production_composition_edge"
    AMBIGUOUS_COMPOSITION_EDGE = "ambiguous_composition_edge"
    IMPORT_RESOLUTION_FAILED = "import_resolution_failed"


@dataclass(frozen=True, slots=True)
class ModelConsumerSurface:
    path: str
    method: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.method)


@dataclass(frozen=True, slots=True)
class CompositionEdge:
    source_module_path: str
    target_module_path: str
    edge_kind: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source_module_path, self.target_module_path, self.edge_kind)


@dataclass(frozen=True, slots=True)
class MechanicalReachabilityResult:
    surface: ModelConsumerSurface
    verdict: ReachabilityVerdict
    reason: ReachabilityReason
    reachable_from: frozenset[str] = frozenset()
    evidence_edges: tuple[CompositionEdge, ...] = ()


@dataclass(frozen=True, slots=True)
class ProductionReachabilityGraphSnapshot:
    sanctioned_seed_modules: frozenset[str]
    import_closure_modules: frozenset[str]
    composition_edges: frozenset[CompositionEdge]
    production_reachable_modules: frozenset[str]
    lab_reachable_modules: frozenset[str]
    unresolved_modules: frozenset[str]


@dataclass(frozen=True, slots=True)
class ReachabilityExpectationParityResult:
    duplicate_registry_keys: frozenset[tuple[str, str]]
    unknown: frozenset[tuple[str, str]]
    orphan: frozenset[tuple[str, str]]
    contradictions: frozenset[tuple[str, str]]
    unresolved: frozenset[tuple[str, str]]

    @property
    def ok(self) -> bool:
        return (
            not self.unknown
            and not self.orphan
            and not self.duplicate_registry_keys
            and not self.contradictions
            and not self.unresolved
        )


__all__ = [
    "CompositionEdge",
    "MechanicalReachabilityResult",
    "ModelConsumerSurface",
    "ProductionReachabilityGraphSnapshot",
    "ReachabilityExpectationParityResult",
    "ReachabilityReason",
    "ReachabilityVerdict",
]
