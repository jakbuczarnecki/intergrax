# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R5 production composition and reachability types."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ProductionCompositionSiteKind(enum.StrEnum):
    STRATEGY_EXECUTION_ROUTER_INSTANTIATION = "strategy_execution_router_instantiation"
    STRATEGY_ROUTER_INFERENCE_EXECUTOR_KW = "strategy_router_inference_executor_kw"
    RUNTIME_INFERENCE_EXECUTOR_INSTANTIATION = "runtime_inference_executor_instantiation"
    GOVERNED_INFERENCE_EXECUTOR_CALL = "governed_inference_executor_call"
    GOVERNED_INFERENCE_EXECUTOR_FACTORY_DEF = "governed_inference_executor_factory_def"
    SANCTIONED_LLM_ADAPTER_P4_WRAP = "sanctioned_llm_adapter_p4_wrap"
    RAW_LLM_ADAPTER_CONFIG_ASSIGNMENT = "raw_llm_adapter_config_assignment"
    RUNTIME_CONFIG_LLM_ADAPTER_KW = "runtime_config_llm_adapter_kw"


class ProductionCompositionSiteClassification(enum.StrEnum):
    SANCTIONED_PRODUCTION_ROOT = "sanctioned_production_root"
    SANCTIONED_PRODUCTION_ROUTER = "sanctioned_production_router"
    INTERNAL_FACTORY_NO_PRODUCTION_CALLER = "internal_factory_no_production_caller"
    DEVELOPMENT_LAB_OR_NON_STRICT_RUNTIME = "development_lab_or_non_strict_runtime"


class NonProductionReachabilityReason(enum.StrEnum):
    TIER2_AGENT_NOT_SANCTIONED_COMPOSITION_ROOT = "tier2_agent_not_sanctioned_composition_root"
    TIER3_APPLICATION_NOT_SANCTIONED_COMPOSITION_ROOT = (
        "tier3_application_not_sanctioned_composition_root"
    )
    AUXILIARY_NOT_REACHABLE_FROM_SANCTIONED_COMPOSITION = (
        "auxiliary_not_reachable_from_sanctioned_composition"
    )
    INTERNAL_INFERENCE_BACKEND_UNREACHABLE = "internal_inference_backend_unreachable"


@dataclass(frozen=True, slots=True)
class DiscoveredProductionCompositionSite:
    path: str
    site_kind: ProductionCompositionSiteKind
    line_number: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.path, self.site_kind, self.line_number)


@dataclass(frozen=True, slots=True)
class RegisteredProductionCompositionSite:
    path: str
    site_kind: ProductionCompositionSiteKind
    line_number: int
    classification: ProductionCompositionSiteClassification
    composition_anchor: str
    inference_executor_supplied: bool
    p4_wrap_applied: bool | None
    evidence_nodeid: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.path, self.site_kind, self.line_number)


@dataclass(frozen=True, slots=True)
class RegisteredNonProductionModelReachability:
    path: str
    method: str
    reason: NonProductionReachabilityReason
    reachability_proof: str
    evidence_nodeid: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.method)


__all__ = [
    "DiscoveredProductionCompositionSite",
    "NonProductionReachabilityReason",
    "ProductionCompositionSiteClassification",
    "ProductionCompositionSiteKind",
    "RegisteredNonProductionModelReachability",
    "RegisteredProductionCompositionSite",
]
