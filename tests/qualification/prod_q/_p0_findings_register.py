# © Artur Czarnecki. All rights reserved.

"""Evidence-based disposition for the eight known PROD-Q-P0 unresolved/composition concerns."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.prod_q._p0_types import ProdQFindingDisposition, ProdQKnownFinding

_KNOWN_FINDINGS: tuple[ProdQKnownFinding, ...] = (
    ProdQKnownFinding(
        finding_id="PROD-Q-P0-SANDBOX-E2B",
        concern_key="concern:operation:PQ-F09:sandbox_provider:e2b",
        disposition=ProdQFindingDisposition.IN_SCOPE_BLOCKER,
        evidence_refs=(
            "intergrax/integrations/providers/sandbox_host/e2b/bundle.py",
            "build_provider_dispositions:E2B:EXTERNAL_PROVIDER_EVIDENCE_REQUIRED",
            "gate:provider_registration_layers_separate_from_live_qualification",
        ),
        recommended_child="PROD-Q-P1",
    ),
    ProdQKnownFinding(
        finding_id="PROD-Q-P0-SANDBOX-MODAL",
        concern_key="concern:operation:PQ-F09:sandbox_provider:modal",
        disposition=ProdQFindingDisposition.IN_SCOPE_BLOCKER,
        evidence_refs=(
            "intergrax/integrations/providers/sandbox_host/modal/bundle.py",
            "build_provider_dispositions:Modal:EXTERNAL_PROVIDER_EVIDENCE_REQUIRED",
            "gate:provider_registration_layers_separate_from_live_qualification",
        ),
        recommended_child="PROD-Q-P1",
    ),
    ProdQKnownFinding(
        finding_id="PROD-Q-P0-SANDBOX-DAYTONA",
        concern_key="concern:operation:PQ-F09:sandbox_provider:daytona",
        disposition=ProdQFindingDisposition.IN_SCOPE_BLOCKER,
        evidence_refs=(
            "intergrax/integrations/providers/sandbox_host/daytona/bundle.py",
            "build_provider_dispositions:Daytona:EXTERNAL_PROVIDER_EVIDENCE_REQUIRED",
            "gate:provider_registration_layers_separate_from_live_qualification",
        ),
        recommended_child="PROD-Q-P1",
    ),
    ProdQKnownFinding(
        finding_id="PROD-Q-P0-BUNDLE-ANCHOR-E2B",
        concern_key="concern:operation:PQ-F09:sandbox_provider:e2b:factory_anchor",
        disposition=ProdQFindingDisposition.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
        evidence_refs=(
            "anchor:intergrax/integrations/providers/sandbox_host/e2b/bundle.py:create_e2b_sandbox_host_integration",
            "layered_registration:implementation_exists+registration_callable_exists",
            "ownership:semantic_mechanism+NOT_YET_RESOLVED_contract",
        ),
        recommended_child=None,
    ),
    ProdQKnownFinding(
        finding_id="PROD-Q-P0-BUNDLE-ANCHOR-MODAL",
        concern_key="concern:operation:PQ-F09:sandbox_provider:modal:factory_anchor",
        disposition=ProdQFindingDisposition.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
        evidence_refs=(
            "anchor:intergrax/integrations/providers/sandbox_host/modal/bundle.py:create_modal_sandbox_host_integration",
            "layered_registration:implementation_exists+registration_callable_exists",
            "ownership:semantic_mechanism+NOT_YET_RESOLVED_contract",
        ),
        recommended_child=None,
    ),
    ProdQKnownFinding(
        finding_id="PROD-Q-P0-BUNDLE-ANCHOR-DAYTONA",
        concern_key="concern:operation:PQ-F09:sandbox_provider:daytona:factory_anchor",
        disposition=ProdQFindingDisposition.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
        evidence_refs=(
            "anchor:intergrax/integrations/providers/sandbox_host/daytona/bundle.py:create_daytona_sandbox_host_integration",
            "layered_registration:implementation_exists+registration_callable_exists",
            "ownership:semantic_mechanism+NOT_YET_RESOLVED_contract",
        ),
        recommended_child=None,
    ),
    ProdQKnownFinding(
        finding_id="PROD-Q-P0-COMPOSITION-EXEC-CAPACITY-PORT",
        concern_key="concern:operation:contract:intergrax.contracts.execution_capacity_admission.ExecutionCapacityAdmissionPort",
        disposition=ProdQFindingDisposition.LEGAL_DISTINCT_COMPOSITION,
        evidence_refs=(
            "contract:intergrax.contracts.execution_capacity_admission.ExecutionCapacityAdmissionPort",
            "composition:intergrax/contracts/execution_capacity/admission_decision.py",
            "composition:intergrax/runtime/execution/capacity.py",
            "composition:intergrax/runtime/execution/local_execution_capacity_admission.py",
            "gate:single_semantic_owner_multiple_sanctioned_implementation_providers",
        ),
        recommended_child=None,
    ),
    ProdQKnownFinding(
        finding_id="PROD-Q-P0-COMPOSITION-AUDITABILITY-HEALTH",
        concern_key="concern:operation:contract:intergrax.runtime.observability.auditability_health",
        disposition=ProdQFindingDisposition.LEGAL_DISTINCT_COMPOSITION,
        evidence_refs=(
            "contract:intergrax.runtime.observability.auditability_health",
            "composition:intergrax/applications/_shared/auditability_health_wiring.py",
            "composition:intergrax/runtime/observability/auditability_health.py",
            "gate:host_wiring_vs_runtime_snapshot_builder_distinct_composition_loci",
        ),
        recommended_child=None,
    ),
)


@lru_cache(maxsize=1)
def known_prod_q_findings() -> tuple[ProdQKnownFinding, ...]:
    return _KNOWN_FINDINGS


def known_finding_ids() -> frozenset[str]:
    return frozenset(item.finding_id for item in _KNOWN_FINDINGS)
