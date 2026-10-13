# © Artur Czarnecki. All rights reserved.

"""Evidence-based disposition for the eight known PROD-Q-P0 unresolved/composition concerns."""

from __future__ import annotations

import hashlib
from functools import lru_cache

from tests.qualification.prod_q._p0_discovery import repo_root
from tests.qualification.prod_q._p0_factory_ast_evidence import analyze_sandbox_factory_ast_evidence
from tests.qualification.prod_q._p0_types import (
    ConcernFindingReconciliationMetrics,
    ProdQFindingDisposition,
    ProdQKnownFinding,
    ProductionReadinessClassification,
    ProductionReadinessFamily,
    SemanticConcernRecord,
    SemanticConcernResolutionState,
)

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
def historical_prod_q_findings() -> tuple[ProdQKnownFinding, ...]:
    return tuple(
        ProdQKnownFinding(
            finding_id=item.finding_id,
            concern_key=item.concern_key,
            disposition=item.disposition,
            evidence_refs=item.evidence_refs,
            recommended_child=item.recommended_child,
            lineage_only=True,
        )
        for item in _KNOWN_FINDINGS
    )


def historical_finding_for_concern(concern_key: str) -> ProdQKnownFinding | None:
    for item in historical_prod_q_findings():
        if item.concern_key == concern_key:
            return item
    return None


_FAMILY_UNRESOLVED_DISPOSITION: dict[
    ProductionReadinessFamily,
    tuple[ProdQFindingDisposition, str],
] = {
    ProductionReadinessFamily.PQ_F07: (
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        "PROD-Q-P3",
    ),
    ProductionReadinessFamily.PQ_F10: (
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        "PROD-Q-P2",
    ),
    ProductionReadinessFamily.PQ_F12: (
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        "PROD-Q-P3",
    ),
    ProductionReadinessFamily.PQ_F13: (
        ProdQFindingDisposition.IN_SCOPE_BLOCKER,
        "PROD-Q-P1",
    ),
    ProductionReadinessFamily.PQ_F14: (
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        "PROD-Q-P3",
    ),
    ProductionReadinessFamily.PQ_F15: (
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        "PROD-Q-P3",
    ),
    ProductionReadinessFamily.PQ_F16: (
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        "PROD-Q-P3",
    ),
    ProductionReadinessFamily.PQ_F17: (
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        "PROD-Q-P2",
    ),
    ProductionReadinessFamily.PQ_F19: (
        ProdQFindingDisposition.IN_SCOPE_BLOCKER,
        "PROD-Q-P1",
    ),
    ProductionReadinessFamily.PQ_F20: (
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        "PROD-Q-P3",
    ),
    ProductionReadinessFamily.PQ_F22: (
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        "PROD-Q-P3",
    ),
    ProductionReadinessFamily.PQ_F23: (
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        "PROD-Q-P3",
    ),
}


def _pq_support_family(concern_key: str) -> ProductionReadinessFamily | None:
    if not concern_key.startswith("concern:pq_support:"):
        return None
    token = concern_key.split(":", 3)[2]
    try:
        return ProductionReadinessFamily(token)
    except ValueError:
        return None


def disposition_for_unresolved_concern(
    concern_key: str,
    family: ProductionReadinessFamily,
    members: tuple[ProductionReadinessClassification, ...],
) -> tuple[ProdQFindingDisposition, str | None]:
    if concern_key.endswith(":factory_anchor"):
        return ProdQFindingDisposition.IN_SCOPE_BLOCKER, "PROD-Q-P1"
    if "sandbox_provider" in concern_key and "factory_anchor" not in concern_key:
        return ProdQFindingDisposition.IN_SCOPE_BLOCKER, "PROD-Q-P1"
    if family in (ProductionReadinessFamily.PQ_F02, ProductionReadinessFamily.PQ_F18):
        return ProdQFindingDisposition.IN_SCOPE_BLOCKER, "PROD-Q-R1"
    if family == ProductionReadinessFamily.PQ_F11:
        return (
            ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
            "PROD-Q-P2",
        )
    if family in (
        ProductionReadinessFamily.PQ_F03,
        ProductionReadinessFamily.PQ_F04,
        ProductionReadinessFamily.PQ_F05,
        ProductionReadinessFamily.PQ_F06,
        ProductionReadinessFamily.PQ_F08,
    ):
        return ProdQFindingDisposition.TRACKED_FREEZE_DEBT, "PROD-Q-P3"
    if family == ProductionReadinessFamily.PQ_F09:
        return ProdQFindingDisposition.IN_SCOPE_BLOCKER, "PROD-Q-P1"
    if any(item.evidence_state.name == "BLOCKER" for item in members):
        return ProdQFindingDisposition.IN_SCOPE_BLOCKER, "PROD-Q-P2"
    pq_support = _pq_support_family(concern_key)
    if pq_support is not None:
        return disposition_for_unresolved_concern(
            concern_key,
            pq_support,
            members,
        )
    explicit = _FAMILY_UNRESOLVED_DISPOSITION.get(family)
    if explicit is not None:
        return explicit
    return ProdQFindingDisposition.UNRESOLVED_DISPOSITION, None


def _auto_finding_id(concern_key: str) -> str:
    digest = hashlib.sha256(concern_key.encode("utf-8")).hexdigest()[:12]
    return f"PROD-Q-P0-CONCERN-{digest}"


_CANONICAL_SANDBOX_HOST_INTEGRATION_CONTRACT = (
    "intergrax.runtime.integrations.categories.devops.SandboxHostIntegrationContract"
)
_UNRESOLVED_COMPOSITION_MARKERS = frozenset({"COMPOSITION_UNRESOLVED", ""})

_SANDBOX_FACTORY_PROVIDER_ARTIFACTS: dict[str, tuple[str, str, str, str]] = {
    "e2b": (
        "intergrax/integrations/providers/sandbox_host/e2b/contract_spec.py",
        "intergrax/integrations/providers/sandbox_host/e2b/integration.py",
        "intergrax/integrations/providers/sandbox_host/e2b/register.py",
        "E2bSandboxHostIntegration",
    ),
    "modal": (
        "intergrax/integrations/providers/sandbox_host/modal/contract_spec.py",
        "intergrax/integrations/providers/sandbox_host/modal/integration.py",
        "intergrax/integrations/providers/sandbox_host/modal/register.py",
        "ModalSandboxHostIntegration",
    ),
    "daytona": (
        "intergrax/integrations/providers/sandbox_host/daytona/contract_spec.py",
        "intergrax/integrations/providers/sandbox_host/daytona/integration.py",
        "intergrax/integrations/providers/sandbox_host/daytona/register.py",
        "DaytonaSandboxHostIntegration",
    ),
}


def _sandbox_provider_key(module_path: str) -> str | None:
    for provider in _SANDBOX_FACTORY_PROVIDER_ARTIFACTS:
        if f"/sandbox_host/{provider}/" in module_path:
            return provider
    return None


def factory_anchor_ownership_current_head_evidence(
    members: tuple[ProductionReadinessClassification, ...],
) -> tuple[str, ...] | None:
    """FACTORY_ANCHOR_IDENTIFIED + OWNERSHIP_RESOLVED evidence, or None (fail-closed)."""
    if not members:
        return None
    member = members[0]
    if len(members) != 1:
        return None
    if not any(ref.startswith("discovery:anchor.registry") for ref in member.current_head_evidence):
        return None
    provider = _sandbox_provider_key(member.module)
    if provider is None:
        return None
    if not member.symbol.startswith("create_") or not member.symbol.endswith("_sandbox_host_integration"):
        return None

    contract_spec_path, integration_path, register_path, _integration_class = (
        _SANDBOX_FACTORY_PROVIDER_ARTIFACTS[provider]
    )
    root = repo_root()
    expected_bundle = f"intergrax/integrations/providers/sandbox_host/{provider}/bundle.py"
    if member.module != expected_bundle:
        return None

    ast_result = analyze_sandbox_factory_ast_evidence(
        root,
        provider,
        member.symbol,
        member.module,
    )
    if not ast_result.proven:
        return None

    sanctioned_composition = {
        ast_result.contract_declaration_owner,
        ast_result.integration_implementation_owner,
        ast_result.registration_composition_owner,
    }
    if (
        member.composition_owner not in _UNRESOLVED_COMPOSITION_MARKERS
        and member.composition_owner not in sanctioned_composition
    ):
        return None

    return (
        f"factory_identified:{member.module}:{member.symbol}",
        f"canonical_contract:{_CANONICAL_SANDBOX_HOST_INTEGRATION_CONTRACT}",
        "semantic_owner:sandbox_host_provider",
        f"contract_declaration_owner:{ast_result.contract_declaration_owner}",
        f"integration_implementation_owner:{ast_result.integration_implementation_owner}",
        f"registration_composition_owner:{ast_result.registration_composition_owner}",
        *ast_result.evidence_refs,
        "ownership_assertion:FACTORY_ANCHOR_IDENTIFIED",
        "ownership_assertion:OWNERSHIP_RESOLVED",
        "runtime_registration:NOT_PROVEN",
        "production_qualification:PROD-Q-P1_required",
    )


def _factory_anchor_currently_resolved(
    members: tuple[ProductionReadinessClassification, ...],
) -> bool:
    return factory_anchor_ownership_current_head_evidence(members) is not None


def _active_findings_by_concern(
    findings: tuple[ProdQKnownFinding, ...],
) -> dict[str, ProdQKnownFinding]:
    index: dict[str, ProdQKnownFinding] = {}
    for finding in findings:
        existing = index.get(finding.concern_key)
        if existing is None:
            index[finding.concern_key] = finding
            continue
        if existing.lineage_only and not finding.lineage_only:
            index[finding.concern_key] = finding
    return index


def _legal_distinct_composition_currently_supported(
    concern_key: str,
    members: tuple[ProductionReadinessClassification, ...],
) -> bool:
    if len(members) < 2:
        return False
    owners = {item.semantic_owner for item in members}
    contracts = {item.canonical_contract for item in members}
    compositions = {item.composition_owner for item in members}
    if len(owners) != 1 or len(contracts) != 1:
        return False
    if len(compositions) <= 1:
        return False
    if concern_key == (
        "concern:operation:contract:intergrax.contracts.execution_capacity_admission."
        "ExecutionCapacityAdmissionPort"
    ):
        return (
            "intergrax.contracts.execution_capacity_admission.ExecutionCapacityAdmissionPort"
            in contracts
        )
    if concern_key == "concern:operation:contract:intergrax.runtime.observability.auditability_health":
        allowed = {
            "intergrax/applications/_shared/auditability_health_wiring.py",
            "intergrax/runtime/observability/auditability_health.py",
        }
        return (
            "intergrax.runtime.observability.auditability_health" in contracts
            and compositions <= allowed
            and len(compositions) >= 2
        )
    return False


def build_semantic_concern_records(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> tuple[SemanticConcernRecord, ...]:
    from tests.qualification.prod_q._p0_classification import classified_inventory
    from tests.qualification.prod_q._p0_ownership import (
        concern_identity_for_ownership,
        is_auxiliary_discovery,
        is_semantic_mechanism,
        ownership_resolved_with_evidence,
    )

    inv = inventory if inventory is not None else classified_inventory()
    grouped: dict[str, list[ProductionReadinessClassification]] = {}
    for record in inv:
        if is_auxiliary_discovery(record) or not is_semantic_mechanism(record):
            continue
        key = concern_identity_for_ownership(record)
        grouped.setdefault(key, []).append(record)

    records: list[SemanticConcernRecord] = []
    for concern_key, members in sorted(grouped.items()):
        member_ids = tuple(sorted(item.candidate_id for item in members))
        sample = members[0]
        historical = historical_finding_for_concern(concern_key)
        factory_evidence: tuple[str, ...] = ()
        factory_resolved = False
        if concern_key.endswith(":factory_anchor"):
            factory_evidence = factory_anchor_ownership_current_head_evidence(tuple(members)) or ()
            factory_resolved = bool(factory_evidence)

        source_evidence = tuple(sample.current_head_evidence)
        if factory_evidence:
            source_evidence = source_evidence + factory_evidence
        if historical is not None:
            source_evidence = source_evidence + (
                f"historical_lineage:{historical.finding_id}:{historical.disposition.value}",
            )

        resolved = all(ownership_resolved_with_evidence(item) for item in members)
        legal_distinct = _legal_distinct_composition_currently_supported(concern_key, tuple(members))

        if legal_distinct:
            resolution = SemanticConcernResolutionState.LEGAL_DISTINCT_COMPOSITION
            disposition = ProdQFindingDisposition.LEGAL_DISTINCT_COMPOSITION
            mandatory_child = None
        elif factory_resolved or resolved:
            resolution = SemanticConcernResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE
            disposition = ProdQFindingDisposition.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE
            mandatory_child = None
        else:
            resolution = SemanticConcernResolutionState.UNRESOLVED
            disposition, mandatory_child = disposition_for_unresolved_concern(
                concern_key,
                sample.family,
                tuple(members),
            )
        records.append(
            SemanticConcernRecord(
                concern_id=concern_key,
                concern_key=concern_key,
                member_candidate_ids=member_ids,
                semantic_responsibility=sample.responsibility,
                canonical_owner=sample.semantic_owner.value,
                canonical_contract=sample.canonical_contract,
                composition_owner=sample.composition_owner,
                production_callers=sample.production_callers,
                source_evidence=source_evidence,
                resolution_state=resolution,
                finding_disposition=disposition,
                mandatory_child=mandatory_child,
            )
        )
    return tuple(records)


def build_reconciled_prod_q_findings(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> tuple[ProdQKnownFinding, ...]:
    concerns = build_semantic_concern_records(inventory)
    findings: list[ProdQKnownFinding] = list(historical_prod_q_findings())
    for concern in concerns:
        if concern.resolution_state != SemanticConcernResolutionState.UNRESOLVED:
            continue
        if concern.finding_disposition is None:
            continue
        findings.append(
            ProdQKnownFinding(
                finding_id=_auto_finding_id(concern.concern_key),
                concern_key=concern.concern_key,
                disposition=concern.finding_disposition,
                evidence_refs=(
                    f"concern_record:{concern.concern_key}",
                    f"member_candidates:{len(concern.member_candidate_ids)}",
                    f"source:{concern.source_evidence[0]}",
                ),
                recommended_child=concern.mandatory_child,
                lineage_only=False,
            )
        )
    return tuple(findings)


@lru_cache(maxsize=1)
def known_prod_q_findings() -> tuple[ProdQKnownFinding, ...]:
    return build_reconciled_prod_q_findings()


def known_finding_ids() -> frozenset[str]:
    return frozenset(item.finding_id for item in known_prod_q_findings())


def unjustified_future_debt_violations(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> tuple[str, ...]:
    from tests.qualification.prod_q._p0_classification import classified_inventory

    inv = inventory if inventory is not None else classified_inventory()
    concerns = build_semantic_concern_records(inv)
    violations: list[str] = []
    for concern in concerns:
        if concern.resolution_state != SemanticConcernResolutionState.UNRESOLVED:
            continue
        if concern.finding_disposition != ProdQFindingDisposition.TRACKED_FREEZE_DEBT:
            continue
        if concern.mandatory_child is None:
            violations.append(f"{concern.concern_key}: future debt without mandatory child")
        if concern.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER:
            violations.append(f"{concern.concern_key}: production blocker labeled future debt")
    return tuple(violations)


def historical_resolution_without_current_proof_violations(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> tuple[str, ...]:
    from tests.qualification.prod_q._p0_classification import classified_inventory

    inv = inventory if inventory is not None else classified_inventory()
    concerns = build_semantic_concern_records(inv)
    findings = build_reconciled_prod_q_findings(inv)
    active_by_concern = _active_findings_by_concern(findings)
    violations: list[str] = []
    for concern in concerns:
        historical = historical_finding_for_concern(concern.concern_key)
        if historical is None:
            continue
        if historical.disposition not in (
            ProdQFindingDisposition.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
            ProdQFindingDisposition.LEGAL_DISTINCT_COMPOSITION,
        ):
            continue
        if concern.resolution_state != SemanticConcernResolutionState.UNRESOLVED:
            continue
        active = active_by_concern.get(concern.concern_key)
        if active is None or active.lineage_only:
            violations.append(
                f"{concern.concern_key}: historical resolution without current-HEAD proof"
            )
            continue
        if active.disposition in (
            ProdQFindingDisposition.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
            ProdQFindingDisposition.LEGAL_DISTINCT_COMPOSITION,
        ):
            violations.append(
                f"{concern.concern_key}: historical resolution masks current unresolved concern"
            )
    return tuple(violations)


def factory_ownership_resolved_without_proof_violations(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> tuple[str, ...]:
    from tests.qualification.prod_q._p0_classification import classified_inventory

    inv = inventory if inventory is not None else classified_inventory()
    inv_by_id = {record.candidate_id: record for record in inv}
    concerns = build_semantic_concern_records(inv)
    violations: list[str] = []
    for concern in concerns:
        if not concern.concern_key.endswith(":factory_anchor"):
            continue
        if concern.resolution_state != SemanticConcernResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE:
            continue
        member_records = tuple(
            inv_by_id[member_id]
            for member_id in concern.member_candidate_ids
            if member_id in inv_by_id
        )
        if not member_records:
            violations.append(f"{concern.concern_key}: factory anchor resolved without members")
            continue
        if factory_anchor_ownership_current_head_evidence(member_records) is None:
            violations.append(f"{concern.concern_key}: factory OWNERSHIP_RESOLVED without proof")
    return tuple(violations)


def unresolved_disposition_violations(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> tuple[str, ...]:
    concerns = build_semantic_concern_records(inventory)
    return tuple(
        f"{concern.concern_key}: unresolved disposition"
        for concern in concerns
        if concern.resolution_state == SemanticConcernResolutionState.UNRESOLVED
        and concern.finding_disposition == ProdQFindingDisposition.UNRESOLVED_DISPOSITION
    )


def concern_finding_reconciliation_violations(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
    findings: tuple[ProdQKnownFinding, ...] | None = None,
) -> tuple[str, ...]:
    from tests.qualification.prod_q._p0_classification import classified_inventory

    inv = inventory if inventory is not None else classified_inventory()
    concerns = build_semantic_concern_records(inv)
    if findings is None:
        findings = build_reconciled_prod_q_findings(inv)
    violations: list[str] = []

    unresolved_keys = {
        concern.concern_key
        for concern in concerns
        if concern.resolution_state == SemanticConcernResolutionState.UNRESOLVED
    }
    active_findings = tuple(finding for finding in findings if not finding.lineage_only)
    finding_by_concern = _active_findings_by_concern(findings)
    active_by_concern = {finding.concern_key: finding for finding in active_findings}
    if len(active_by_concern) != len(active_findings):
        violations.append("duplicate active finding disposition for one or more concern keys")

    covered_unresolved = {
        key
        for key in unresolved_keys
        if key in active_by_concern
        and active_by_concern[key].disposition
        not in (
            ProdQFindingDisposition.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
            ProdQFindingDisposition.LEGAL_DISTINCT_COMPOSITION,
        )
    }
    if covered_unresolved != unresolved_keys:
        missing = sorted(unresolved_keys - covered_unresolved)
        violations.append(f"missing finding coverage for unresolved concerns: {missing[:3]}")

    orphan = [
        finding.finding_id
        for finding in findings
        if finding.concern_key not in {concern.concern_key for concern in concerns}
        and not finding.lineage_only
    ]
    if orphan:
        violations.append(f"orphan findings: {orphan[:3]}")

    concern_by_key = {concern.concern_key: concern for concern in concerns}
    for finding in findings:
        concern = concern_by_key.get(finding.concern_key)
        if concern is None or finding.lineage_only:
            continue
        for ref in finding.evidence_refs:
            if ref.startswith("member_candidates:"):
                recorded = int(ref.split(":", 1)[1])
                if recorded != len(concern.member_candidate_ids):
                    violations.append(
                        f"{finding.concern_key}: finding member coverage stale ({recorded} != {len(concern.member_candidate_ids)})"
                    )

    inv_by_id = {record.candidate_id: record for record in inv}
    for concern in concerns:
        if concern.resolution_state != SemanticConcernResolutionState.UNRESOLVED:
            continue
        for member_id in concern.member_candidate_ids:
            if member_id not in inv_by_id:
                violations.append(
                    f"{concern.concern_key}: unresolved member {member_id} missing from inventory"
                )
        finding = active_by_concern.get(concern.concern_key)
        if finding is None:
            continue
        if (
            finding.disposition == ProdQFindingDisposition.TRACKED_FREEZE_DEBT
            and concern.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER
        ):
            violations.append(
                f"{concern.concern_key}: current blocker masked as future-stage debt"
            )
        if finding.disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER and (
            not concern.mandatory_child or not finding.recommended_child
        ):
            violations.append(f"{concern.concern_key}: in-scope blocker missing mandatory child")
        if concern.finding_disposition == ProdQFindingDisposition.UNRESOLVED_DISPOSITION:
            violations.append(f"{concern.concern_key}: unresolved disposition blocks P0 inventory")

    violations.extend(unjustified_future_debt_violations(inv))
    violations.extend(historical_resolution_without_current_proof_violations(inv))
    violations.extend(unresolved_disposition_violations(inv))
    violations.extend(factory_ownership_resolved_without_proof_violations(inv))

    return tuple(violations)


def concern_finding_pair_violation(
    concern: SemanticConcernRecord,
    finding: ProdQKnownFinding,
) -> str | None:
    if concern.resolution_state != SemanticConcernResolutionState.UNRESOLVED:
        return None
    if (
        finding.disposition == ProdQFindingDisposition.TRACKED_FREEZE_DEBT
        and concern.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER
    ):
        return f"{concern.concern_key}: current blocker masked as future-stage debt"
    return None


def build_concern_finding_metrics(
    inventory: tuple[ProductionReadinessClassification, ...] | None = None,
) -> ConcernFindingReconciliationMetrics:
    from tests.qualification.prod_q._p0_classification import classified_inventory

    inv = inventory if inventory is not None else classified_inventory()
    concerns = build_semantic_concern_records(inv)
    findings = build_reconciled_prod_q_findings(inv)
    unresolved = [concern for concern in concerns if concern.resolution_state == SemanticConcernResolutionState.UNRESOLVED]
    resolved = len(concerns) - len(unresolved)
    finding_by_concern = {finding.concern_key: finding for finding in findings}
    duplicate_dispositions = len(findings) - len(finding_by_concern)
    covered = sum(1 for concern in unresolved if concern.concern_key in finding_by_concern)
    missing = len(unresolved) - covered
    orphan = sum(
        1
        for finding in findings
        if finding.concern_key not in {concern.concern_key for concern in concerns}
        and not finding.lineage_only
    )
    return ConcernFindingReconciliationMetrics(
        concerns_discovered=len(concerns),
        concerns_resolved=resolved,
        concerns_unresolved=len(unresolved),
        unresolved_candidate_members=sum(len(concern.member_candidate_ids) for concern in unresolved),
        concerns_with_dispositions=covered,
        concerns_missing_dispositions=missing,
        orphan_findings=orphan,
        missing_findings=missing,
        duplicate_dispositions=duplicate_dispositions,
    )
