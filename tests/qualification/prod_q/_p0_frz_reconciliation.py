# © Artur Czarnecki. All rights reserved.

"""FRZ-PRD / FRZ-SEC / FRZ-REL reconciliation for PROD-Q-P0 (evidence state only — not PASS)."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.prod_q._p0_types import (
    FrzCriterionRecord,
    ProdQFindingDisposition,
    ProductionReadinessEvidenceState,
    ProductionReadinessFamily,
)

_REQUIRED_FRZ_IDS: tuple[str, ...] = (
    "FRZ-PRD-01",
    "FRZ-PRD-02",
    "FRZ-PRD-03",
    "FRZ-PRD-04",
    "FRZ-PRD-05",
    "FRZ-PRD-06",
    "FRZ-PRD-07",
    "FRZ-PRD-08",
    "FRZ-PRD-09",
    "FRZ-PRD-10",
    "FRZ-PRD-11",
    "FRZ-PRD-12",
    "FRZ-SEC-01",
    "FRZ-SEC-02",
    "FRZ-SEC-03",
    "FRZ-SEC-04",
    "FRZ-SEC-05",
    "FRZ-SEC-06",
    "FRZ-SEC-07",
    "FRZ-SEC-08",
    "FRZ-SEC-09",
    "FRZ-SEC-10",
    "FRZ-REL-01",
    "FRZ-REL-02",
    "FRZ-REL-03",
    "FRZ-REL-04",
    "FRZ-REL-05",
    "FRZ-REL-06",
    "FRZ-REL-07",
    "FRZ-REL-08",
    "FRZ-REL-09",
    "FRZ-REL-10",
    "FRZ-REL-11",
)

_FRZ_ROWS: tuple[tuple[str, str, str, ProductionReadinessEvidenceState, ProdQFindingDisposition, tuple[ProductionReadinessFamily, ...], tuple[str, ...]], ...] = (
    (
        "FRZ-PRD-01",
        "Production",
        "production qualification independent of lab/harness",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.IN_SCOPE_BLOCKER,
        (ProductionReadinessFamily.PQ_F01, ProductionReadinessFamily.PQ_F11),
        ("tests/qualification/prod_q/", "intergrax/runtime/execution/lab_reference_agent_runtime.py"),
    ),
    (
        "FRZ-PRD-02",
        "Production",
        "provider/plugin qualification enforced",
        ProductionReadinessEvidenceState.EXTERNAL_PROVIDER_EVIDENCE_REQUIRED,
        ProdQFindingDisposition.IN_SCOPE_BLOCKER,
        (ProductionReadinessFamily.PQ_F09,),
        ("intergrax/integrations/providers/sandbox_host/", "AW-7C reference only"),
    ),
    (
        "FRZ-PRD-03",
        "Production",
        "unsupported production configuration fails closed",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F10, ProductionReadinessFamily.PQ_F18),
        ("CONFIG-X closed evidence",),
    ),
    (
        "FRZ-PRD-04",
        "Production",
        "startup lifecycle qualified",
        ProductionReadinessEvidenceState.BLOCKER,
        ProdQFindingDisposition.IN_SCOPE_BLOCKER,
        (ProductionReadinessFamily.PQ_F02, ProductionReadinessFamily.PQ_F18),
        ("R1-SQLITE-ENV-01", "applications/governed_contractor_application/host/factory.py"),
    ),
    (
        "FRZ-PRD-05",
        "Production",
        "shutdown/resource cleanup qualified",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F03,),
        ("intergrax/contracts/execution_reliability/shutdown_contract.py",),
    ),
    (
        "FRZ-PRD-06",
        "Production",
        "degraded behavior explicit",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F06,),
        ("docs/project/maintainers/architecture/EXECUTION_ENGINE_OPERATIONAL_READINESS_SLO_HEALTH_MODEL.md",),
    ),
    (
        "FRZ-PRD-07",
        "Production",
        "strict vs lab semantics separated",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F11,),
        ("PROD-Q-P0 strict-vs-lab branch inventory",),
    ),
    (
        "FRZ-PRD-08",
        "Production",
        "production-only bypass count = 0",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.IN_SCOPE_BLOCKER,
        (ProductionReadinessFamily.PQ_F11,),
        ("tests/qualification/prod_q/_p0_discovery.py AST_STRICT_LAB_BRANCH",),
    ),
    (
        "FRZ-PRD-09",
        "Production",
        "production health/readiness/degraded-state semantics qualified",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (
            ProductionReadinessFamily.PQ_F04,
            ProductionReadinessFamily.PQ_F05,
            ProductionReadinessFamily.PQ_F06,
        ),
        ("intergrax/hosting/engine/engine.py", "auditability_health_wiring"),
    ),
    (
        "FRZ-PRD-10",
        "Production",
        "terminal and critical production failures operator-visible",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F07,),
        ("observability non-authoritative invariant",),
    ),
    (
        "FRZ-PRD-11",
        "Production",
        "production capacity/overload qualified and bounded",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (
            ProductionReadinessFamily.PQ_F13,
            ProductionReadinessFamily.PQ_F14,
        ),
        ("HARNESS-W4 f38e455", "intergrax/contracts/execution_capacity_admission.py"),
    ),
    (
        "FRZ-PRD-12",
        "Production",
        "critical operational recovery responsibility explicit",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F08, ProductionReadinessFamily.PQ_F23),
        ("STATE-X bd54941d",),
    ),
    (
        "FRZ-SEC-01",
        "Security",
        "tenant isolation proven",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F22,),
        ("TENANT-X 5ccb72fc", "FRZ-TEN-01..12 PASS — not reopened"),
    ),
    (
        "FRZ-SEC-02",
        "Security",
        "secrets references / late resolution",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F12,),
        ("CTRL-X 318a4d83",),
    ),
    (
        "FRZ-SEC-03",
        "Security",
        "raw secrets do not cross unrelated layers",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F12,),
        ("CTRL-X 318a4d83",),
    ),
    (
        "FRZ-SEC-04",
        "Security",
        "production auth bypass = 0",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F11,),
        ("CTRL-X 318a4d83",),
    ),
    (
        "FRZ-SEC-05",
        "Security",
        "sandbox/isolation boundaries qualified",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F19,),
        ("AW-7C-P0-3B-PHYSQ deaafee1 — scoped only",),
    ),
    (
        "FRZ-SEC-06",
        "Security",
        "provider/plugin trust qualified",
        ProductionReadinessEvidenceState.EXTERNAL_PROVIDER_EVIDENCE_REQUIRED,
        ProdQFindingDisposition.IN_SCOPE_BLOCKER,
        (ProductionReadinessFamily.PQ_F09,),
        ("E2B/Modal/Daytona",),
    ),
    (
        "FRZ-SEC-07",
        "Security",
        "security claim bounded by explicit threat model",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F19,),
        ("AW-7C-P0-3B-PHYSQ threat model bounds",),
    ),
    (
        "FRZ-SEC-08",
        "Security",
        "sensitive/tenant data propagation explicit",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F12, ProductionReadinessFamily.PQ_F22),
        ("CTRL-X 318a4d83",),
    ),
    (
        "FRZ-SEC-09",
        "Security",
        "retention/deletion/redaction responsibility explicit",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F20,),
        ("CTRL-X 318a4d83",),
    ),
    (
        "FRZ-SEC-10",
        "Security",
        "platform vs infrastructure encryption/residency responsibility",
        ProductionReadinessEvidenceState.N_A_WITH_EVIDENCE,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F21,),
        ("infrastructure operator ownership documented",),
    ),
    (
        "FRZ-REL-01",
        "Reliability",
        "timeout ownership explicit",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F16,),
        ("HARNESS-W4 f38e455", "CTRL-X"),
    ),
    (
        "FRZ-REL-02",
        "Reliability",
        "retry ownership explicit",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F16,),
        ("HARNESS-W4 f38e455",),
    ),
    (
        "FRZ-REL-03",
        "Reliability",
        "idempotency ownership explicit",
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F17,),
        ("STATE-X bd54941d",),
    ),
    (
        "FRZ-REL-04",
        "Reliability",
        "cancellation propagates correctly",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F16,),
        ("HARNESS-W4 W4-A f38e455",),
    ),
    (
        "FRZ-REL-05",
        "Reliability",
        "external operation termination semantics explicit",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F17,),
        ("HARNESS-W4 W4-C f38e455",),
    ),
    (
        "FRZ-REL-06",
        "Reliability",
        "provider failure typed",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F16,),
        ("HARNESS-W4 W4-D f38e455",),
    ),
    (
        "FRZ-REL-07",
        "Reliability",
        "no hidden infinite retry/fallback",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F16,),
        ("HARNESS-W4 f38e455",),
    ),
    (
        "FRZ-REL-08",
        "Reliability",
        "bounded concurrency and work-admission explicit",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F13,),
        ("LocalExecutionCapacityAdmission", "HARNESS-W4 f38e455"),
    ),
    (
        "FRZ-REL-09",
        "Reliability",
        "queue/backpressure and overload behavior explicit",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F14,),
        ("HARNESS-W4 f38e455",),
    ),
    (
        "FRZ-REL-10",
        "Reliability",
        "resource saturation deterministic reject/degrade/fail-closed",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F13,),
        ("HARNESS-W4 f38e455",),
    ),
    (
        "FRZ-REL-11",
        "Reliability",
        "provider throttling cannot create unbounded retry/bypass",
        ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
        ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
        (ProductionReadinessFamily.PQ_F15,),
        ("HARNESS-W4 f38e455",),
    ),
)


def _historical_stage_ancestry_ok(stage_token: str) -> bool:
    from tests.qualification.prod_q._p0_certification import HISTORICAL_EVIDENCE, current_git_head
    import subprocess
    from tests.qualification.prod_q._p0_discovery import repo_root

    head = current_git_head()
    for item in HISTORICAL_EVIDENCE:
        if item.stage_id.startswith(stage_token):
            proc = subprocess.run(
                ["git", "merge-base", "--is-ancestor", item.accepted_sha, head],
                cwd=repo_root(),
                capture_output=True,
            )
            return proc.returncode == 0
    return False


def _derived_frz_assessment(
    criterion_id: str,
    pq_families: tuple[ProductionReadinessFamily, ...],
    canonical_refs: tuple[str, ...],
) -> tuple[ProductionReadinessEvidenceState, ProdQFindingDisposition, tuple[str, ...]]:
    from tests.qualification.prod_q._p0_certification import (
        build_provider_dispositions,
        build_r1_sqlite_disposition,
    )
    from tests.qualification.prod_q._p0_classification import classified_inventory

    refs = list(canonical_refs)
    inventory = classified_inventory()
    family_states = {
        family: [r for r in inventory if r.family == family]
        for family in pq_families
    }

    if criterion_id == "FRZ-PRD-04":
        sqlite = build_r1_sqlite_disposition()
        refs.append(f"sqlite.reproduction:{sqlite.reproduction_classification.value}")
        if sqlite.reproduction_executed:
            refs.append(f"sqlite.exit_code:{sqlite.reproduction_exit_code}")
        return (
            ProductionReadinessEvidenceState.BLOCKER,
            ProdQFindingDisposition.IN_SCOPE_BLOCKER,
            tuple(refs),
        )

    if criterion_id in ("FRZ-PRD-02", "FRZ-SEC-06"):
        providers = build_provider_dispositions()
        refs.extend(f"provider:{p.provider_id}:registered={p.registered}" for p in providers)
        return (
            ProductionReadinessEvidenceState.EXTERNAL_PROVIDER_EVIDENCE_REQUIRED,
            ProdQFindingDisposition.IN_SCOPE_BLOCKER,
            tuple(refs),
        )

    if criterion_id == "FRZ-SEC-10":
        refs.append("PQ-F21 platform-vs-infrastructure responsibility")
        return (
            ProductionReadinessEvidenceState.N_A_WITH_EVIDENCE,
            ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
            tuple(refs),
        )

    blockers = [
        r
        for family in pq_families
        for r in family_states.get(family, ())
        if r.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER
    ]
    if blockers:
        refs.append(f"inventory.blockers:{len(blockers)}")
        return (
            ProductionReadinessEvidenceState.BLOCKER,
            ProdQFindingDisposition.IN_SCOPE_BLOCKER,
            tuple(refs),
        )

    for token in ("TENANT-X", "CTRL-X", "STATE-X", "HARNESS-W4", "CONFIG-X", "COMPAT-X"):
        if any(token in ref for ref in canonical_refs) and _historical_stage_ancestry_ok(token):
            refs.append(f"historical.ancestry:{token}")
            return (
                ProductionReadinessEvidenceState.SUPPORTING_HISTORICAL_EVIDENCE_REVALIDATED,
                ProdQFindingDisposition.TRACKED_FREEZE_DEBT,
                tuple(refs),
            )

    refs.append("derived.current_head.partial")
    disposition = (
        ProdQFindingDisposition.IN_SCOPE_BLOCKER
        if criterion_id == "FRZ-PRD-08"
        else ProdQFindingDisposition.TRACKED_FREEZE_DEBT
    )
    return (
        ProductionReadinessEvidenceState.PARTIAL_CURRENT_HEAD,
        disposition,
        tuple(refs),
    )


@lru_cache(maxsize=1)
def build_frz_reconciliation() -> tuple[FrzCriterionRecord, ...]:
    records: list[FrzCriterionRecord] = []
    for row in _FRZ_ROWS:
        evidence, disposition, refs = _derived_frz_assessment(row[0], row[5], row[6])
        records.append(
            FrzCriterionRecord(
                criterion_id=row[0],
                domain=row[1],
                summary=row[2],
                evidence_state=evidence,
                finding_disposition=disposition,
                pq_families=row[5],
                evidence_refs=refs,
            )
        )
    return tuple(records)


def required_frz_ids() -> tuple[str, ...]:
    return _REQUIRED_FRZ_IDS
