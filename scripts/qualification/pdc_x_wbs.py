# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E501  (WBS text data)

"""PDC-X P4 implementation design (T-IGX-036-G).

Reads ``PDC_DOMAIN_REQUIREMENTS_MATRIX.json`` and the checked-in ``BATCHES`` table, then
renders the implementation WBS, the dependency DAG and the cross-domain test matrix.

``--check`` fails when:

* a work-requiring requirement row (gap classes) is not assigned to exactly one batch;
* two batches of the same wave own the same write path (or one is a prefix of the other);
* more than one batch of a wave writes under a shared integration hotspot;
* a batch depends on an unknown batch, or the dependency graph has a cycle;
* a rendered output is stale.

Usage::

    uv run python scripts/qualification/pdc_x_wbs.py --write
    uv run python scripts/qualification/pdc_x_wbs.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Final, NamedTuple

REPO: Final[Path] = Path(__file__).resolve().parents[2]
DOSSIER: Final[Path] = REPO / "docs/project/maintainers/qualification/platform_domain_completeness"
MATRIX: Final[Path] = DOSSIER / "PDC_DOMAIN_REQUIREMENTS_MATRIX.json"
OUT_WBS: Final[Path] = DOSSIER / "PDC_IMPLEMENTATION_WBS.md"
OUT_DAG: Final[Path] = DOSSIER / "PDC_DEPENDENCY_DAG.md"
OUT_XD: Final[Path] = DOSSIER / "PDC_CROSS_DOMAIN_TEST_MATRIX.md"

GAP_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "IMPLEMENTATION_COMPLETION",
        "INTEGRATION_CORRECTION",
        "DOCUMENTATION_RECONCILIATION",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "BLOCKED_EXACT_REASON",
    }
)
# Shared integration hotspots: one integration owner (the PDC orchestrator) serializes writers,
# so at most one batch per wave may write under each prefix (PDC-P4-02).
HOTSPOTS: Final[tuple[str, ...]] = ("intergrax/applications/_shared/", "intergrax/contracts/")
INTEGRATION_OWNER: Final[str] = "PDC orchestrator session (single integration owner for shared hotspots)"
PRODUCT_PREFIXES: Final[tuple[str, ...]] = ("intergrax/", "agents/", "applications/")

GATE_PRODUCT: Final[str] = (
    "OD-PDC-SEQ-01: implement in an isolated worktree; merge only after MCP-SERVER-X P8 publication; ACT-N11 re-activation at PDC-P7-00"
)
GATE_FREE: Final[str] = "No product path: may merge before MCP-SERVER-X P8"
GATE_DONE: Final[str] = "Closed in PDC-P3 (ownership record)"
GATE_EXTERNAL: Final[str] = "External: owned by MCP-SERVER-X P8"


class Batch(NamedTuple):
    bid: str
    lane: str
    title: str
    ticket: str
    wave: str
    writes: tuple[str, ...]
    adr: str
    deps: tuple[str, ...]
    gate: str
    auditor: str
    tests: str


BATCHES: Final[tuple[Batch, ...]] = (
    Batch(
        "B-P3",
        "GOV",
        "Ownership records closed in P3",
        "T-IGX-036-D",
        "W0-done",
        (),
        "—",
        (),
        GATE_DONE,
        "IA of the owning domain (P7 replay)",
        "PDC_OWNERSHIP_MATRIX.md",
    ),
    Batch(
        "B-MCP",
        "EXT",
        "MCP exposure security (MCP-SERVER-X P8)",
        "MCP-SERVER-X",
        "W0-external",
        (),
        "—",
        (),
        GATE_EXTERNAL,
        "MCP-SERVER-X P8 auditor; PDC IA-D6 consumes the result",
        "MCP-SERVER-X qualification package",
    ),
    Batch(
        "B-DOC",
        "DOC",
        "Documentation reconciliations (wave A)",
        "T-IGX-036-H",
        "WA",
        (
            "docs/project/architecture/AGENT_DISTRIBUTION.md",
            "docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md",
            "docs/project/architecture/BACKGROUND_TASKS.md",
            "docs/project/architecture/intergrax_runtime_architecture.md",
            "docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md",
            "docs/project/maintainers/plans/CRITIC_VERIFICATION.md",
            "docs/project/maintainers/plans/BACKGROUND_TASKS.md",
            "docs/project/maintainers/qualification/HARNESS_FINAL_CURRENT_HEAD_CERTIFICATION.md",
        ),
        "—",
        (),
        GATE_FREE,
        "IA of the affected domain (doc delta at P7)",
        "check_docs_domain_pairs.py; pdc_x_inventory.py --check",
    ),
    Batch(
        "B-REG-T",
        "REG",
        "Stale-gate repair: ebh_5 (EBH-7 reclassification), GR-12 fixture authority, GR-10 AST gates",
        "T-IGX-036-I",
        "WB",
        (
            "tests/qualification/ebh_5/",
            "tests/qualification/governance/",
            "tests/unit/applications/task_control_product_host_test_support.py",
            "docs/project/maintainers/qualification/EBH_5_CROSS_LAYER_REPLACEABILITY_MATRIX.md",
        ),
        "—",
        (),
        GATE_FREE,
        "IA-D3 (ebh_5), IA-D4 (GR-12), IA-D6 (GR-10)",
        "tests/qualification/ebh_5 + tests/qualification/governance green",
    ),
    Batch(
        "B-REG-P",
        "REG",
        "Regression: duplicate `ollama` catalog slug breaks full integration bootstrap (7aa9cfa18)",
        "T-IGX-036-M",
        "W1",
        ("intergrax/integrations/providers/embedding_provider/", "tests/qualification/pdc_x/reg_p/"),
        "—",
        (),
        GATE_PRODUCT,
        "IA-D1 (plug_03 replay)",
        "tests/unit/rag/test_rag_plugin_discovery.py + tests/qualification/plug_03 green",
    ),
    Batch(
        "B-D1",
        "D1",
        "Agent lifecycle: upgrade/retire, skill lifecycle, discovery owner, adversarial lifecycle",
        "T-IGX-036-J",
        "W1",
        (
            "intergrax/agent_distribution/",
            "intergrax/skills/",
            "intergrax/runtime/registry/",
            "intergrax/core/plugins/discovery.py",
            "intergrax/rag/bootstrap/entry_point_load.py",
            "intergrax/runtime/vendor_knowledge/contribution_catalog.py",
            "intergrax/contracts/agent_lifecycle_state.py",
            "tests/qualification/pdc_x/d1/",
        ),
        "ADR-PDC-01",
        (),
        GATE_PRODUCT,
        "IA-D1",
        "tests/qualification/pdc_x/d1/",
    ),
    Batch(
        "B-D3",
        "D3",
        "Model runtime: declared streaming/tool-streaming/cancel capabilities, conformance gate, catalog-owned context windows",
        "T-IGX-036-K",
        "W1",
        ("intergrax/llm_adapters/", "tests/qualification/pdc_x/d3/"),
        "ADR-PDC-03",
        (),
        GATE_PRODUCT,
        "IA-D3",
        "tests/qualification/pdc_x/d3/",
    ),
    Batch(
        "B-D7",
        "D7",
        "Background: recurrence on SchedulerLedger, dead-letter state, identity on enqueue, two-process fencing",
        "T-IGX-036-L",
        "W1",
        (
            "intergrax/queueing/",
            "intergrax/background_tasks/",
            "intergrax/runtime/long_running/",
            "intergrax/runtime/background_execution/",
            "tests/qualification/pdc_x/d7/",
        ),
        "ADR-PDC-07",
        (),
        GATE_PRODUCT,
        "IA-D7",
        "tests/qualification/pdc_x/d7/",
    ),
    Batch(
        "B-D2",
        "D2",
        "Context: trim fidelity, durable compaction (CE-02 closure), memory retention, RAG freshness, adversarial context",
        "T-IGX-036-N",
        "W2",
        (
            "intergrax/context/",
            "intergrax/runtime/nexus/context/",
            "intergrax/runtime/token_optimization/",
            "intergrax/runtime/context_lifecycle/",
            "intergrax/memory/",
            "intergrax/rag/governance/",
            "intergrax/rag/retrieval/retrieval_result.py",
            "intergrax/rag/retrieval/retrieval_service.py",
            "intergrax/rag/ingest/ingest_pipeline.py",
            "tests/qualification/pdc_x/d2/",
        ),
        "ADR-PDC-02",
        ("B-D1",),
        GATE_PRODUCT,
        "IA-D2",
        "tests/qualification/pdc_x/d2/ + ce_02 replay",
    ),
    Batch(
        "B-D5",
        "D5",
        "Evaluation: typed EvalDataset/EvalRun/EvalComparison, evaluator kinds, evidence persistence, advisory promotion",
        "T-IGX-036-O",
        "W2",
        (
            "intergrax/eval/",
            "intergrax/experiments/",
            "intergrax/runtime/evaluation/",
            "intergrax/runtime/architecture/evaluation_assets.py",
            "intergrax/runtime/architecture/evaluation_automation.py",
            "intergrax/runtime/architecture/evaluation_registry_trends.py",
            "intergrax/runtime/architecture/evaluation_scenario_loader.py",
            "intergrax/runtime/architecture/online_evaluation_registry.py",
            "intergrax/runtime/architecture/agent_promotion.py",
            "intergrax/runtime/evidence/eval_evidence_contracts.py",
            "intergrax/contracts/evaluation_contracts.py",
            "intergrax/applications/_shared/evaluation_wiring.py",
            "scripts/gates/check_mvp_promotion_gates.py",
            "tests/qualification/pdc_x/d5/",
        ),
        "ADR-PDC-05",
        ("B-D3",),
        GATE_PRODUCT,
        "IA-D5",
        "tests/qualification/pdc_x/d5/",
    ),
    Batch(
        "B-D6",
        "D6",
        "Security: fail-closed signing/poisoning defaults, single tool-input defense, durable audit trail, secret gate, plugin provenance",
        "T-IGX-036-P",
        "W3a",
        (
            "intergrax/runtime/security/",
            "intergrax/core/security/",
            "intergrax/core/catalog_bootstrap.py",
            "intergrax/core/plugins/admission.py",
            "intergrax/runtime/execution_evidence/attestor.py",
            "intergrax/tools/providers/rag/service.py",
            "intergrax/applications/_shared/critical_action_signing_wiring.py",
            "intergrax/applications/_shared/security_assembly_resolver.py",
            "intergrax/applications/_shared/security_audit_trail_wiring.py",
            "intergrax/applications/_shared/application_security_wiring.py",
            "docs/project/maintainers/architecture/satellites/",
            "intergrax/runtime/attestation/host_attestation.py",
            "intergrax/runtime/attestation/__init__.py",
            "intergrax/applications/_shared/attestation_runtime_bridge.py",
            "intergrax/core/plugins/discovery.py",
            "intergrax/runtime/nexus/tools/plan_context_invocation.py",
            "scripts/maintenance/check_critical_action_signing.py",
            "scripts/maintenance/check_immutable_security_audit_trail.py",
            "tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py",
            "tests/qualification/pdc_x/d6/",
        ),
        "ADR-PDC-06",
        ("B-D2", "B-D5"),
        GATE_PRODUCT,
        "IA-D6",
        "tests/qualification/pdc_x/d6/",
    ),
    Batch(
        "B-D8",
        "D8",
        "Artifacts: taxonomy, mandatory digest, governed read/download port, durable store, retention, spill, dedupe",
        "T-IGX-036-Q",
        "W3b",
        (
            "intergrax/contracts/artifact_ref.py",
            "intergrax/contracts/task_artifacts.py",
            "intergrax/applications/contracts/application_artifacts.py",
            "intergrax/runtime/nexus/artifacts/",
            "intergrax/runtime/nexus/tracing/trace_models.py",
            "intergrax/collaborative_work/",
            "intergrax/runtime/execution/run_artifact_bundle_composition.py",
            "intergrax/runtime/nexus/orchestration/run_artifact_bundle_builder.py",
            "intergrax/applications/_shared/run_artifact_bundle_builder.py",
            "docs/project/capabilities/architecture/ARTIFACT_LIFECYCLE.md",
            "docs/project/capabilities/plan/ARTIFACT_LIFECYCLE.md",
            "tests/qualification/state_x/_state_x_explicit_mechanism_classifications.py",
            "intergrax/tools/providers/storage/service.py",
            "intergrax/runtime/replay/models.py",
            "intergrax/runtime/replay/replay_engine.py",
            "tests/qualification/pdc_x/d8/",
        ),
        "ADR-PDC-08",
        ("B-D6",),
        GATE_PRODUCT,
        "IA-D8",
        "tests/qualification/pdc_x/d8/",
    ),
    Batch(
        "B-D4",
        "D4",
        "Hosting: lifecycle suite + crash/restart proof, metrics, digest-preserving restart, stable API, TL-FIX-C",
        "T-IGX-036-R",
        "W3c",
        (
            "intergrax/hosting/",
            "intergrax/applications/_shared/environment_wiring.py",
            "intergrax/applications/contracts/environment_profile/sub_profiles.py",
            "intergrax/applications/_shared/lkw_hybrid_daemon_wiring.py",
            "tests/qualification/pdc_x/d4/",
        ),
        "ADR-PDC-04",
        ("B-D7", "B-D8"),
        GATE_PRODUCT,
        "IA-D4",
        "tests/qualification/pdc_x/d4/",
    ),
    Batch(
        "B-P6",
        "ALL",
        "Cross-domain integration XD-01..12, DUP-X/EBH-4 replay, full ECAP change-delta",
        "T-IGX-036-S",
        "W4",
        ("tests/qualification/pdc_x/xd/",),
        "—",
        ("B-D1", "B-D2", "B-D3", "B-D4", "B-D5", "B-D6", "B-D7", "B-D8", "B-REG-P", "B-REG-T"),
        GATE_PRODUCT,
        "IA-GLOBAL (P8) and each IA-Dn for its XD rows",
        "tests/qualification/pdc_x/xd/",
    ),
    Batch(
        "B-P6-R",
        "ALL",
        "Integration repairs found by the P5 lanes and the XD runs (tzdata, capability flags, freshness, launcher path, circular import, trace alias callers, feature-doc registration)",
        "T-IGX-036-T",
        "W5",
        (
            "pyproject.toml",
            "uv.lock",
            "intergrax/runtime/llm/model_call_runtime_evidence_adapter.py",
            "intergrax/rag/retrieval/agentic_loop.py",
            "intergrax/applications/_shared/lkw_hybrid_daemon_wiring.py",
            "intergrax/runtime/integrations/observability.py",
            "intergrax/runtime/execution/agent_runtime_context.py",
            "intergrax/runtime/execution/trace_persistence_models_bridge.py",
            "intergrax/runtime/nexus/context/context_manager.py",
            "intergrax/runtime/nexus/context/shared_task_context.py",
            "intergrax/runtime/nexus/engine/runtime_state.py",
            "intergrax/runtime/nexus/engine/contracts/runtime_state_contract.py",
            "intergrax/runtime/nexus/tracing/persistence_models.py",
            "docs/project/capabilities/README.md",
            "docs/project/architecture/intergrax_runtime_architecture.md",
            "intergrax/runtime/nexus/tools/tool_runtime.py",
            "intergrax/runtime/nexus/tools/plan_context_invocation.py",
            "intergrax/runtime/task/worker_bootstrap.py",
            "intergrax/runtime/persistence/sqlite_opens.py",
            "intergrax/runtime/persistence/sqlite_composition.py",
            "intergrax/runtime/nexus/artifacts/context_reference.py",
            "intergrax/runtime/nexus/artifacts/spill.py",
            "intergrax/memory/default_memory_control_plane.py",
            "intergrax/runtime/agent_governance/pipeline.py",
            "intergrax/llm_adapters/contracts/capabilities.py",
            "tests/qualification/ebh_5/_sqlite_touchpoint_registry.py",
            "intergrax/runtime/observability/export_boundary.py",
            "tests/qualification/pdc_x/d4/test_d4_no_product_vocabulary.py",
            "tests/qualification/pdc_x/xd/test_xd_02_agent_context.py",
            "tests/qualification/pdc_x/xd/test_xd_03_agent_governance.py",
            "tests/qualification/pdc_x/xd/test_xd_09_artifacts_context.py",
            "tests/qualification/pdc_x/xd/test_xd_10_artifacts_security.py",
            "tests/unit/memory/test_mem_ent10_memory_security_governance.py",
            "tests/qualification/ebh_5/test_ebh_5_proof_integrity_gates.py",
            "tests/qualification/pdc_x/p6_repair/",
        ),
        "—",
        ("B-D1", "B-D2", "B-D3", "B-D4", "B-D5", "B-D6", "B-D7", "B-D8", "B-REG-P", "B-REG-T", "B-P6"),
        GATE_PRODUCT,
        "IA-GLOBAL (P8)",
        "tests/qualification/pdc_x/p6_repair/",
    ),
    Batch(
        "B-P6-M",
        "ALL",
        "Model binding found by XD-01: activated revision pins the LLM profile digest; required model features of an agent are checked when the adapter is bound",
        "T-IGX-036-U",
        "W6",
        (
            "intergrax/agent_distribution/runtime_revision.py",
            "intergrax/agent_distribution/admin_service.py",
            "intergrax/contracts/agent_contract_meta.py",
            "intergrax/applications/_shared/llm_resolver.py",
            "tests/qualification/pdc_x/xd/test_xd_01_agent_model.py",
            "tests/qualification/pdc_x/p6_model/",
        ),
        "—",
        ("B-P6-R",),
        GATE_PRODUCT,
        "IA-GLOBAL (P8)",
        "tests/qualification/pdc_x/p6_model/",
    ),
    Batch(
        "B-P6-G",
        "ALL",
        "ENT-AUDIT impact-map coverage of the PDC-touched product modules (XD finding 8)",
        "T-IGX-036-V",
        "W7",
        (
            "intergrax/platform/enterprise_audit/impact_map.py",
            "intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json",
            "tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py",
            "tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py",
            "tests/qualification/pdc_x/p6_impact/",
        ),
        "—",
        ("B-P6-R", "B-P6-M"),
        GATE_PRODUCT,
        "IA-GLOBAL (P8)",
        "tests/qualification/pdc_x/p6_impact/",
    ),
)

# Requirement -> batch overrides where the corrective-task text alone is ambiguous.
OVERRIDES: Final[dict[str, str]] = {
    "PDC-D1-R022": "B-REG-P",
    "PDC-D3-R024": "B-REG-T",
    "PDC-D4-R020": "B-REG-T",
    "PDC-D6-R028": "B-REG-T",
    "PDC-D1-R021": "B-P3",
    "PDC-D7-R022": "B-P3",
    "PDC-D6-R017": "B-MCP",
    "PDC-D5-R017": "B-D5",
    "PDC-D6-R021": "B-D6",
    "PDC-D6-R023": "B-D6",
    "PDC-D6-R022": "B-D8",
}
_LANE_TO_BATCH: Final[dict[str, str]] = {
    "PDC-P5-D1": "B-D1",
    "PDC-P5-D2": "B-D2",
    "PDC-P5-D3": "B-D3",
    "PDC-P5-D4": "B-D4",
    "PDC-P5-D5": "B-D5",
    "PDC-P5-D6": "B-D6",
    "PDC-P5-D7": "B-D7",
    "PDC-P5-D8": "B-D8",
    "PDC-P5-DOC": "B-DOC",
    "PDC-P6": "B-P6",
}
_LANE = re.compile(r"PDC-P5-D\d|PDC-P5-DOC|PDC-P6")

XD: Final[tuple[tuple[str, str, str, str], ...]] = (
    (
        "XD-01",
        "Agent ↔ Model",
        "Activated agent revision pins profile/model; provider swap by config leaves the agent contract unchanged; capability mismatch fails closed",
        "test_xd_01_agent_model.py",
    ),
    (
        "XD-02",
        "Agent ↔ Context",
        "Tenant A agent cannot read tenant B memory/RAG; revoked agent loses context access",
        "test_xd_02_agent_context.py",
    ),
    (
        "XD-03",
        "Agent ↔ Tools/Governance",
        "Agent-proposed tool call without a governance permit is denied; no agent-owned execution path",
        "test_xd_03_agent_governance.py",
    ),
    (
        "XD-04",
        "Agent ↔ Host",
        "Configured vs effective roster; restart preserves active revision and profile digest",
        "test_xd_04_agent_host.py",
    ),
    (
        "XD-05",
        "Host ↔ Background",
        "Shutdown drains in-flight tasks; scheduler lease released/fenced; no duplicate on restart",
        "test_xd_05_host_background.py",
    ),
    (
        "XD-06",
        "Background ↔ Security",
        "Retried task re-authorizes; credential revoked mid-retry fails closed; no duplicate side effect",
        "test_xd_06_background_security.py",
    ),
    (
        "XD-07",
        "Model ↔ Context",
        "Budget/truncation keeps mandatory instructions; streamed tool-call provenance recorded; restart reconstruction",
        "test_xd_07_model_context.py",
    ),
    (
        "XD-08",
        "Evaluation ↔ Model/Agent",
        "Baseline vs candidate rerun is digest-identical; an evaluation PASS cannot authorize promotion",
        "test_xd_08_evaluation.py",
    ),
    (
        "XD-09",
        "Artifacts ↔ Context/Tools",
        "Large tool output spills to an artifact ref; compaction keeps the ref; evidence artifacts undeletable",
        "test_xd_09_artifacts_context.py",
    ),
    (
        "XD-10",
        "Artifacts ↔ Security",
        "Cross-tenant artifact read denied; TTL deletion honored; no raw secret in artifacts",
        "test_xd_10_artifacts_security.py",
    ),
    (
        "XD-11",
        "All ↔ Observability/Trace",
        "Every mandatory operation emits canonical trace/evidence (inventory-driven assertion)",
        "test_xd_11_observability.py",
    ),
    (
        "XD-12",
        "All ↔ EVOL/ENT-AUDIT",
        "ENT-AUDIT changed_since maps every PDC-touched path to checks (impact-map completeness)",
        "test_xd_12_evol_ent_audit.py",
    ),
)


def load_rows() -> list[dict[str, str]]:
    rows = json.loads(MATRIX.read_text(encoding="utf-8"))["rows"]
    assert isinstance(rows, list)
    return rows


def work_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if r["GAP_CLASS"] in GAP_CLASSES]


def assign(row: dict[str, str]) -> list[str]:
    rid = row["REQUIREMENT_ID"]
    if rid in OVERRIDES:
        return [OVERRIDES[rid]]
    return sorted({_LANE_TO_BATCH[m] for m in _LANE.findall(row["CORRECTIVE_TASK"])})


def _overlap(a: str, b: str) -> bool:
    return a == b or a.startswith(b) or b.startswith(a)


def validate(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    ids = {b.bid for b in BATCHES}
    if len(ids) != len(BATCHES):
        errors.append("DUPLICATE_BATCH_ID")
    for row in work_rows(rows):
        got = assign(row)
        if len(got) != 1:
            errors.append(f"ASSIGNMENT:{row['REQUIREMENT_ID']}:{got}")
        elif got[0] not in ids:
            errors.append(f"UNKNOWN_BATCH:{row['REQUIREMENT_ID']}:{got[0]}")
    for b in BATCHES:
        errors.extend(f"UNKNOWN_DEP:{b.bid}:{d}" for d in b.deps if d not in ids)
    by_wave: dict[str, list[Batch]] = {}
    for b in BATCHES:
        by_wave.setdefault(b.wave, []).append(b)
    for wave, group in by_wave.items():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                errors.extend(
                    f"WRITE_CONFLICT:{wave}:{a.bid}:{b.bid}:{pa}"
                    for pa in a.writes
                    for pb in b.writes
                    if _overlap(pa, pb)
                )
        for hot in HOTSPOTS:
            writers = [b.bid for b in group if any(w.startswith(hot) for w in b.writes)]
            if len(writers) > 1:
                errors.append(f"HOTSPOT:{wave}:{hot}:{writers}")
    seen: set[str] = set()
    stack: set[str] = set()
    graph = {b.bid: b.deps for b in BATCHES}

    def visit(node: str) -> None:
        if node in stack:
            errors.append(f"CYCLE:{node}")
            return
        if node in seen or node not in graph:
            return
        stack.add(node)
        for dep in graph[node]:
            visit(dep)
        stack.discard(node)
        seen.add(node)

    for bid in graph:
        visit(bid)
    return errors


def _is_product(batch: Batch) -> bool:
    return any(w.startswith(PRODUCT_PREFIXES) for w in batch.writes)


def render_wbs(rows: list[dict[str, str]]) -> str:
    members: dict[str, list[dict[str, str]]] = {b.bid: [] for b in BATCHES}
    for row in work_rows(rows):
        members[assign(row)[0]].append(row)
    total = len(work_rows(rows))
    out = [
        "# PDC-X — implementation WBS (PDC-P4)",
        "",
        "**Generated** by `scripts/qualification/pdc_x_wbs.py` from `PDC_DOMAIN_REQUIREMENTS_MATRIX.json` and the checked-in batch table; do not edit by hand. Ticket T-IGX-036-G.",
        "",
        f"**Coverage:** {total} work-requiring requirement rows, each assigned to exactly one batch (checked).",
        "Rows classed `CURRENT_HEAD_REQUALIFICATION`, `ALREADY_CERTIFIED_REUSABLE`, `N/A_WITH_EVIDENCE` or `DEFERRED_OUTSIDE_FREEZE_SCOPE` close at PDC-P7 by replay or carry their authorizing OD, so they are not P5 work.",
        "",
        f"**Integration owner:** {INTEGRATION_OWNER}. Shared hotspots: "
        + ", ".join(f"`{h}`" for h in HOTSPOTS)
        + ". At most one batch per wave may write under each hotspot (checked).",
        "",
        "**Ticket letters** for P5 batches are reserved here. Each ticket file is scaffolded with its own plan at batch start, so draft tickets do not go stale while product merges wait on MCP-SERVER-X P8.",
        "",
        "## Batches",
        "",
        "| Batch | Lane | Wave | Ticket (reserved) | Rows | ADR | Depends on | Product code | Merge gate | Auditor |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for b in BATCHES:
        out.append(
            f"| {b.bid} | {b.lane} | {b.wave} | {b.ticket} | {len(members[b.bid])} | {b.adr} | {', '.join(b.deps) or '—'} | "
            f"{'YES' if _is_product(b) else 'no'} | {b.gate} | {b.auditor} |"
        )
    for b in BATCHES:
        out += [
            "",
            f"## {b.bid} — {b.title}",
            "",
            f"- **Ticket:** {b.ticket} · **Wave:** {b.wave} · **ADR:** {b.adr} · **Auditor:** {b.auditor}",
            f"- **Write set:** {', '.join(f'`{w}`' for w in b.writes) if b.writes else '— (no writes)'}",
            f"- **Tests:** {b.tests}",
            f"- **Evidence (exact SHA):** implementation SHA and publication SHA recorded in `PDC_SHA_LADDER.md`; test log under `.tmp/proof/pdc-x-p5/{b.bid}/`; audited by {b.auditor} at the PDC-P7 audited SHA.",
            f"- **Merge gate:** {b.gate}",
            "",
            "| Requirement | Class | Test case |",
            "| --- | --- | --- |",
        ]
        out += [f"| {r['REQUIREMENT_ID']} | {r['GAP_CLASS']} | {r['TEST_CASE']} |" for r in members[b.bid]] or [
            "| — | — | — |"
        ]
    return "\n".join(out) + "\n"


def render_dag() -> str:
    out = [
        "# PDC-X — dependency DAG (PDC-P4-02)",
        "",
        "**Generated** by `scripts/qualification/pdc_x_wbs.py`. Waves run in order; batches within a wave run in parallel in separate worktrees (at most 3 concurrent implementer sessions, one bounded pytest each). Fala 3 is deliberately sequential because its batches share `intergrax/applications/_shared/` and `intergrax/contracts/`.",
        "",
        "```mermaid",
        "graph LR",
        "  P4[PDC-P4 design] --> WA",
        "  subgraph WA[Wave A - now]",
        "    B-DOC",
        "  end",
        "  subgraph WB[Wave B - tests only]",
        "    B-REG-T",
        "  end",
        "  WA --> WB",
        "  GATE{{OD-PDC-SEQ-01: MCP-SERVER-X P8 published}}",
    ]
    for b in BATCHES:
        if b.wave in {"W0-done", "W0-external", "WA", "WB"}:
            continue
        for dep in b.deps:
            out.append(f"  {dep} --> {b.bid}")
    out += [
        "  P4 --> B-D1 & B-D3 & B-D7 & B-REG-P",
        "  GATE -. merges .-> B-D1 & B-D3 & B-D7 & B-REG-P & B-D2 & B-D5 & B-D6 & B-D8 & B-D4",
        "  B-P6 --> ACT{{PDC-P7-00 ACT-N11 activation}} --> P7[PDC-P7: 8 domain audits]",
        "  B-REG-T --> P7",
        "  P7 --> MCPC{{MCP-SERVER-X CLOSED}} --> P8[PDC-P8 global audit] --> P9[PDC-P9 closure + ARCH-FREEZE handoff]",
        "```",
        "",
        "## Critical path",
        "",
        "PDC-P4 → (implementation in worktrees, merges gated) → **MCP-SERVER-X P8 published** → B-D6 → B-D8 → B-D4 → B-P6 → PDC-P7-00 ACT-N11 → PDC-P7 → PDC-P8 → PDC-P9.",
        "",
        "Wave 3 sets the length of the path. Its three batches run in sequence because they share hotspot files. Waves 1–2 run alongside it and finish earlier.",
        "",
        "## Waves",
        "",
        "| Wave | Batches | Can start | Merge |",
        "| --- | --- | --- | --- |",
    ]
    waves: dict[str, list[str]] = {}
    for b in BATCHES:
        waves.setdefault(b.wave, []).append(b.bid)
    starts = {
        "W0-done": ("—", "done"),
        "W0-external": ("MCP-SERVER-X campaign", "MCP-SERVER-X"),
        "WA": ("now", "any time"),
        "WB": ("after P4", "any time (tests only)"),
        "W1": ("after P4", "after MCP-SERVER-X P8"),
        "W2": ("after W1 dependencies", "after MCP-SERVER-X P8"),
        "W3a": ("after W2", "after MCP-SERVER-X P8"),
        "W3b": ("after B-D6", "after MCP-SERVER-X P8"),
        "W3c": ("after B-D7 and B-D8", "after MCP-SERVER-X P8"),
        "W4": ("after all P5 batches", "after MCP-SERVER-X P8"),
        "W5": ("after B-P6 (XD runs)", "after MCP-SERVER-X P8"),
        "W6": ("after B-P6-R", "after MCP-SERVER-X P8"),
        "W7": ("after B-P6-R and B-P6-M", "after MCP-SERVER-X P8"),
    }
    for wave, bids in waves.items():
        start, merge = starts.get(wave, ("?", "?"))
        out.append(f"| {wave} | {', '.join(bids)} | {start} | {merge} |")
    out += [
        "",
        "## Operator gates",
        "",
        "| Gate | Blocks | Authority |",
        "| --- | --- | --- |",
        "| MCP-SERVER-X P8 published | every product-path merge (B-REG-P, B-D1..B-D8, B-P6) | OD-PDC-SEQ-01 |",
        "| PDC-P7-00 ACT-N11 activation at INTEGRATED_BASELINE_SHA | PDC-P7 audits | OD-PDC-IIAC-SCOPE + Operator activation line |",
        "| IIAC protocol header sub-question | PDC-P7-00 | Operator (open) |",
        "| MCP-SERVER-X CLOSED | PDC-P8 | roadmap §3.2 |",
        "| PDC-P9 closure | ARCH-FREEZE handoff | IIAC-B6 + Operator |",
    ]
    return "\n".join(out) + "\n"


def render_xd(rows: list[dict[str, str]]) -> str:
    out = [
        "# PDC-X — cross-domain test matrix (PDC-P4-03)",
        "",
        "**Generated** by `scripts/qualification/pdc_x_wbs.py`. Owner batch: **B-P6** (T-IGX-036-S). Every scenario is a real-integration qualification test: mandatory paths use the canonical composition roots, not mocks; fakes are allowed only for external vendors (Path B).",
        "",
        "| XD | Pair | Scenario | Planned module | Requirements proven |",
        "| --- | --- | --- | --- | --- |",
    ]
    for xid, pair, scenario, module in XD:
        tag = "xd_" + xid[-2:]
        proven = [r["REQUIREMENT_ID"] for r in rows if tag in r["TEST_CASE"]]
        out.append(
            f"| {xid} | {pair} | {scenario} | `tests/qualification/pdc_x/xd/{module}` | {', '.join(proven) or 'scenario-level (no single row)'} |"
        )
    return "\n".join(out) + "\n"


def _outputs(rows: list[dict[str, str]]) -> tuple[tuple[Path, str], ...]:
    return ((OUT_WBS, render_wbs(rows)), (OUT_DAG, render_dag()), (OUT_XD, render_xd(rows)))


def check() -> list[str]:
    rows = load_rows()
    errors = validate(rows)
    if errors:
        return errors
    for path, text in _outputs(rows):
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            errors.append(f"STALE_OUTPUT:{path.name} (re-run with --write)")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rows = load_rows()
    if args.write:
        errors = validate(rows)
        if not errors:
            for path, text in _outputs(rows):
                path.write_text(text, encoding="utf-8", newline="\n")
            print(f"[OK] wrote WBS for {len(work_rows(rows))} rows in {len(BATCHES)} batches")
            return 0
    else:
        errors = check()
        if not errors:
            print(f"[OK] pdc_x_wbs: {len(work_rows(rows))} rows assigned, {len(BATCHES)} batches, no write conflicts")
            return 0
    for err in errors:
        print(f"[ERROR] {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
