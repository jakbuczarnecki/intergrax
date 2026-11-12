# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E501  (requirement text data)

"""PDC-X P2 requirements matrix (T-IGX-036-C).

Single source for ``PDC_DOMAIN_REQUIREMENTS_MATRIX.{json,md}``. Rows are authored compactly in
``ROWS``; every mandated matrix field is derived deterministically. ``--check`` validates the
schema, the gap classes, FRZ identifiers against the freeze checklist (225 / 26 unchanged), and
that the committed outputs are current.

No row may be certified here: ``CERTIFICATION_STATUS`` is ``NOT_CERTIFIED`` until PDC-P7.

Usage::

    uv run python scripts/qualification/pdc_x_requirements.py --write
    uv run python scripts/qualification/pdc_x_requirements.py --check
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
OUT_JSON: Final[Path] = DOSSIER / "PDC_DOMAIN_REQUIREMENTS_MATRIX.json"
OUT_MD: Final[Path] = DOSSIER / "PDC_DOMAIN_REQUIREMENTS_MATRIX.md"
CHECKLIST: Final[Path] = (
    REPO / "docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md"
)
SCHEMA_VERSION: Final[str] = "pdc-x-requirements.v1"
AUDIT_HEAD: Final[str] = "1ebd2ab4c2e6b7b03c858f0ede172466cac1f089"
EXPECTED_FRZ: Final[int] = 225
EXPECTED_FAMILIES: Final[int] = 26

FIELDS: Final[tuple[str, ...]] = (
    "REQUIREMENT_ID",
    "DOMAIN_ID",
    "REQUIREMENT",
    "SEMANTIC_OWNER",
    "IMPLEMENTATION_PATH",
    "PUBLIC_CONTRACT",
    "CURRENT_IMPLEMENTATION_STATUS",
    "EXISTING_CERTIFICATION",
    "EXISTING_ACCEPTANCE_SHA",
    "CURRENT_HEAD_DELTA",
    "MANDATORY_GAP",
    "CORRECTIVE_TASK",
    "TEST_CASE",
    "FRZ_IDS",
    "SECURITY_IMPACT",
    "TENANT_IMPACT",
    "DEPENDENCIES",
    "CERTIFICATION_STATUS",
    "AUDITOR_ID",
    "AUDITED_SHA",
    "EVIDENCE_PATH",
    "ARCH_FREEZE_BLOCKER",
    "FINAL_DISPOSITION",
    "GAP_CLASS",
)

CLASSES: Final[frozenset[str]] = frozenset(
    {
        "ALREADY_CERTIFIED_REUSABLE",
        "CURRENT_HEAD_REQUALIFICATION",
        "IMPLEMENTATION_COMPLETION",
        "INTEGRATION_CORRECTION",
        "DOCUMENTATION_RECONCILIATION",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "N/A_WITH_EVIDENCE",
        "DEFERRED_OUTSIDE_FREEZE_SCOPE",
        "BLOCKED_EXACT_REASON",
    }
)
GAP_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "IMPLEMENTATION_COMPLETION",
        "INTEGRATION_CORRECTION",
        "DOCUMENTATION_RECONCILIATION",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "BLOCKED_EXACT_REASON",
    }
)
NO_TASK_CLASSES: Final[frozenset[str]] = frozenset({"ALREADY_CERTIFIED_REUSABLE", "N/A_WITH_EVIDENCE"})

# Accepted certifications: stage -> (accepted SHA, delta to AUDIT_HEAD over the row scope).
CERTS: Final[dict[str, tuple[str, str]]] = {
    "SKILL-X": ("552e379845a9512c275f3ebe520ac51767c97cf8", "1 file (skills/skill_execution_pinning_wiring.py)"),
    "AC1-AC6-FINAL-AUDIT": (
        "67f57331835c247716ff028c0029157ab3f53207",
        "34 files (agent_distribution, runtime/registry)",
    ),
    "EBH-2G": ("ba41ef7592c07df04d75a15cd2efc36fe86bd041", "53 files (intergrax/rag)"),
    "EBH-2H": ("86fafaa24ccdfb47faba44ddcec1b6eaec5bf195", "13 files (intergrax/memory)"),
    "HARNESS-RESIDUAL/CE-01": (
        "a8d47a6396c3e5e73a979be74429ed59a93e1551",
        "2 files (context tracking, nexus context_manager)",
    ),
    "TENANT-X": (
        "5ccb72fccd4e3820ebdc9c18008afe6f7d59f558",
        "1 file in security scope (policy/side_effect_authorization_errors.py)",
    ),
    "EBH-2E": ("6fa75172468461dc084a2221a5fdff6f23c1fa1c", "9 files (llm_adapters, runtime/llm)"),
    "EBH-5": ("33ecf19cc020b112d9e586c87b49e819202f5d6d", "0 files (llm_adapters)"),
    "HARNESS-W7/HOST-01": (
        "5d5442dae7769671a7532a973428c8203b492ec6",
        "1 file (applications/_shared/environment_wiring.py)",
    ),
    "CTRL-X": ("318a4d835534ef5c48575f4649551c957dacd8ea", "0 files (eval, experiments, evaluation scope)"),
    "GOV-X2": ("e2d7a40b291c658677cc6d981183e20957c7cf0e", "16 files (runtime/governance, runtime/policy, delegation)"),
    "HARNESS-W8/BG-01": (
        "3a9b58e6fa1e14ddc9786bd9a447a1d76552d196",
        "0 files (queueing, background_tasks, background_execution)",
    ),
    "HARNESS-RESIDUAL/SCHED-01": ("433ff5fb97262c4bad84ea2a4173dc3711d7aa08", "9 files (runtime/long_running)"),
    "TOOL-LIFE-X": (
        "1548640708c850b3fbfca3f2db0e5ecfee4274dd",
        "1 file (long_running/postgresql_task_checkpoint_store.py)",
    ),
    "STATE-X": ("bd54941d933069b8bfb2819bb819c1cbdbe71576", "3 files (long_running, persistence)"),
}

# PDC-P2-09 replay at AUDIT_HEAD (one bounded pytest over all suites; `--extra dev-ci --extra dev-unit-cert`):
# 1514 passed, 13 failed, 2 skipped in 549.70s. The branch has zero product-path delta versus origin/development
# (bc9ef52b), so every failure is pre-existing on development. Suite -> (passed, failed, skipped).
REPLAY: Final[dict[str, tuple[int, int, int]]] = {
    "plug_03": (17, 1, 0),
    "plug_04": (4, 0, 0),
    "plug_final": (5, 0, 0),
    "ce_01": (15, 0, 0),
    "ce_02": (21, 0, 0),
    "memory_behavior": (47, 0, 0),
    "ebh_5": (239, 4, 0),
    "host_01": (20, 0, 0),
    "control_planes": (76, 0, 0),
    "tenant_x": (194, 0, 0),
    "governance": (395, 8, 0),
    "prod_q": (445, 0, 2),
    "bg_01": (22, 0, 0),
    "sched_01": (14, 0, 0),
}
# Certification -> qualification suites whose replay must be green before reuse is allowed.
CERT_SUITES: Final[dict[str, tuple[str, ...]]] = {
    "EBH-5": ("ebh_5",),
    "TENANT-X": ("tenant_x",),
    "HARNESS-W8/BG-01": ("bg_01",),
    "HARNESS-RESIDUAL/SCHED-01": ("sched_01",),
    "HARNESS-RESIDUAL/CE-01": ("ce_01",),
    "HARNESS-W7/HOST-01": ("host_01",),
    "CTRL-X": ("control_planes",),
    "GOV-X2": ("governance",),
}
REPLAY_FINDING_ROW: Final[dict[str, str]] = {
    "ebh_5": "PDC-D3-R024",
    "governance": "PDC-D6-R028",
    "plug_03": "PDC-D1-R022",
}

# Operator decisions recorded under T-IGX-036-E (PDC-P3-08, 2026-11-12: "Accept all A").
APPROVED_ODS: Final[frozenset[str]] = frozenset(
    {f"OD-PDC-SCOPE-0{i}" for i in range(1, 8)} | {"OD-PDC-PATHB-01", "OD-PDC-ADR-01", "OD-PDC-IIAC-SCOPE"}
)
# Row -> (class, corrective task) after the decision. The source rows keep their P2 classification.
OD_RESOLUTIONS: Final[dict[str, tuple[str, str]]] = {
    "PDC-D1-R018": ("IMPLEMENTATION_COMPLETION", "PDC-P5-D1 (ADR-PDC-01; OD-PDC-SCOPE-01 A: implement)"),
    "PDC-D1-R019": ("DEFERRED_OUTSIDE_FREEZE_SCOPE", "OD-PDC-SCOPE-03 A: authorized exclusion"),
    "PDC-D2-R005": ("IMPLEMENTATION_COMPLETION", "PDC-P5-D2 (ADR-PDC-02; OD-PDC-SCOPE-06 A: implement)"),
    "PDC-D3-R020": ("DEFERRED_OUTSIDE_FREEZE_SCOPE", "OD-PDC-SCOPE-03 A: authorized exclusion"),
    "PDC-D4-R011": ("DEFERRED_OUTSIDE_FREEZE_SCOPE", "OD-PDC-SCOPE-02 A: authorized exclusion"),
    "PDC-D4-R012": ("DEFERRED_OUTSIDE_FREEZE_SCOPE", "OD-PDC-SCOPE-02 A: authorized exclusion"),
    "PDC-D4-R013": ("DEFERRED_OUTSIDE_FREEZE_SCOPE", "OD-PDC-SCOPE-02 A: authorized exclusion"),
    "PDC-D6-R016": ("IMPLEMENTATION_COMPLETION", "PDC-P5-D6 (ADR-PDC-06; OD-PDC-SCOPE-07 A: implement)"),
    "PDC-D7-R006": ("IMPLEMENTATION_COMPLETION", "PDC-P5-D7 (ADR-PDC-07; OD-PDC-SCOPE-05 A: implement)"),
    "PDC-D7-R007": ("IMPLEMENTATION_COMPLETION", "PDC-P5-D7 (ADR-PDC-07; OD-PDC-SCOPE-05 A: implement)"),
    "PDC-D7-R008": ("IMPLEMENTATION_COMPLETION", "PDC-P5-D7 (ADR-PDC-07; OD-PDC-SCOPE-05 A: implement)"),
    "PDC-D8-R001": ("IMPLEMENTATION_COMPLETION", "PDC-P5-D8 (ADR-PDC-08 accepted)"),
    "PDC-D8-R010": ("IMPLEMENTATION_COMPLETION", "PDC-P5-D8 (ADR-PDC-08; OD-PDC-SCOPE-04 A: implement)"),
}
_OD_REF = re.compile(r"OD-PDC-[A-Z0-9]+(?:-\d+)?")


def _red_suites(cert: str | None) -> list[str]:
    return [s for s in CERT_SUITES.get(cert or "", ()) if REPLAY[s][1] > 0]


def _suite_summary(*suites: str) -> str:
    return "; ".join(f"{s} {REPLAY[s][0]}p/{REPLAY[s][1]}f/{REPLAY[s][2]}s" for s in suites)


DOMAIN_EVIDENCE: Final[dict[str, str]] = {
    "PDC-D1": ".tmp/proof/pdc-x-p2/replay.{log,xml}: " + _suite_summary("plug_03", "plug_04", "plug_final"),
    "PDC-D2": ".tmp/proof/pdc-x-p2/replay.{log,xml}: " + _suite_summary("ce_01", "ce_02", "memory_behavior"),
    "PDC-D3": ".tmp/proof/pdc-x-p2/replay.{log,xml}: " + _suite_summary("ebh_5"),
    "PDC-D4": ".tmp/proof/pdc-x-p2/replay.{log,xml}: " + _suite_summary("host_01"),
    "PDC-D5": ".tmp/proof/pdc-x-p2/replay.{log,xml}: " + _suite_summary("control_planes"),
    "PDC-D6": ".tmp/proof/pdc-x-p2/replay.{log,xml}: " + _suite_summary("tenant_x", "governance", "prod_q"),
    "PDC-D7": ".tmp/proof/pdc-x-p2/replay.{log,xml}: " + _suite_summary("bg_01", "sched_01"),
    "PDC-D8": "no qualification suite exists (PDC_CLOSED_WORLD_INVENTORY.md)",
}


class Row(NamedTuple):
    rid: str
    requirement: str
    path: str
    contract: str
    status: str  # PRESENT | PARTIAL | ABSENT
    cls: str
    cert: str | None
    task: str
    test: str
    frz: str
    deps: str = "—"


Q = "tests/qualification/pdc_x"

ROWS: Final[tuple[Row, ...]] = (
    # ---------------- PDC-D1 Agent Lifecycle ----------------
    Row(
        "PDC-D1-R001",
        "One AgentContract identity and version per agent",
        "intergrax/contracts/agent_contract_meta.py",
        "AgentContract",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "AC1-AC6-FINAL-AUDIT",
        "PDC-P7-D1 replay",
        f"{Q}/d1 identity uniqueness gate",
        "FRZ-OWN-01,FRZ-CTR-01,FRZ-CMP-02",
    ),
    Row(
        "PDC-D1-R002",
        "Typed agent authoring contract",
        "intergrax/agents/authoring",
        "AgentRuntime / on_next_step",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "AC1-AC6-FINAL-AUDIT",
        "PDC-P7-D1 replay",
        "existing ACP unit suites",
        "FRZ-TYP-01,FRZ-CTR-01",
    ),
    Row(
        "PDC-D1-R003",
        "Package identity, provenance and attestation verified before install",
        "intergrax/agent_distribution/package_attestation.py",
        "AgentPackageIdentity, AgentPackageAttestationStatement",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "AC1-AC6-FINAL-AUDIT",
        "PDC-P5-D1",
        f"{Q}/d1 tampered/unsigned package rejected",
        "FRZ-SEC-06,FRZ-PLG-05",
    ),
    Row(
        "PDC-D1-R004",
        "Single agent registry authority (runtime projection, not an install API)",
        "intergrax/runtime/registry/agent_registry.py",
        "AgentRegistry",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "AC1-AC6-FINAL-AUDIT",
        "PDC-P5-D1",
        f"{Q}/d1 no-shadow-registry gate",
        "FRZ-OWN-04,FRZ-OWN-05,FRZ-REG-03",
    ),
    Row(
        "PDC-D1-R005",
        "Capability declarations and dependency resolution",
        "intergrax/runtime/registry/agent_assembly_resolver.py",
        "validate_contract_metadata",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "AC1-AC6-FINAL-AUDIT",
        "PDC-P7-D1 replay",
        "existing registry unit suites",
        "FRZ-CTR-05,FRZ-PLG-01",
    ),
    Row(
        "PDC-D1-R006",
        "Agent assembly validation (contract + lifecycle metadata)",
        "intergrax/runtime/registry/agent_assembly_resolver.py",
        "validate_lifecycle_metadata",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "AC1-AC6-FINAL-AUDIT",
        "PDC-P7-D1 replay",
        "existing registry unit suites",
        "FRZ-CTR-01",
    ),
    Row(
        "PDC-D1-R007",
        "Tenant-scoped install, bind and activate",
        "intergrax/agent_distribution/admin_service.py",
        "AgentPlatformAdminService",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "TENANT-X",
        "PDC-P5-D1",
        f"{Q}/d1 cross-tenant install/bind/activate denied",
        "FRZ-TEN-04,FRZ-TEN-11",
    ),
    Row(
        "PDC-D1-R008",
        "Installed != configured != effective != authorized distinction",
        "intergrax/agent_distribution/installation.py",
        "InstallationState, AgentBinding",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "AC1-AC6-FINAL-AUDIT",
        "PDC-P7-D1 replay",
        "existing distribution unit suites",
        "FRZ-CFG-03",
    ),
    Row(
        "PDC-D1-R009",
        "Dynamic discovery and acquisition (AC-4); architecture doc still says READY_FOR_INDEPENDENT_AUDIT",
        "intergrax/agent_distribution/dynamic_acquisition.py",
        "DynamicAgentAcquisitionService",
        "PRESENT",
        "DOCUMENTATION_RECONCILIATION",
        "AC1-AC6-FINAL-AUDIT",
        "PDC-P5-DOC",
        "doc status matches AGENT_PLATFORM_AC1_AC6_FINAL_ARCHITECTURE_AUDIT",
        "FRZ-DOC-01,FRZ-PLG-08",
    ),
    Row(
        "PDC-D1-R010",
        "Version-pinned runtime binding",
        "intergrax/agent_distribution/runtime_revision.py",
        "RuntimeRevision",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "AC1-AC6-FINAL-AUDIT",
        "PDC-P7-D1 replay",
        "existing distribution unit suites",
        "FRZ-EVOL-04,FRZ-CMP-05",
    ),
    Row(
        "PDC-D1-R011",
        "UPGRADE is an explicit governed operation with a compatibility check",
        "intergrax/agent_distribution/installation_service.py",
        "none (implicit slot re-install)",
        "ABSENT",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D1 (ADR-PDC-01)",
        f"{Q}/d1 upgrade compat accept/reject + rollback",
        "FRZ-CMP-05,FRZ-EVOL-07,FRZ-PLG-08",
        "ADR-PDC-01",
    ),
    Row(
        "PDC-D1-R012",
        "Rollback safety",
        "intergrax/agent_distribution/activation.py",
        "ActivationService.rollback",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "AC1-AC6-FINAL-AUDIT",
        "PDC-P5-D1",
        f"{Q}/d1 rollback after failed activation",
        "FRZ-REC-02,FRZ-EVOL-07",
    ),
    Row(
        "PDC-D1-R013",
        "Revocation and emergency revocation reach active executions",
        "intergrax/agent_distribution/emergency_revocation_response.py",
        "AgentEmergencyRevocationService",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "AC1-AC6-FINAL-AUDIT",
        "PDC-P5-D1",
        f"{Q}/d1 revoke during execution -> fail closed",
        "FRZ-SEC-06,FRZ-GOV-05",
    ),
    Row(
        "PDC-D1-R014",
        "RETIRE is an explicit operation (state exists, no operation)",
        "intergrax/contracts/agent_lifecycle_state.py",
        "AgentLifecycleState.RETIRED",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D1 (ADR-PDC-01)",
        f"{Q}/d1 retire removes routing + keeps evidence",
        "FRZ-PLG-08,FRZ-OBS-05",
        "ADR-PDC-01",
    ),
    Row(
        "PDC-D1-R015",
        "Cross-host agent reuse",
        "intergrax/agent_distribution/agent_manager_command_facade.py",
        "agent manager facade",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "AC1-AC6-FINAL-AUDIT",
        "PDC-P7-D1 replay",
        "existing distribution integration suites",
        "FRZ-RPL-04",
    ),
    Row(
        "PDC-D1-R016",
        "Activation continuity across restart",
        "intergrax/agent_distribution/activation.py",
        "ActivationService",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "STATE-X",
        "PDC-P7-D1 replay",
        f"{Q}/xd_04 restart preserves active revision",
        "FRZ-REC-02,FRZ-STA-06",
    ),
    Row(
        "PDC-D1-R017",
        "Every lifecycle transition emits canonical evidence",
        "intergrax/agent_distribution/admin_service.py",
        "lifecycle events",
        "PARTIAL",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D1",
        f"{Q}/d1 transition evidence completeness",
        "FRZ-OBS-01,FRZ-TRC-08",
    ),
    Row(
        "PDC-D1-R018",
        "Skill disable/remove/upgrade (SKILL-X F-04) kept separate from agent lifecycle",
        "intergrax/skills/host_lifecycle.py",
        "SkillHostLifecycleService",
        "PARTIAL",
        "BLOCKED_EXACT_REASON",
        "SKILL-X",
        "OD-PDC-SCOPE-01 (implement vs authorized exclusion)",
        f"{Q}/d1 skill disable/remove (if implemented)",
        "FRZ-PLG-08,FRZ-DEBT-05",
    ),
    Row(
        "PDC-D1-R019",
        "Runtime factory beyond VENV_BUNDLE (OCI/sidecar deferred)",
        "intergrax/agent_distribution",
        "runtime factory adapter",
        "PARTIAL",
        "BLOCKED_EXACT_REASON",
        None,
        "OD-PDC-SCOPE-03 (authorized exclusion proposed)",
        "n/a if excluded",
        "FRZ-DEBT-05,FRZ-DEBT-06",
    ),
    Row(
        "PDC-D1-R020",
        "One owner for plugin discovery (core/plugins/discovery.py vs vendor_knowledge and rag entry-point loaders)",
        "intergrax/core/plugins/discovery.py",
        "PlatformPluginManifest",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P3-01 ownership decision -> PDC-P5-D1",
        f"{Q}/d1 single discovery owner gate",
        "FRZ-PLG-03,FRZ-OWN-05",
    ),
    Row(
        "PDC-D1-R021",
        "plug_03/plug_04/plug_final suites have no owning roadmap stage",
        "tests/qualification/plug_final",
        "plugin qualification catalogs",
        "PRESENT",
        "DOCUMENTATION_RECONCILIATION",
        None,
        "PDC-P3-01 ownership record",
        "inventory ownership record",
        "FRZ-REG-05,FRZ-DOC-03",
    ),
    # ---------------- PDC-D2 Context & Knowledge ----------------
    Row(
        "PDC-D2-R001",
        "Source identity and context provenance",
        "intergrax/runtime/nexus/context/context_models.py",
        "ContextProvenance",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "HARNESS-RESIDUAL/CE-01",
        "PDC-P7-D2 replay",
        "ce_01",
        "FRZ-TRC-05",
    ),
    Row(
        "PDC-D2-R002",
        "Tenant/workspace/session isolation of memory, RAG and context",
        "intergrax/memory/contracts/memory_control.py",
        "MemoryControlPlane",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "TENANT-X",
        "—",
        "tenant_x replay",
        "FRZ-TEN-04,FRZ-SEC-01",
    ),
    Row(
        "PDC-D2-R003",
        "Memory lifecycle has one owner",
        "intergrax/memory/contracts/memory_control.py",
        "MemoryControlPlane",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "EBH-2H",
        "PDC-P7-D2 replay",
        "memory_behavior",
        "FRZ-OWN-01,FRZ-STA-01",
    ),
    Row(
        "PDC-D2-R004",
        "RAG retrieval correctness",
        "intergrax/rag",
        "retrieval contracts",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "EBH-2G",
        "PDC-P7-D2 replay",
        "rag unit/integration suites",
        "FRZ-CTR-01,FRZ-TYP-01",
    ),
    Row(
        "PDC-D2-R005",
        "Index consistency and source freshness (only an embedding-version warning + optional reindex hook)",
        "intergrax/rag/governance/embedding_version_policy.py",
        "evaluate_ingest_embedding_version",
        "PARTIAL",
        "BLOCKED_EXACT_REASON",
        "EBH-2G",
        "OD-PDC-SCOPE-06 (freshness policy contract vs authorized exclusion)",
        f"{Q}/d2 stale source detected",
        "FRZ-STA-05,FRZ-DEBT-04",
    ),
    Row(
        "PDC-D2-R006",
        "Context assembly ownership",
        "intergrax/context/planner.py",
        "ContextPlanner",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "HARNESS-RESIDUAL/CE-01",
        "PDC-P7-D2 replay",
        "ce_01",
        "FRZ-OWN-01",
    ),
    Row(
        "PDC-D2-R007",
        "Mandatory instructions survive compaction and degradation",
        "intergrax/runtime/token_optimization",
        "protected regions",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D2",
        f"{Q}/d2 mandatory retention under pressure",
        "FRZ-CTR-01,FRZ-REG-03",
    ),
    Row(
        "PDC-D2-R008",
        "Retrieval-poisoning filter consumed on both paths; RAG service fallback is fail-open (False when profile absent)",
        "intergrax/tools/providers/rag/service.py",
        "retrieval_poisoning_defense_enabled",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D6 (shared with PDC-D6-R014)",
        f"{Q}/d6 missing profile -> filter enforced",
        "FRZ-SEC-08,FRZ-CFG-02",
    ),
    Row(
        "PDC-D2-R009",
        "Tool-output ingestion boundary",
        "intergrax/applications/_shared/application_security_wiring.py",
        "ToolInjectionDefenseMiddleware",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D2",
        f"{Q}/d2 injected tool output not promoted to instruction",
        "FRZ-SEC-08",
    ),
    Row(
        "PDC-D2-R010",
        "Token budget and hard model-window enforcement (CE-02 has no closure record)",
        "intergrax/runtime/nexus/context/context_compiler.py",
        "ContextCompiler, resolve_authoritative_model_budget",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        None,
        "PDC-P7-D2 replay of ce_02 (CE-02 closure via PDC)",
        "ce_02",
        "FRZ-CTR-01,FRZ-REL-10",
    ),
    Row(
        "PDC-D2-R011",
        "Degradation step TOKENIZER_HARD_TRIM drops entry_id/tool_calls/tool_call_id",
        "intergrax/runtime/nexus/context/degradation_ladder.py",
        "ContextDegradationPolicy",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D2 (ADR-PDC-02)",
        f"{Q}/d2 hard trim preserves tool-call linkage",
        "FRZ-TRC-05,FRZ-REC-04",
        "ADR-PDC-02",
    ),
    Row(
        "PDC-D2-R012",
        "Durable compaction semantics (CE-02 PARTIAL in gap audit)",
        "intergrax/runtime/token_optimization",
        "durable compaction",
        "PARTIAL",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D2 (ADR-PDC-02)",
        f"{Q}/d2 compaction round-trip + reconstruction",
        "FRZ-REC-04,FRZ-STA-06",
        "ADR-PDC-02",
    ),
    Row(
        "PDC-D2-R013",
        "Reconstruction and session continuity after restart",
        "intergrax/runtime/context_lifecycle",
        "UCL repository",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "STATE-X",
        "PDC-P7-D2 replay",
        f"{Q}/xd_07 restart reconstruction",
        "FRZ-REC-01,FRZ-TRC-09",
    ),
    Row(
        "PDC-D2-R014",
        "Sensitive data in context is classified and redacted",
        "intergrax/context/contracts.py",
        "ContextFragment sensitivity",
        "PARTIAL",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D2",
        f"{Q}/d2 sensitive fragment not leaked to provider",
        "FRZ-SEC-08",
    ),
    Row(
        "PDC-D2-R015",
        "Memory retention enforcement (retention is decision-only, no TTL; MEMORY-DURABILITY-LIFECYCLE planned)",
        "intergrax/memory",
        "MemoryRetentionPolicy",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        "EBH-2H",
        "PDC-P5-D2 (ADR-PDC-02)",
        f"{Q}/d2 retention expiry + vector tombstone",
        "FRZ-SEC-09",
        "ADR-PDC-02",
    ),
    Row(
        "PDC-D2-R016",
        "Citation and evidence fidelity",
        "intergrax/runtime/context",
        "CitationChainLink",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        None,
        "PDC-P7-D2 replay",
        "context golden harness",
        "FRZ-TRC-05",
    ),
    Row(
        "PDC-D2-R017",
        "ContextFragment name used by two different concepts (context contracts vs token-optimization pack)",
        "intergrax/runtime/token_optimization/context_pack.py",
        "ContextFragment (pack)",
        "PRESENT",
        "N/A_WITH_EVIDENCE",
        None,
        "—",
        "UNIFIED_CONTEXT_LIFECYCLE.md:390 classifies the pack separately",
        "FRZ-OWN-02",
    ),
    Row(
        "PDC-D2-R018",
        "Compaction never deletes audit evidence",
        "intergrax/runtime/context_lifecycle",
        "UclArtifactOwnership",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P6 (XD-09)",
        f"{Q}/xd_09 evidence survives compaction",
        "FRZ-AUD-05,FRZ-REC-10",
    ),
    # ---------------- PDC-D3 Model Runtime ----------------
    Row(
        "PDC-D3-R001",
        "Typed provider-neutral adapter ABI",
        "intergrax/llm_adapters/contracts/llm_adapter.py",
        "LLMAdapter",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "EBH-2E",
        "PDC-P7-D3 replay",
        "llm_adapters unit suites",
        "FRZ-CTR-01,FRZ-CTR-06",
    ),
    Row(
        "PDC-D3-R002",
        "ModelCatalog capability metadata",
        "intergrax/llm_adapters/registry/model_catalog.py",
        "ModelCatalog",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "EBH-5",
        "—",
        "ebh_5 replay",
        "FRZ-PLG-01",
    ),
    Row(
        "PDC-D3-R003",
        "Configuration-driven provider selection",
        "intergrax/llm_adapters/llm_provider_registry.py",
        "LLMAdapterRegistry",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "EBH-5",
        "—",
        "ebh_5 replay",
        "FRZ-CFG-01,FRZ-PLG-04",
    ),
    Row(
        "PDC-D3-R004",
        "Routing and failover bounded",
        "intergrax/llm_adapters/registry/failover_adapter.py",
        "FailoverLLMAdapter",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "EBH-2E",
        "PDC-P5-D3",
        f"{Q}/d3 failover exhaustion bounded",
        "FRZ-REL-07,FRZ-REL-11",
    ),
    Row(
        "PDC-D3-R005",
        "Model version pinning and runtime attribution",
        "intergrax/llm_adapters/contracts/llm_profile.py",
        "LLMProfile",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "EBH-2E",
        "PDC-P7-D3 replay",
        "llm_adapters unit suites",
        "FRZ-TRC-04,FRZ-GOV-08",
    ),
    Row(
        "PDC-D3-R006",
        "Streaming / tool-call streaming parity or explicit capability fail-closed (Ollama adapters lack stream_with_tools)",
        "intergrax/llm_adapters/base/base_llm_adapter.py",
        "stream_messages / stream_with_tools",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D3 (ADR-PDC-03)",
        f"{Q}/d3 unsupported tool streaming fails closed",
        "FRZ-CTR-01,FRZ-REL-06",
        "ADR-PDC-03",
    ),
    Row(
        "PDC-D3-R007",
        "Structured outputs",
        "intergrax/llm_adapters/contracts/structured_result.py",
        "generate_structured",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "EBH-2E",
        "—",
        "llm_adapters unit suites",
        "FRZ-TYP-01",
    ),
    Row(
        "PDC-D3-R008",
        "Tool-call normalization",
        "intergrax/llm_adapters/contracts/tool_call.py",
        "LLMToolCall",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "EBH-2E",
        "PDC-P7-D3 replay",
        "strict_tool_call_validation suites",
        "FRZ-TYP-01",
    ),
    Row(
        "PDC-D3-R009",
        "Refusal semantics: typed finish reason exists, refusal payload is a string",
        "intergrax/llm_adapters/contracts/finish_reason.py",
        "LLMFinishReason.REFUSAL / CONTENT_FILTER",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D3",
        f"{Q}/d3 refusal normalized across providers",
        "FRZ-REL-06,FRZ-TYP-05",
    ),
    Row(
        "PDC-D3-R010",
        "Usage, token and cost accounting",
        "intergrax/llm_adapters/tracking/llm_usage_track.py",
        "LLMUsageTracker",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "EBH-2E",
        "PDC-P7-D3 replay",
        "llm_adapters unit suites",
        "FRZ-OBS-06",
    ),
    Row(
        "PDC-D3-R011",
        "Cancellation explicit per provider (seam beside the ABI; Ollama has no native remote cancel)",
        "intergrax/llm_adapters/registry/registration_contract.py",
        "ProviderExternalOperationSeam",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "EBH-2E",
        "PDC-P5-D3",
        f"{Q}/d3 cancel outcome typed (terminated / transport_closed / not_supported)",
        "FRZ-REL-04,FRZ-REL-05",
    ),
    Row(
        "PDC-D3-R012",
        "Timeouts",
        "intergrax/llm_adapters/contracts/call_config.py",
        "call config",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "EBH-2E",
        "PDC-P7-D3 replay",
        "llm_adapters unit suites",
        "FRZ-REL-01",
    ),
    Row(
        "PDC-D3-R013",
        "Provider error taxonomy",
        "intergrax/llm_adapters",
        "typed provider errors",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "EBH-2E",
        "PDC-P7-D3 replay",
        "llm_adapters unit suites",
        "FRZ-REL-06",
    ),
    Row(
        "PDC-D3-R014",
        "Context-window enforcement",
        "intergrax/llm_adapters/registry/context_window.py",
        "context window resolution",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        None,
        "PDC-P7-D3 replay (ce_02)",
        "ce_02",
        "FRZ-REL-10",
    ),
    Row(
        "PDC-D3-R015",
        "Capability mismatch fails closed",
        "intergrax/llm_adapters/contracts/llm_adapter.py",
        "supports_* flags",
        "PARTIAL",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D3",
        f"{Q}/d3 mismatch denied before call",
        "FRZ-CFG-02",
    ),
    Row(
        "PDC-D3-R016",
        "Provider replaceability",
        "intergrax/llm_adapters",
        "LLMAdapter",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "EBH-5",
        "—",
        "ebh_5 replay",
        "FRZ-RPL-01,FRZ-RPL-04",
    ),
    Row(
        "PDC-D3-R017",
        "Local/offline models and native Ollama",
        "intergrax/llm_adapters/providers/native_ollama_adapter.py",
        "NativeOllamaAdapter",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "EBH-2E",
        "PDC-P7-D3 replay; live parity = Path B",
        "Ollama parity satellites",
        "FRZ-RPL-02",
    ),
    Row(
        "PDC-D3-R018",
        "Tenant-aware model configuration",
        "intergrax/llm_adapters/llm_provider_registry.py",
        "LLMAdapterRegistry",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "TENANT-X",
        "—",
        "tenant_x replay",
        "FRZ-TEN-05",
    ),
    Row(
        "PDC-D3-R019",
        "No vendor selection hidden in core (provider DEFAULT_MODEL constants, LLMProfile.lab, rag default embedding provider)",
        "intergrax/llm_adapters/providers",
        "provider defaults",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "EBH-5",
        "PDC-P7-D3 against VBN CORE_HARDCODING_FINDINGS_REGISTER",
        "VBN register classification of each default",
        "FRZ-CFG-08,FRZ-CTR-06",
    ),
    Row(
        "PDC-D3-R020",
        "Provider plugin registration layer (LLM-PROVIDER-PLUGIN-1 Planned)",
        "intergrax/llm_adapters/registry",
        "LLMProviderRegistration",
        "ABSENT",
        "BLOCKED_EXACT_REASON",
        None,
        "OD-PDC-SCOPE-03 (authorized exclusion proposed)",
        "n/a if excluded",
        "FRZ-PLG-02,FRZ-DEBT-05",
    ),
    Row(
        "PDC-D3-R021",
        "Live provider qualification",
        "intergrax/llm_adapters/providers",
        "live providers",
        "PRESENT",
        "N/A_WITH_EVIDENCE",
        None,
        "—",
        "Path B (roadmap §1.2 / §2.0.3; PROD-Q Dimension B)",
        "FRZ-PRD-01",
    ),
    Row(
        "PDC-D3-R022",
        "Streaming capability flags misreport: six adapters stream but supports_streaming() returns False",
        "intergrax/llm_adapters/base/base_llm_adapter.py",
        "supports_streaming",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D3 (ADR-PDC-03)",
        f"{Q}/d3 capability flags match implemented methods",
        "FRZ-CTR-01,FRZ-REG-09",
        "ADR-PDC-03",
    ),
    Row(
        "PDC-D3-R023",
        "Context-window tables duplicated inside providers and ModelCatalog",
        "intergrax/llm_adapters/providers/ollama_adapter.py",
        "context window data",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P3-01 ownership decision -> PDC-P5-D3",
        f"{Q}/d3 single context-window source",
        "FRZ-OWN-05",
    ),
    # ---------------- PDC-D4 Host Runtime ----------------
    Row(
        "PDC-D4-R001",
        "Canonical ApplicationEnvironmentProfile and HostedApplicationProfile",
        "intergrax/hosting/contracts/profile.py",
        "HostedApplicationProfile",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "HARNESS-W7/HOST-01",
        "PDC-P7-D4 replay",
        "host_01",
        "FRZ-CTR-01,FRZ-OWN-02",
    ),
    Row(
        "PDC-D4-R002",
        "Single effective composition root",
        "intergrax/applications/_shared/environment_wiring.py",
        "wire_application_environment",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "HARNESS-W7/HOST-01",
        "PDC-P7-D4 replay",
        "host_01",
        "FRZ-OWN-03,FRZ-BND-05",
    ),
    Row(
        "PDC-D4-R003",
        "Startup dependency ordering and rollback",
        "intergrax/hosting/engine/engine.py",
        "HostedApplicationEngine",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D4",
        f"{Q}/d4 startup failure rolls back in reverse order",
        "FRZ-REL-10,FRZ-OBS-05",
    ),
    Row(
        "PDC-D4-R004",
        "Readiness and degraded states",
        "intergrax/hosting/engine/health.py",
        "HostedApplicationReadinessService",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D4",
        f"{Q}/d4 degraded component reported",
        "FRZ-OBS-05",
    ),
    Row(
        "PDC-D4-R005",
        "Graceful shutdown with in-flight drain",
        "intergrax/hosting/shutdown.py",
        "HostedApplicationActiveWorkController",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D4",
        f"{Q}/xd_05 drain with background work",
        "FRZ-REL-04",
    ),
    Row(
        "PDC-D4-R006",
        "Instance ownership guard",
        "intergrax/hosting/instance",
        "InstanceGuard",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D4",
        f"{Q}/d4 second instance rejected",
        "FRZ-STA-01",
    ),
    Row(
        "PDC-D4-R007",
        "Restart supervision with an OS-process crash proof (APP-HOST-5E Planned)",
        "intergrax/hosting/supervisor/supervisor.py",
        "HostedApplicationSupervisor",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D4",
        f"{Q}/d4 subprocess kill + supervised restart",
        "FRZ-REC-01,FRZ-REC-02",
    ),
    Row(
        "PDC-D4-R008",
        "Failure classification",
        "intergrax/hosting/supervisor/classification.py",
        "HostedApplicationExitClassifier",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        None,
        "PDC-P7-D4 replay",
        "hosting unit suites",
        "FRZ-REL-06",
    ),
    Row(
        "PDC-D4-R009",
        "Hosting lifecycle metrics through canonical observability (APP-HOST-3D; no metrics today)",
        "intergrax/hosting",
        "hosting events",
        "ABSENT",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D4 (ADR-PDC-04)",
        f"{Q}/d4 lifecycle metrics emitted via observability port",
        "FRZ-OBS-01,FRZ-OBS-05",
        "ADR-PDC-04",
    ),
    Row(
        "PDC-D4-R010",
        "Restart preserves configuration/profile digest generically (APP-HOST-5D)",
        "intergrax/hosting/supervisor/restart.py",
        "HostedApplicationRestartPolicyEvaluator",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D4 (ADR-PDC-04)",
        f"{Q}/d4 digest mismatch on restart rejected",
        "FRZ-REC-02,FRZ-TRC-11",
        "ADR-PDC-04",
    ),
    Row(
        "PDC-D4-R011",
        "InteractionProfile facade (APP-HOST-6; no class exists)",
        "intergrax/hosting",
        "InteractionProfile",
        "ABSENT",
        "BLOCKED_EXACT_REASON",
        None,
        "OD-PDC-SCOPE-02 (authorized exclusion proposed; ApplicationEnvironmentProfile authority PRESERVE)",
        "n/a if excluded",
        "FRZ-DEBT-05,FRZ-DEBT-06",
    ),
    Row(
        "PDC-D4-R012",
        "OS service-manager adapters (APP-HOST-7)",
        "intergrax/hosting",
        "OS services contract",
        "ABSENT",
        "BLOCKED_EXACT_REASON",
        None,
        "OD-PDC-SCOPE-02 (authorized exclusion proposed)",
        "n/a if excluded",
        "FRZ-DEBT-05",
    ),
    Row(
        "PDC-D4-R013",
        "Host plugins in HostedApplicationProfile (APP-HOST-9E)",
        "intergrax/hosting/contracts/profile.py",
        "hosting.plugin events only",
        "PARTIAL",
        "BLOCKED_EXACT_REASON",
        None,
        "OD-PDC-SCOPE-02 (authorized exclusion proposed)",
        "n/a if excluded",
        "FRZ-PLG-01,FRZ-DEBT-05",
    ),
    Row(
        "PDC-D4-R014",
        "Stable public hosting API declaration (APP-HOST-9F)",
        "intergrax/hosting",
        "public hosting API",
        "ABSENT",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D4 (ADR-PDC-04)",
        f"{Q}/d4 public API manifest + COMPAT gate",
        "FRZ-CTR-04,FRZ-DOC-05,FRZ-CMP-01",
        "ADR-PDC-04",
    ),
    Row(
        "PDC-D4-R015",
        "Generic HostDeploymentProfile carries LKW-specific fields (TL-FIX-C)",
        "intergrax/applications/contracts/environment_profile/sub_profiles.py",
        "HostDeploymentProfile",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D4",
        f"{Q}/d4 no product vocabulary in generic profile",
        "FRZ-BND-04,FRZ-CTR-06",
    ),
    Row(
        "PDC-D4-R016",
        "TL-FIX-D implemented in code (bind_execution_adapter) but plan row still ACCEPTED/PLANNED",
        "intergrax/fastapi_core/runs/default_service.py",
        "bind_execution_adapter",
        "PRESENT",
        "DOCUMENTATION_RECONCILIATION",
        None,
        "PDC-P5-DOC",
        "plan status matches code",
        "FRZ-DOC-01",
    ),
    Row(
        "PDC-D4-R017",
        "No host-owned execution authority",
        "intergrax/hosting",
        "host execution boundary",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "HARNESS-W7/HOST-01",
        "PDC-P7-D4 replay",
        "host_01",
        "FRZ-EXE-01,FRZ-EXE-02",
    ),
    Row(
        "PDC-D4-R018",
        "Tenant-specific host configuration",
        "intergrax/applications/_shared/environment_wiring.py",
        "environment profile",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "TENANT-X",
        "—",
        "tenant_x replay",
        "FRZ-TEN-05",
    ),
    Row(
        "PDC-D4-R019",
        "Sandbox/shadow app isolation is not a universal production guarantee; plan marks APP-CON-8 Done",
        "intergrax/applications/_shared",
        "SandboxSessionManager / ShadowWorkspaceManager",
        "PARTIAL",
        "DOCUMENTATION_RECONCILIATION",
        None,
        "PDC-P5-DOC",
        "bounded claim stated consistently in arch + plan",
        "FRZ-SEC-07,FRZ-DOC-02",
    ),
    # ---------------- PDC-D5 Evaluation ----------------
    Row(
        "PDC-D5-R001",
        "Typed EvalDataset / EvalScenario / EvalRun contracts",
        "intergrax/eval",
        "none",
        "ABSENT",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/d5 contract schema + persistence",
        "FRZ-CTR-01,FRZ-TYP-01",
        "ADR-PDC-05",
    ),
    Row(
        "PDC-D5-R002",
        "Dataset versioning with content digest (GoldenDatasetAsset has free-string version)",
        "intergrax/runtime/architecture/evaluation_assets.py",
        "GoldenDatasetAsset",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        "CTRL-X",
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/d5 digest mismatch rejected",
        "FRZ-CMP-03,FRZ-REG-06",
        "ADR-PDC-05",
    ),
    Row(
        "PDC-D5-R003",
        "Evaluation scenario identity",
        "intergrax/runtime/architecture/evaluation_scenario_loader.py",
        "scenario ids",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/d5 scenario identity stable",
        "FRZ-TYP-01",
        "ADR-PDC-05",
    ),
    Row(
        "PDC-D5-R004",
        "Agent/model/prompt/profile/code-SHA references on every run (EvalResult lacks them)",
        "intergrax/eval/eval_result.py",
        "EvalResult",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/d5 run provenance complete",
        "FRZ-TRC-04,FRZ-TRC-05",
        "ADR-PDC-05",
    ),
    Row(
        "PDC-D5-R005",
        "Task-success, tool-use, groundedness and citation evaluators behind one contract",
        "intergrax/rag/evaluation/golden_harness.py",
        "per-harness JSON loaders",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/d5 evaluator registry covers kinds",
        "FRZ-OWN-05,FRZ-PLG-01",
        "ADR-PDC-05",
    ),
    Row(
        "PDC-D5-R006",
        "Safety evaluation hook (no safety/harm evaluator exists)",
        "intergrax/eval",
        "none",
        "ABSENT",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/d5 safety evaluator kind via guardrail provider",
        "FRZ-SEC-07",
        "ADR-PDC-05",
    ),
    Row(
        "PDC-D5-R007",
        "Deterministic vs model-assisted evaluator separation",
        "intergrax/runtime/architecture/evaluation_automation.py",
        "EvaluatorType",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/d5 judge results isolated from rule results",
        "FRZ-TYP-05",
        "ADR-PDC-05",
    ),
    Row(
        "PDC-D5-R008",
        "Human review queue",
        "intergrax/runtime/evaluation/human_review_sample_queue.py",
        "human review queue",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "CTRL-X",
        "PDC-P7-D5 replay",
        "evaluation unit suites",
        "FRZ-GOV-06",
    ),
    Row(
        "PDC-D5-R009",
        "Reproducible evaluation runs",
        "intergrax/eval/eval_runner.py",
        "EvalRunner",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/xd_08 digest-identical rerun",
        "FRZ-REG-06",
        "ADR-PDC-05",
    ),
    Row(
        "PDC-D5-R010",
        "Baseline vs candidate comparison (only a pass-rate delta exists)",
        "intergrax/runtime/architecture/evaluation_registry_trends.py",
        "EvaluationComparisonSummary",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/d5 candidate regression detected",
        "FRZ-EVOL-06",
        "ADR-PDC-05",
    ),
    Row(
        "PDC-D5-R011",
        "Quality regression detection",
        "intergrax/runtime/architecture/evaluation_registry_trends.py",
        "EvaluationComparisonSummary",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/d5 threshold breach flagged",
        "FRZ-EVOL-06",
        "ADR-PDC-05",
    ),
    Row(
        "PDC-D5-R012",
        "Cost-to-success and latency/quality comparison",
        "intergrax/eval/eval_result.py",
        "tokens, cost fields",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/d5 comparison carries cost and latency",
        "FRZ-OBS-06",
        "ADR-PDC-05",
    ),
    Row(
        "PDC-D5-R013",
        "Evaluation provenance in the canonical evidence store",
        "intergrax/runtime/evidence/eval_evidence_contracts.py",
        "EvalEvidenceReport",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/d5 run recorded as evidence",
        "FRZ-OBS-01,FRZ-AUD-05",
        "ADR-PDC-05",
    ),
    Row(
        "PDC-D5-R014",
        "Promotion gates G0-G2 are file-existence checks; evaluation must stay advisory and never authorize",
        "scripts/gates/check_mvp_promotion_gates.py",
        "evaluate_agent_promotion",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/xd_08 eval PASS cannot authorize promotion",
        "FRZ-GOV-01,FRZ-GOV-02,FRZ-REG-09",
        "ADR-PDC-05",
    ),
    Row(
        "PDC-D5-R015",
        "tenant_id on OnlineEvaluationObservation (TENANT-X debt recorded by CTRL-X)",
        "intergrax/runtime/architecture/online_evaluation_registry.py",
        "OnlineEvaluationObservation",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        "CTRL-X",
        "PDC-P5-D5",
        f"{Q}/d5 cross-tenant observation denied",
        "FRZ-TEN-01,FRZ-TEN-07",
    ),
    Row(
        "PDC-D5-R016",
        "CRITIC_VERIFICATION findings are legacy lineage; canonical authority is DECISION_SYSTEM",
        "docs/project/architecture/CRITIC_VERIFICATION.md",
        "legacy docs",
        "PRESENT",
        "DOCUMENTATION_RECONCILIATION",
        None,
        "PDC-P5-DOC",
        "audit findings re-dispositioned as legacy",
        "FRZ-DOC-02",
    ),
    Row(
        "PDC-D5-R017",
        "Production-feedback boundary of evaluation",
        "docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md",
        "online evaluation",
        "PARTIAL",
        "DOCUMENTATION_RECONCILIATION",
        None,
        "PDC-P5-DOC (ADR-PDC-05)",
        "boundary documented",
        "FRZ-DOC-01",
    ),
    Row(
        "PDC-D5-R018",
        "File-based evaluation registries have no concurrency semantics (DX-07)",
        "intergrax/runtime/architecture/online_evaluation_registry.py",
        "FileOnlineEvaluationRegistry",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D5 (ADR-PDC-05)",
        f"{Q}/d5 concurrent writes safe or rejected",
        "FRZ-STA-03",
        "ADR-PDC-05",
    ),
    # ---------------- PDC-D6 Runtime Security ----------------
    Row(
        "PDC-D6-R001",
        "Principal and workload identity",
        "intergrax/contracts/agent_run.py",
        "RequestIdentity",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "TENANT-X",
        "PDC-P7-D6 replay",
        "tenant_x, governance",
        "FRZ-TEN-01",
    ),
    Row(
        "PDC-D6-R002",
        "Authentication (architecture scope)",
        "intergrax/applications/_shared/harness_principal.py",
        "HarnessAuthenticatedPrincipal",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        None,
        "PDC-P7-D6 replay; live = Path B",
        "prod_q architecture gates",
        "FRZ-SEC-04",
    ),
    Row(
        "PDC-D6-R003",
        "Single authorization / policy authority",
        "intergrax/runtime/policy/policy_engine.py",
        "PolicyEngine",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "GOV-X2",
        "PDC-P7-D6 replay",
        "governance",
        "FRZ-GOV-01,FRZ-OWN-04",
    ),
    Row(
        "PDC-D6-R004",
        "Delegation and authority narrowing",
        "intergrax/contracts/delegation_authority.py",
        "EffectiveDelegationAuthority",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "GOV-X2",
        "PDC-P5-D6",
        f"{Q}/d6 child widening denied",
        "FRZ-GOV-03,FRZ-GOV-04",
    ),
    Row(
        "PDC-D6-R005",
        "Tenant isolation evidence for FRZ-SEC-01",
        "intergrax/runtime/security/tenant_scope.py",
        "tenant scope",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "TENANT-X",
        "—",
        "tenant_x replay",
        "FRZ-SEC-01,FRZ-TEN-12",
    ),
    Row(
        "PDC-D6-R006",
        "Secret references with late resolution",
        "intergrax/integrations/contracts/credential.py",
        "CredentialRef, CredentialResolver",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D6",
        f"{Q}/d6 secret resolved only at use site",
        "FRZ-SEC-02,FRZ-TEN-06",
    ),
    Row(
        "PDC-D6-R007",
        "Raw secrets never cross layers, logs or traces (no static gate exists)",
        "intergrax/core/security/secret_safety.py",
        "SecretSafeValidationPolicy",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D6 (ADR-PDC-06)",
        f"{Q}/d6 caplog/trace secret scan + static gate",
        "FRZ-SEC-03,FRZ-REG-09",
        "ADR-PDC-06",
    ),
    Row(
        "PDC-D6-R008",
        "Critical-action signing falls back to literal 'harness-dev-signing-key' (SEC-BND-05)",
        "intergrax/applications/_shared/critical_action_signing_wiring.py",
        "signing secret env",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D6 (ADR-PDC-06)",
        f"{Q}/d6 missing signing secret fails closed outside lab",
        "FRZ-CFG-02,FRZ-SEC-02",
        "ADR-PDC-06",
    ),
    Row(
        "PDC-D6-R009",
        "Credential scope and revocation",
        "intergrax/integrations/credentials/broker.py",
        "ScopedCredentialBroker",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "TENANT-X",
        "PDC-P5-D6",
        f"{Q}/xd_06 revoked credential mid-retry denied",
        "FRZ-SEC-02,FRZ-TEN-06",
    ),
    Row(
        "PDC-D6-R010",
        "Sandbox enforcement",
        "intergrax/runtime/sandbox/isolation_gate.py",
        "SandboxProfile",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        None,
        "PDC-P7-D6 replay (AW-7C PHYSQ scoped); provider live = Path B",
        "sandbox egress integration",
        "FRZ-SEC-05",
    ),
    Row(
        "PDC-D6-R011",
        "Filesystem and process isolation",
        "intergrax/runtime/sandbox/contracts.py",
        "sandbox contracts",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        None,
        "PDC-P7-D6 replay",
        "sandbox unit suites",
        "FRZ-SEC-05",
    ),
    Row(
        "PDC-D6-R012",
        "Network egress controls",
        "intergrax/contracts/sandbox_network_egress.py",
        "NetworkEgressAllowlist",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        None,
        "PDC-P7-D6 replay",
        "egress qualification (reference substrate)",
        "FRZ-SEC-05",
    ),
    Row(
        "PDC-D6-R013",
        "Tool-input injection guard assert_tool_input_safe has zero consumers",
        "intergrax/runtime/security/tool_injection_guard.py",
        "assert_tool_input_safe",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D6 (ADR-PDC-06)",
        f"{Q}/d6 single tool-input defense path",
        "FRZ-OWN-05,FRZ-SEC-08",
        "ADR-PDC-06",
    ),
    Row(
        "PDC-D6-R014",
        "Retrieval-poisoning defense must fail closed when the profile is absent (SEC-BND-04)",
        "intergrax/tools/providers/rag/service.py",
        "retrieval_poisoning_defense_enabled",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D6 (ADR-PDC-06)",
        f"{Q}/d6 missing profile -> filter enforced",
        "FRZ-SEC-08,FRZ-CFG-02",
        "ADR-PDC-06",
    ),
    Row(
        "PDC-D6-R015",
        "Data exfiltration / propagation across provider boundaries",
        "intergrax/runtime/security",
        "defense plugins",
        "PARTIAL",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D6",
        f"{Q}/d6 tenant data not sent to unapproved provider",
        "FRZ-SEC-08",
    ),
    Row(
        "PDC-D6-R016",
        "Plugin entry-point admission is structural only (no signature/provenance); agent packages are attested",
        "intergrax/core/catalog_bootstrap.py",
        "PluginAdmissionRejection",
        "PARTIAL",
        "BLOCKED_EXACT_REASON",
        None,
        "OD-PDC-SCOPE-07 (plugin provenance enforcement vs authorized scope)",
        f"{Q}/d6 unsigned plugin rejected (if in scope)",
        "FRZ-SEC-06,FRZ-PLG-05",
    ),
    Row(
        "PDC-D6-R017",
        "MCP exposure security",
        "intergrax/applications/_shared/mcp_application_server.py",
        "MCP exposure engine",
        "PRESENT",
        "BLOCKED_EXACT_REASON",
        None,
        "await MCP-SERVER-X P8 certification",
        "MCP-SERVER-X qualification package",
        "FRZ-SEC-04,FRZ-TEN-11",
        "MCP-SERVER-X",
    ),
    Row(
        "PDC-D6-R018",
        "Side-effect governance",
        "intergrax/runtime/policy",
        "side-effect authorization",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "GOV-X2",
        "PDC-P7-D6 replay",
        "governance",
        "FRZ-GOV-05,FRZ-TRC-06",
    ),
    Row(
        "PDC-D6-R019",
        "Cross-tenant negative tests across D1-D8",
        "tests/qualification/pdc_x",
        "cross-domain matrix",
        "ABSENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "TENANT-X",
        "PDC-P6 (XD-02/06/10)",
        f"{Q}/xd_02, xd_06, xd_10",
        "FRZ-TEN-12,FRZ-SEC-01",
    ),
    Row(
        "PDC-D6-R020",
        "Security audit trail is in-memory only (SEC-BND-06)",
        "intergrax/runtime/security/security_audit_trail.py",
        "SecurityAuditTrail",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D6 (ADR-PDC-06)",
        f"{Q}/d6 audit trail survives restart",
        "FRZ-AUD-05,FRZ-OBS-01",
        "ADR-PDC-06",
    ),
    Row(
        "PDC-D6-R021",
        "Consolidated security architecture and threat model (no doc pair; broken IDENTITY_TRUST.md link)",
        "docs/project/maintainers/architecture/EXECUTION_ENGINE_SECURITY_BOUNDARY_AND_THREAT_MODEL.md",
        "threat model",
        "PARTIAL",
        "DOCUMENTATION_RECONCILIATION",
        None,
        "PDC-P5-DOC (ADR-PDC-06)",
        "threat model covers D1-D8 surfaces",
        "FRZ-SEC-07,FRZ-DOC-02",
    ),
    Row(
        "PDC-D6-R022",
        "Retention/deletion/redaction responsibility explicit",
        "intergrax",
        "retention policies",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D8 + PDC-P5-D2 (ADR-PDC-08, ADR-PDC-02)",
        f"{Q}/xd_10 TTL deletion honored",
        "FRZ-SEC-09",
        "ADR-PDC-08",
    ),
    Row(
        "PDC-D6-R023",
        "Encryption / residency responsibility matrix",
        "docs/project/maintainers/architecture",
        "responsibility matrix",
        "ABSENT",
        "DOCUMENTATION_RECONCILIATION",
        None,
        "PDC-P5-DOC (ADR-PDC-06)",
        "responsibility matrix published",
        "FRZ-SEC-10",
    ),
    Row(
        "PDC-D6-R024",
        "Fail-closed security defaults",
        "intergrax/applications/_shared/security_assembly_resolver.py",
        "security assembly",
        "PARTIAL",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D6",
        f"{Q}/d6 missing security config fails closed",
        "FRZ-CFG-02",
    ),
    Row(
        "PDC-D6-R025",
        "HARNESS-FINAL 'ENTERPRISE QUALIFIED' for plugin trust/secrets/security contradicts FRZ-SEC OPEN",
        "docs/project/maintainers/qualification/HARNESS_FINAL_CURRENT_HEAD_CERTIFICATION.md",
        "certification text",
        "PRESENT",
        "DOCUMENTATION_RECONCILIATION",
        None,
        "PDC-P5-DOC",
        "scoped wording reconciled",
        "FRZ-DOC-02",
    ),
    Row(
        "PDC-D6-R026",
        "Production auth-bypass live proof",
        "intergrax/applications/_shared/harness_principal.py",
        "live auth",
        "PRESENT",
        "N/A_WITH_EVIDENCE",
        None,
        "—",
        "Path B live slice (checklist lines 57-68)",
        "FRZ-SEC-04",
    ),
    Row(
        "PDC-D6-R027",
        "Demo attestation signing seed reachable only in lab/proof paths",
        "intergrax/runtime/execution_evidence/attestor.py",
        "Ed25519HostAttestor",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D6",
        f"{Q}/d6 demo seed unreachable in strict profile",
        "FRZ-CFG-05,FRZ-SEC-02",
    ),
    # ---------------- PDC-D7 Background & Scheduling ----------------
    Row(
        "PDC-D7-R001",
        "Task definition and queue contract",
        "intergrax/queueing/contracts/task_queue.py",
        "TaskQueue, TaskRequest",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "HARNESS-W8/BG-01",
        "—",
        "bg_01 replay",
        "FRZ-CTR-01",
    ),
    Row(
        "PDC-D7-R002",
        "Worker admission through canonical execution",
        "intergrax/runtime/background_execution/reentry_admission.py",
        "BackgroundExecutionReentry",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "HARNESS-W8/BG-01",
        "—",
        "bg_01 replay",
        "FRZ-EXE-02,FRZ-EXE-07",
    ),
    Row(
        "PDC-D7-R003",
        "Durable dispatch",
        "intergrax/queueing/worker/dispatcher.py",
        "dispatcher",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "HARNESS-W8/BG-01",
        "—",
        "bg_01 replay",
        "FRZ-REC-01",
    ),
    Row(
        "PDC-D7-R004",
        "One-shot delayed execution",
        "intergrax/runtime/long_running/scheduled_resume.py",
        "ScheduledResume",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "HARNESS-RESIDUAL/SCHED-01",
        "PDC-P7-D7 replay",
        "sched_01",
        "FRZ-EXE-07",
    ),
    Row(
        "PDC-D7-R005",
        "Queue-level delay/ETA (TaskRequest has none; delayed work owned by SchedulerLedger)",
        "intergrax/queueing/contracts/task_queue.py",
        "TaskRequest",
        "ABSENT",
        "N/A_WITH_EVIDENCE",
        "HARNESS-RESIDUAL/SCHED-01",
        "—",
        "SCHED-01 one-shot delayed resume is the delayed-execution owner",
        "FRZ-OWN-01",
    ),
    Row(
        "PDC-D7-R006",
        "Recurrence (interval/cron) as occurrences on SchedulerLedger; no second engine",
        "intergrax/runtime/long_running/scheduler.py",
        "none",
        "ABSENT",
        "BLOCKED_EXACT_REASON",
        None,
        "OD-PDC-SCOPE-05 (implement via ADR-PDC-07 vs authorized exclusion)",
        f"{Q}/d7 recurrence occurrences + DST",
        "FRZ-EXE-07,FRZ-DEBT-05",
        "ADR-PDC-07",
    ),
    Row(
        "PDC-D7-R007",
        "Time-zone and DST semantics for recurrence",
        "intergrax/runtime/long_running/scheduled_resume.py",
        "run_at_utc",
        "ABSENT",
        "BLOCKED_EXACT_REASON",
        None,
        "OD-PDC-SCOPE-05",
        f"{Q}/d7 DST transition",
        "FRZ-DEBT-05",
        "PDC-D7-R006",
    ),
    Row(
        "PDC-D7-R008",
        "Recurrence misfire / catch-up policy",
        "intergrax/runtime/long_running/scheduler.py",
        "misfire fires once (one-shot)",
        "PARTIAL",
        "BLOCKED_EXACT_REASON",
        None,
        "OD-PDC-SCOPE-05",
        f"{Q}/d7 misfire policy",
        "FRZ-DEBT-05",
        "PDC-D7-R006",
    ),
    Row(
        "PDC-D7-R009",
        "Aggregate retry budget",
        "intergrax/queueing/worker/retry_policy.py",
        "RetryPolicy",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "TOOL-LIFE-X",
        "PDC-P5-D7",
        f"{Q}/d7 retry budget exhaustion terminal",
        "FRZ-REL-02,FRZ-REL-07",
    ),
    Row(
        "PDC-D7-R010",
        "Idempotency and duplicate handling",
        "intergrax/queueing/contracts/task_queue.py",
        "idempotency_key",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "HARNESS-W8/BG-01",
        "—",
        "bg_01 replay",
        "FRZ-REL-03",
    ),
    Row(
        "PDC-D7-R011",
        "At-least-once delivery consequences documented",
        "docs/project/architecture/BACKGROUND_TASKS.md",
        "delivery semantics",
        "PARTIAL",
        "DOCUMENTATION_RECONCILIATION",
        None,
        "PDC-P5-DOC",
        "delivery semantics section",
        "FRZ-REL-03,FRZ-DOC-01",
    ),
    Row(
        "PDC-D7-R012",
        "Lease/fencing proven across processes (today: two schedulers in one process)",
        "intergrax/runtime/long_running/scheduler_claim.py",
        "ScheduledResumeClaim",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        "HARNESS-RESIDUAL/SCHED-01",
        "PDC-P5-D7",
        f"{Q}/d7 two-process fencing over SQLite",
        "FRZ-REL-09,FRZ-STA-03",
    ),
    Row(
        "PDC-D7-R013",
        "Dead-letter / terminal-failed semantics (no DLQ anywhere)",
        "intergrax/queueing/contracts/task_queue.py",
        "none",
        "ABSENT",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D7 (ADR-PDC-07)",
        f"{Q}/d7 poison task reaches terminal dead-letter state",
        "FRZ-REL-07,FRZ-REL-09",
        "ADR-PDC-07",
    ),
    Row(
        "PDC-D7-R014",
        "Graceful worker shutdown with host drain",
        "intergrax/background_tasks/worker_runtime.py",
        "WorkerRuntime",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P6 (XD-05)",
        f"{Q}/xd_05 drain + lease release",
        "FRZ-REL-04",
    ),
    Row(
        "PDC-D7-R015",
        "Crash recovery and resume",
        "intergrax/runtime/long_running",
        "scheduler ledger",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "STATE-X",
        "PDC-P7-D7 replay",
        "sched_01",
        "FRZ-REC-01,FRZ-REC-03",
    ),
    Row(
        "PDC-D7-R016",
        "Full execution identity carried on enqueue (adoption gap)",
        "intergrax/queueing/contracts/task_queue.py",
        "TaskRequest",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D7",
        f"{Q}/d7 enqueue envelope carries canonical identity",
        "FRZ-EXE-06,FRZ-TRC-12",
    ),
    Row(
        "PDC-D7-R017",
        "Tenant scope of background work",
        "intergrax/runtime/background_execution/bootstrap.py",
        "BackgroundExecutionIdentity",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "HARNESS-W8/BG-01",
        "—",
        "bg_01 replay",
        "FRZ-TEN-08",
    ),
    Row(
        "PDC-D7-R018",
        "Governance of background work",
        "intergrax/runtime/background_execution/admission_wiring.py",
        "admission wiring",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "HARNESS-W8/BG-01",
        "—",
        "bg_01 replay",
        "FRZ-GOV-05",
    ),
    Row(
        "PDC-D7-R019",
        "Observability of background work",
        "intergrax/runtime/background_execution",
        "background evidence",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "HARNESS-W8/BG-01",
        "PDC-P7-D7 replay",
        "bg_01",
        "FRZ-OBS-05",
    ),
    Row(
        "PDC-D7-R020",
        "Operational control (list/cancel/purge)",
        "intergrax/queueing/contracts/task_queue.py",
        "TaskQueue",
        "PRESENT",
        "ALREADY_CERTIFIED_REUSABLE",
        "HARNESS-W8/BG-01",
        "—",
        "bg_01 replay",
        "FRZ-REL-04",
    ),
    Row(
        "PDC-D7-R021",
        "Horizontal scaling boundary documented (DIST-SCALE-X scoped)",
        "docs/project/architecture/BACKGROUND_TASKS.md",
        "scaling boundary",
        "PARTIAL",
        "DOCUMENTATION_RECONCILIATION",
        None,
        "PDC-P5-DOC",
        "scaling boundary section",
        "FRZ-REL-08",
    ),
    Row(
        "PDC-D7-R022",
        "TaskRegistry/WorkerRuntime wraps TaskExecutionRegistry (layered, not duplicate); ownership unrecorded",
        "intergrax/background_tasks/registry.py",
        "TaskRegistry -> TaskExecutionRegistry",
        "PRESENT",
        "DOCUMENTATION_RECONCILIATION",
        None,
        "PDC-P3-01 ownership record",
        "ownership matrix entry",
        "FRZ-OWN-01,FRZ-DOC-03",
    ),
    Row(
        "PDC-D7-R023",
        "BACKGROUND_TASKS plan statuses stale; doc not registered in runtime hub (PDC-P1-F01)",
        "docs/project/maintainers/plans/BACKGROUND_TASKS.md",
        "plan",
        "PRESENT",
        "DOCUMENTATION_RECONCILIATION",
        None,
        "PDC-P5-DOC",
        "plan + hub registration",
        "FRZ-DOC-01,FRZ-DOC-03",
    ),
    # ---------------- PDC-D8 Artifact Lifecycle ----------------
    Row(
        "PDC-D8-R001",
        "Artifact taxonomy (business / execution evidence / context / temporary tool output) with one owner per class; no global storage engine",
        "intergrax/contracts/artifact_ref.py",
        "none (no taxonomy)",
        "ABSENT",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P3 ADR-PDC-08 -> PDC-P5-D8",
        "taxonomy + ownership gate",
        "FRZ-OWN-01,FRZ-DOC-03",
        "ADR-PDC-08",
    ),
    Row(
        "PDC-D8-R002",
        "Mandatory content digest on durable artifact refs (optional outside Collaborative Work)",
        "intergrax/contracts/artifact_ref.py",
        "ArtifactRef",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D8 (ADR-PDC-08)",
        f"{Q}/d8 digest mismatch rejected",
        "FRZ-REC-10,FRZ-AUD-05",
        "ADR-PDC-08",
    ),
    Row(
        "PDC-D8-R003",
        "Versioning (business artifacts certified; run artifacts unversioned)",
        "intergrax/contracts/collaborative_work.py",
        "WorkArtifactVersion",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D8 (ADR-PDC-08)",
        f"{Q}/d8 run artifact version append-only",
        "FRZ-CMP-03",
        "ADR-PDC-08",
    ),
    Row(
        "PDC-D8-R004",
        "Producer identity and execution lineage",
        "intergrax/runtime/execution/run_artifact_bundle_composition.py",
        "RunArtifactBundle",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D8",
        f"{Q}/d8 artifact lineage reconstructable",
        "FRZ-TRC-01,FRZ-TRC-03",
    ),
    Row(
        "PDC-D8-R005",
        "Tenant/workspace scope of artifacts",
        "intergrax/runtime/nexus/artifacts/store_base.py",
        "ArtifactStore",
        "PARTIAL",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D8",
        f"{Q}/xd_10 cross-tenant read denied",
        "FRZ-TEN-04",
    ),
    Row(
        "PDC-D8-R006",
        "Governed artifact read/access authorization port (store explicitly has no security)",
        "intergrax/runtime/nexus/artifacts/store_base.py",
        "ArtifactStore",
        "ABSENT",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D8 (ADR-PDC-08)",
        f"{Q}/d8 unauthorized read denied",
        "FRZ-GOV-05,FRZ-SEC-08",
        "ADR-PDC-08",
    ),
    Row(
        "PDC-D8-R007",
        "Durable execution artifact store (only InMemoryArtifactStore)",
        "intergrax/runtime/nexus/artifacts/store_base.py",
        "ArtifactStore",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D8 (ADR-PDC-08)",
        f"{Q}/d8 artifacts survive restart",
        "FRZ-STA-07,FRZ-REC-01",
        "ADR-PDC-08",
    ),
    Row(
        "PDC-D8-R008",
        "Retention/TTL enforcement (ArtifactRetentionPolicy has zero consumers)",
        "intergrax/contracts/task_artifacts.py",
        "ArtifactRetentionPolicy",
        "PARTIAL",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D8 (ADR-PDC-08)",
        f"{Q}/d8 TTL expiry enforced",
        "FRZ-SEC-09",
        "ADR-PDC-08",
    ),
    Row(
        "PDC-D8-R009",
        "Deletion and legal hold; audit evidence non-deletable (Collaborative Work defers hard delete)",
        "intergrax/collaborative_work",
        "no hard-delete API",
        "ABSENT",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D8 (ADR-PDC-08)",
        f"{Q}/d8 evidence delete refused; legal hold respected",
        "FRZ-SEC-09,FRZ-AUD-05",
        "ADR-PDC-08",
    ),
    Row(
        "PDC-D8-R010",
        "Governed secure download (only provider presigned_url)",
        "intergrax/integrations/contracts/object_storage.py",
        "presigned_url",
        "PARTIAL",
        "BLOCKED_EXACT_REASON",
        None,
        "OD-PDC-SCOPE-04 (implement vs authorized exclusion)",
        f"{Q}/d8 governed download (if in scope)",
        "FRZ-SEC-08,FRZ-DEBT-05",
    ),
    Row(
        "PDC-D8-R011",
        "Partial outputs are owned by PartialResultContract (task metadata), not by artifacts",
        "intergrax/contracts/partial_result_contract.py",
        "PartialResultContract",
        "PRESENT",
        "N/A_WITH_EVIDENCE",
        None,
        "—",
        "partial_results_wiring.py owns partial results",
        "FRZ-OWN-01",
    ),
    Row(
        "PDC-D8-R012",
        "Large output / tool-result spill to artifact refs (none exists)",
        "intergrax/runtime/nexus/artifacts",
        "none",
        "ABSENT",
        "IMPLEMENTATION_COMPLETION",
        None,
        "PDC-P5-D8 (ADR-PDC-08)",
        f"{Q}/xd_09 spill round-trip",
        "FRZ-REL-10",
        "ADR-PDC-08",
    ),
    Row(
        "PDC-D8-R013",
        "Durable context references to artifacts",
        "intergrax/runtime/context_lifecycle/contracts.py",
        "UclArtifactOwnership",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D8",
        f"{Q}/xd_09 context ref resolves after compaction",
        "FRZ-REC-04",
    ),
    Row(
        "PDC-D8-R014",
        "Artifact recovery after restart",
        "intergrax/runtime/nexus/artifacts",
        "ArtifactStore",
        "ABSENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P5-D8",
        f"{Q}/d8 restart recovery",
        "FRZ-REC-01",
        "PDC-D8-R007",
    ),
    Row(
        "PDC-D8-R015",
        "Cross-provider portability of artifact content storage",
        "intergrax/collaborative_work/content_storage.py",
        "ArtifactContentStore",
        "PRESENT",
        "CURRENT_HEAD_REQUALIFICATION",
        "EBH-5",
        "PDC-P7-D8 replay",
        "collaborative work repository suites",
        "FRZ-RPL-04",
    ),
    Row(
        "PDC-D8-R016",
        "Auditability of artifact operations",
        "intergrax/runtime/runtime_inspection/adapters/artifact_read.py",
        "inspection adapter",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D8",
        f"{Q}/d8 artifact ops in evidence",
        "FRZ-OBS-06,FRZ-AUD-05",
    ),
    Row(
        "PDC-D8-R017",
        "Identical contracts in contracts/task_artifacts.py and applications/contracts/application_artifacts.py (copy has 0 production importers)",
        "intergrax/applications/contracts/application_artifacts.py",
        "ApplicationArtifactRef et al.",
        "PRESENT",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D8 (COMPAT-X R5 shim policy)",
        f"{Q}/d8 single definition gate",
        "FRZ-OWN-02,FRZ-OWN-05,FRZ-CMP-08",
    ),
    Row(
        "PDC-D8-R018",
        "Three types named ArtifactRef (pydantic ArtifactRef vs TraceArtifactRef aliases)",
        "intergrax/runtime/nexus/artifacts/models.py",
        "ArtifactRef alias",
        "PRESENT",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-D8",
        f"{Q}/d8 no ambiguous ArtifactRef alias",
        "FRZ-TYP-01,FRZ-OWN-02",
    ),
    Row(
        "PDC-D8-R019",
        "Compaction never deletes evidence artifacts (same invariant as PDC-D2-R018)",
        "intergrax/runtime/context_lifecycle",
        "UclArtifactOwnership",
        "PRESENT",
        "ADVERSARIAL_COVERAGE_REQUIRED",
        None,
        "PDC-P6 (XD-09)",
        f"{Q}/xd_09",
        "FRZ-AUD-05",
        "PDC-D2-R018",
    ),
    # ---------------- PDC-P2-09 replay findings (red gates at AUDIT_HEAD, pre-existing on development) ----------------
    Row(
        "PDC-D1-R022",
        "plug_03 external chunker entry-point flow fails at HEAD (test_plug_03_full_qualification_evidence_batch)",
        "intergrax/rag/bootstrap/entry_point_load.py",
        "entry-point chunker registration",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-REG (root cause: stale gate vs regression)",
        "tests/qualification/plug_03 green",
        "FRZ-PLG-02,FRZ-REG-02",
        "PDC-D1-R020",
    ),
    Row(
        "PDC-D3-R024",
        "EBH-5 replaceability gates red at HEAD: proof matrix/registry drift (48 vs 56 qualified concerns; RPL-F N/A vs PASS)",
        "tests/qualification/ebh_5",
        "RPL proof registry",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        "EBH-5",
        "PDC-P5-REG (root cause: stale matrix vs regression)",
        "tests/qualification/ebh_5 green",
        "FRZ-RPL-01,FRZ-REG-05",
    ),
    Row(
        "PDC-D4-R020",
        "GR-12 product host composition fails under STRICT agent contract authority (revision-bound authority required)",
        "intergrax/applications/_shared/roster_agent_contract_authority.py",
        "StrictAgentContractAuthority",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        None,
        "PDC-P5-REG (root cause: stale gate vs regression)",
        "tests/qualification/governance gr12 green",
        "FRZ-OWN-03,FRZ-REG-02",
    ),
    Row(
        "PDC-D6-R028",
        "GR-10 inference governance gates red at HEAD (pre-model enforcement AST, evidence-persistence wiring, checkpoint authority)",
        "intergrax/runtime/policy",
        "governed inference executor",
        "PARTIAL",
        "INTEGRATION_CORRECTION",
        "GOV-X2",
        "PDC-P5-REG (root cause: stale gate vs regression)",
        "tests/qualification/governance gr10 green",
        "FRZ-GOV-05,FRZ-REG-02",
    ),
)

_FRZ_ROW = re.compile(r"^\| (FRZ-[A-Z]+-\d+) \|", re.MULTILINE)


def checklist_ids() -> set[str]:
    return set(_FRZ_ROW.findall(CHECKLIST.read_text(encoding="utf-8")))


def _impact(frz: list[str], domain: str, prefixes: tuple[str, ...]) -> str:
    if any(f.startswith(prefixes) for f in frz):
        return "HIGH"
    return "MEDIUM" if domain == "PDC-D6" else "LOW"


def expand(row: Row) -> dict[str, str]:
    domain = row.rid[:6]
    frz = [f.strip() for f in row.frz.split(",") if f.strip()]
    sha, delta = CERTS[row.cert] if row.cert else ("—", "— (no accepted certification)")
    if row.rid in OD_RESOLUTIONS:
        cls, task = OD_RESOLUTIONS[row.rid]
        row = row._replace(cls=cls, task=task)
    red = _red_suites(row.cert)
    if row.cls == "ALREADY_CERTIFIED_REUSABLE" and red:
        # Reuse is forbidden while the certification's own qualification replay is red (PDC-P2-09).
        findings = ", ".join(REPLAY_FINDING_ROW[s] for s in red)
        row = row._replace(
            cls="CURRENT_HEAD_REQUALIFICATION",
            task=f"PDC-P7-{domain[4:]} replay after {findings}",
            deps=findings,
        )
        delta = f"{delta}; replay RED ({_suite_summary(*red)})"
    gap = row.cls in GAP_CLASSES
    if row.cls == "ALREADY_CERTIFIED_REUSABLE":
        disposition = "REUSE — requalify by replay at PDC-P7"
    elif row.cls == "N/A_WITH_EVIDENCE":
        disposition = "N/A — evidence in TEST_CASE; confirm at PDC-P7"
    elif row.cls == "BLOCKED_EXACT_REASON":
        disposition = f"BLOCKED — {row.task}"
    elif row.cls == "DEFERRED_OUTSIDE_FREEZE_SCOPE":
        disposition = f"DEFERRED — {row.task}; evidence-backed exclusion (FRZ-DEBT-06)"
    else:
        disposition = f"OPEN — closure via {row.task}"
    return {
        "REQUIREMENT_ID": row.rid,
        "DOMAIN_ID": domain,
        "REQUIREMENT": row.requirement,
        "SEMANTIC_OWNER": row.path,
        "IMPLEMENTATION_PATH": row.path,
        "PUBLIC_CONTRACT": row.contract,
        "CURRENT_IMPLEMENTATION_STATUS": row.status,
        "EXISTING_CERTIFICATION": row.cert or "NONE",
        "EXISTING_ACCEPTANCE_SHA": sha,
        "CURRENT_HEAD_DELTA": f"{delta} @ {AUDIT_HEAD[:12]}" if row.cert else delta,
        "MANDATORY_GAP": "YES" if gap else "NO",
        "CORRECTIVE_TASK": row.task,
        "TEST_CASE": row.test,
        "FRZ_IDS": ",".join(frz),
        "SECURITY_IMPACT": _impact(frz, domain, ("FRZ-SEC", "FRZ-GOV")),
        "TENANT_IMPACT": "HIGH" if any(f.startswith(("FRZ-TEN", "FRZ-SEC-01")) for f in frz) else "LOW",
        "DEPENDENCIES": row.deps,
        "CERTIFICATION_STATUS": "NOT_CERTIFIED",
        "AUDITOR_ID": "—",
        "AUDITED_SHA": "—",
        "EVIDENCE_PATH": DOMAIN_EVIDENCE[domain],
        "ARCH_FREEZE_BLOCKER": "YES" if gap else "NO",
        "FINAL_DISPOSITION": disposition,
        "GAP_CLASS": row.cls,
    }


def build() -> dict[str, object]:
    rows = [expand(r) for r in ROWS]
    totals: dict[str, dict[str, int]] = {}
    for r in rows:
        bucket = totals.setdefault(r["DOMAIN_ID"], {})
        bucket[r["GAP_CLASS"]] = bucket.get(r["GAP_CLASS"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_head": AUDIT_HEAD,
        "fields": list(FIELDS),
        "row_count": len(rows),
        "totals_by_domain": {k: dict(sorted(v.items())) for k, v in sorted(totals.items())},
        "rows": rows,
    }


def render_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def render_md(payload: dict[str, object]) -> str:
    rows = payload["rows"]
    assert isinstance(rows, list)
    out = [
        "# PDC-X — domain requirements matrix (PDC-P2)",
        "",
        "**Generated** by `scripts/qualification/pdc_x_requirements.py` from its `ROWS` source; do not edit by hand. "
        f"Machine-readable twin: [`PDC_DOMAIN_REQUIREMENTS_MATRIX.json`](./PDC_DOMAIN_REQUIREMENTS_MATRIX.json) (all {len(FIELDS)} fields). "
        f"Audit head `{AUDIT_HEAD}`. **No row is certified:** `CERTIFICATION_STATUS` = `NOT_CERTIFIED` until PDC-P7.",
        "",
        f"**Rows:** {payload['row_count']}.",
        "",
        "## Totals by domain and class",
        "",
        "| Domain | " + " | ".join(sorted(CLASSES)) + " |",
        "| --- | " + " | ".join("---:" for _ in CLASSES) + " |",
    ]
    totals = payload["totals_by_domain"]
    assert isinstance(totals, dict)
    for domain, counts in totals.items():
        out.append(f"| {domain} | " + " | ".join(str(counts.get(c, 0)) for c in sorted(CLASSES)) + " |")
    out += [
        "",
        "## Requirements",
        "",
        "| ID | Requirement | Status | Class | Certification @ delta | Corrective task | FRZ | Freeze blocker |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        cert = (
            "—"
            if r["EXISTING_CERTIFICATION"] == "NONE"
            else f"{r['EXISTING_CERTIFICATION']} ({r['CURRENT_HEAD_DELTA']})"
        )
        out.append(
            f"| {r['REQUIREMENT_ID']} | {r['REQUIREMENT']} | {r['CURRENT_IMPLEMENTATION_STATUS']} | {r['GAP_CLASS']} | "
            f"{cert} | {r['CORRECTIVE_TASK']} | {r['FRZ_IDS']} | {r['ARCH_FREEZE_BLOCKER']} |"
        )
    return "\n".join(out) + "\n"


def validate(payload: dict[str, object]) -> list[str]:
    errors: list[str] = []
    known = checklist_ids()
    families = {f.rsplit("-", 1)[0] for f in known}
    if len(known) != EXPECTED_FRZ or len(families) != EXPECTED_FAMILIES:
        errors.append(f"CHECKLIST_CARDINALITY:{len(known)}/{len(families)} != {EXPECTED_FRZ}/{EXPECTED_FAMILIES}")
    rows = payload["rows"]
    assert isinstance(rows, list)
    seen: set[str] = set()
    for r in rows:
        rid = r["REQUIREMENT_ID"]
        if rid in seen:
            errors.append(f"DUPLICATE_ID:{rid}")
        seen.add(rid)
        if set(r) != set(FIELDS):
            errors.append(f"FIELDS:{rid}")
        if r["GAP_CLASS"] not in CLASSES:
            errors.append(f"CLASS:{rid}:{r['GAP_CLASS']}")
        if "UNKNOWN" in r["GAP_CLASS"] or "UNKNOWN" in r["CURRENT_IMPLEMENTATION_STATUS"]:
            errors.append(f"TRACKED_UNKNOWN:{rid}")
        if r["CERTIFICATION_STATUS"] != "NOT_CERTIFIED" or "PASS" in r["FINAL_DISPOSITION"]:
            errors.append(f"PREMATURE_CERTIFICATION:{rid}")
        if r["GAP_CLASS"] not in NO_TASK_CLASSES and r["CORRECTIVE_TASK"] in {"", "—"}:
            errors.append(f"NO_CORRECTIVE_TASK:{rid}")
        if not r["FRZ_IDS"]:
            errors.append(f"NO_FRZ:{rid}")
        errors.extend(f"UNKNOWN_FRZ:{rid}:{f}" for f in r["FRZ_IDS"].split(",") if f not in known)
        if r["GAP_CLASS"] == "ALREADY_CERTIFIED_REUSABLE" and r["EXISTING_ACCEPTANCE_SHA"] == "—":
            errors.append(f"REUSE_WITHOUT_SHA:{rid}")
        if r["GAP_CLASS"] == "ALREADY_CERTIFIED_REUSABLE" and _red_suites(r["EXISTING_CERTIFICATION"]):
            errors.append(f"REUSE_WITH_RED_REPLAY:{rid}")
        cited = set(_OD_REF.findall(r["CORRECTIVE_TASK"]))
        if r["GAP_CLASS"] == "DEFERRED_OUTSIDE_FREEZE_SCOPE" and not cited & APPROVED_ODS:
            errors.append(f"DEFERRED_WITHOUT_APPROVED_OD:{rid}")
        if r["GAP_CLASS"] == "BLOCKED_EXACT_REASON" and cited & APPROVED_ODS:
            errors.append(f"BLOCKED_ON_DECIDED_OD:{rid}")
        if not rid.startswith(r["DOMAIN_ID"]):
            errors.append(f"DOMAIN_MISMATCH:{rid}")
    domains = {r["DOMAIN_ID"] for r in rows}
    if domains != {f"PDC-D{i}" for i in range(1, 9)}:
        errors.append(f"DOMAINS:{sorted(domains)}")
    return errors


_FRZ_META = re.compile(r"^\| (FRZ-[A-Z]+-\d+) \|[^|]*\|([^|]*)\|[^|]*\|([^|]*)\|", re.MULTILINE)


def checklist_rows() -> dict[str, tuple[str, str]]:
    text = CHECKLIST.read_text(encoding="utf-8")
    return {m.group(1): (m.group(2).strip(), m.group(3).strip()) for m in _FRZ_META.finditer(text)}


def _frz_key(frz: str) -> tuple[str, int]:
    family, number = frz.rsplit("-", 1)
    return family, int(number)


def render_frz(payload: dict[str, object]) -> str:
    rows = payload["rows"]
    assert isinstance(rows, list)
    by_frz: dict[str, list[str]] = {}
    for r in rows:
        for frz in r["FRZ_IDS"].split(","):
            by_frz.setdefault(frz, []).append(r["REQUIREMENT_ID"])
    meta = checklist_rows()
    out = [
        "# PDC-X — FRZ coverage matrix (PDC-P2-10)",
        "",
        "**Generated** by `scripts/qualification/pdc_x_requirements.py`. Existing FRZ identifiers only "
        "(OD-PDC-FRZ-01 Option A, pending signature). Checklist cardinality verified: "
        f"**{EXPECTED_FRZ}** criteria in **{EXPECTED_FAMILIES}** families. No status is changed here; promotion happens "
        "only at PDC-P9 through the checklist update protocol.",
        "",
        "## Coverage by family",
        "",
        "| Family | Criteria | Referenced by PDC | Not referenced |",
        "| --- | ---: | ---: | --- |",
    ]
    for family in sorted({_frz_key(f)[0] for f in meta}):
        ids = sorted((f for f in meta if _frz_key(f)[0] == family), key=_frz_key)
        missing = [f for f in ids if f not in by_frz]
        out.append(f"| {family} | {len(ids)} | {len(ids) - len(missing)} | {', '.join(missing) if missing else '—'} |")
    out += [
        "",
        "Criteria not referenced stay with their certifying stages or with ARCH-FREEZE itself; PDC-P3 records any "
        "exception.",
        "",
        "## Criterion → PDC requirements",
        "",
        "| FRZ | Criterion | Checklist status | PDC requirements |",
        "| --- | --- | --- | --- |",
    ]
    for frz in sorted(by_frz, key=_frz_key):
        criterion, status = meta.get(frz, ("?", "?"))
        out.append(f"| {frz} | {criterion} | {status} | {', '.join(by_frz[frz])} |")
    return "\n".join(out) + "\n"


_LANE = re.compile(
    r"(PDC-P5-D\d|PDC-P5-DOC|PDC-P5-REG|PDC-P6|PDC-P3-01|PDC-P3 ADR-PDC-08|OD-PDC-SCOPE-\d+|MCP-SERVER-X)"
)


def render_gaps(payload: dict[str, object]) -> str:
    rows = payload["rows"]
    assert isinstance(rows, list)
    lanes: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        if r["GAP_CLASS"] not in GAP_CLASSES:
            continue
        match = _LANE.search(r["CORRECTIVE_TASK"])
        lanes.setdefault(match.group(1) if match else r["CORRECTIVE_TASK"], []).append(r)
    out = [
        "# PDC-X — gap register (PDC-P2-11)",
        "",
        "**Generated** by `scripts/qualification/pdc_x_requirements.py`. One entry per requirement that needs work "
        "beyond replay. Rows classed `CURRENT_HEAD_REQUALIFICATION`, `ALREADY_CERTIFIED_REUSABLE` or "
        "`N/A_WITH_EVIDENCE` are closed by PDC-P7 replay and appear only in the requirements matrix. "
        "`BLOCKED_EXACT_REASON` rows wait on the named Operator Decision.",
        "",
        "| Lane | Count |",
        "| --- | ---: |",
    ]
    out += [f"| {lane} | {len(lanes[lane])} |" for lane in sorted(lanes)]
    for lane in sorted(lanes):
        out += [
            "",
            f"## {lane}",
            "",
            "| ID | Gap | Class | Test / evidence | Depends on |",
            "| --- | --- | --- | --- | --- |",
        ]
        out += [
            f"| {r['REQUIREMENT_ID']} | {r['REQUIREMENT']} | {r['GAP_CLASS']} | {r['TEST_CASE']} | {r['DEPENDENCIES']} |"
            for r in lanes[lane]
        ]
    return "\n".join(out) + "\n"


def _outputs(payload: dict[str, object]) -> tuple[tuple[Path, str], ...]:
    return (
        (OUT_JSON, render_json(payload)),
        (OUT_MD, render_md(payload)),
        (DOSSIER / "PDC_FRZ_COVERAGE_MATRIX.md", render_frz(payload)),
        (DOSSIER / "PDC_GAP_REGISTER.md", render_gaps(payload)),
    )


def check(payload: dict[str, object]) -> list[str]:
    errors = validate(payload)
    for path, text in _outputs(payload):
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            errors.append(f"STALE_OUTPUT:{path.name} (re-run with --write)")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = build()
    if args.write:
        errors = validate(payload)
        if errors:
            for err in errors:
                print(f"[ERROR] {err}", file=sys.stderr)
            return 1
        for path, text in _outputs(payload):
            path.write_text(text, encoding="utf-8", newline="\n")
        print(f"[OK] wrote {payload['row_count']} requirements")
        return 0
    errors = check(payload)
    if errors:
        for err in errors:
            print(f"[ERROR] {err}", file=sys.stderr)
        return 1
    print(f"[OK] pdc_x_requirements: {payload['row_count']} rows, schema valid, FRZ {EXPECTED_FRZ}/{EXPECTED_FAMILIES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
