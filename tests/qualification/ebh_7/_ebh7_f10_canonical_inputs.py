# © Artur Czarnecki. All rights reserved.

"""B6-F10 canonical inputs — closed normative sets parsed from authoritative sources @ HEAD."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.ebh_7._ebh7_f10_types import ChangeClass

_REPO: Final[Path] = Path(__file__).resolve().parents[3]
_QUAL = "docs/project/maintainers/qualification"
FRZ_CHECKLIST_REL: Final[str] = f"{_QUAL}/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md"
ECAP_PROTOCOL_REL: Final[str] = f"{_QUAL}/ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md"
ROADMAP_REL: Final[str] = "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"
AGENTS_REL: Final[str] = "AGENTS.md"
EBH7_FAMILY_MATRIX_REL: Final[str] = f"{_QUAL}/EBH_7_ENTERPRISE_FAMILY_COVERAGE_MATRIX.md"
DUP_X_CLASS_MATRIX_REL: Final[str] = f"{_QUAL}/DUP_X_CLASS_COVERAGE_MATRIX.md"
MANIFEST_REL: Final[str] = f"{_QUAL}/ebh_7_p03_independent_replay/MANIFEST.json"
ECAP_REPORT_REL: Final[str] = f"{_QUAL}/EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md"
F10_MATRIX_REL: Final[str] = f"{_QUAL}/EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md"
_IDENTITY_SCRIPT_REL: Final[str] = "scripts/qualification/verify_ebh7_p03_immutable_evidence.py"

DUP_X_INDEPENDENT_ACCEPTANCE_SHA: Final[str] = "a7d83925b375b5f08eb0f175ced2c17fa026bba9"
ROADMAP_REPLAY_X_ACCEPTANCE_SHA: Final[str] = "5a3a05347aaea93a8e4086fc90bc08e6d5eeaf92"

_TIER_LINE_RE = re.compile(r"Tier-(\d) (\S+/)")
_FRZ_ID_RE = re.compile(r"\bFRZ-([A-Z]+)-\d+\b")
_ECAP_FAMILY_HEADER_RE = re.compile(r"^#### FRZ-([A-Z]+) — ")
_ECAP_CLOSING_RE = re.compile(r"^\| \*\*Primary closing stage\(s\)\*\* \| (.+?) \|\s*$")
_ROADMAP_STAGE_ROW_RE = re.compile(r"^\| \*\*([A-Z0-9][A-Za-z0-9\-]+)\*\* \|")
_ROADMAP_STATUS_MARK_RE = re.compile(r"\[([ x])\]")
_ROADMAP_VBN_RE = re.compile(r"^(\d{1,2})\. (.+)$")
_EBH7_FAMILY_ROW_RE = re.compile(r"^\| ([A-Z]{3,4}) \|")
_DUP_CLASS_ROW_RE = re.compile(r"^\| \*\*(DUP-\d{2})\*\* \|(?:[^|]*\|){2}\s*(.+?)\s*\|\s*$")
_CLOSING_STAGE_TOKEN_RE = re.compile(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*\*?")


@dataclass(frozen=True, slots=True)
class TierRecord:
    tier_id: str
    root: str
    forbidden_top_level_imports: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TierImportException:
    path: str
    module_prefix: str
    evidence_doc: str


@dataclass(frozen=True, slots=True)
class FamilyRecord:
    family_id: str
    owner: str
    gate_nodes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DupRecord:
    record_id: str
    dup_classes: tuple[str, ...]
    status: str


@dataclass(frozen=True, slots=True)
class ResponsibilityFamily:
    family_id: str
    owner_doc: str
    implementation_path: str
    composition_root: str


@dataclass(frozen=True, slots=True)
class OwnerDeltaBinding:
    """Current-HEAD owner resolution where the accepted DUP-X SSOT pointer no longer resolves."""

    owner_doc: str = ""
    implementation_path: str = ""
    discovery_check: str = ""


@dataclass(frozen=True, slots=True)
class VendorConcern:
    concern_id: str
    domain: str
    semantic_owner: str
    composition_owner: str
    classification: str
    structural_evidence: str
    outstanding_debt: str


@dataclass(frozen=True, slots=True)
class DomainNaDisposition:
    domain: str
    kind: str
    discovery_paths: tuple[str, ...]
    rationale: str


@dataclass(frozen=True, slots=True)
class C4Rule:
    prefix: str
    classes: tuple[ChangeClass, ...]


@dataclass(frozen=True, slots=True)
class F10Inputs:
    # Execution identity
    head_sha: str
    test_execution_sha: str
    identity_violations: tuple[str, ...]
    profile_argv: tuple[str, ...]
    in_process_failures: dict[str, str]
    # C1
    frz_families: tuple[str, ...]
    ecap_closing_stages: dict[str, tuple[str, ...]]
    roadmap_stage_order: dict[str, int]
    roadmap_closed_stages: frozenset[str]
    ebh6_families: tuple[FamilyRecord, ...]
    ebh7_family_matrix_ids: tuple[str, ...]
    rollup_families: dict[str, str]
    tiers: tuple[TierRecord, ...]
    tier_import_violations: tuple[tuple[str, str], ...]
    tier_import_exceptions: tuple[TierImportException, ...]
    # C2
    dup_classes: tuple[str, ...]
    dup_class_doc_results: dict[str, str]
    dup_records: tuple[DupRecord, ...]
    rf_families: tuple[ResponsibilityFamily, ...]
    rf_owner_delta: dict[str, OwnerDeltaBinding]
    c2_surfaces: tuple[tuple[str, tuple[str, ...]], ...]
    dup_counters: dict[str, int]
    # C3
    roadmap_vbn_domains: tuple[tuple[int, str], ...]
    vbn_domains: tuple[str, ...]
    vendor_concerns: tuple[VendorConcern, ...]
    domain_na: dict[str, DomainNaDisposition]
    # C4
    accepted_baseline_sha: str
    authorized_baseline_sha: str
    ancestry: dict[tuple[str, str], bool]
    changed_paths: tuple[str, ...]
    c4_rules: tuple[C4Rule, ...]
    c4_class_regressions: dict[ChangeClass, tuple[str, ...]]
    # C5
    report_text: str
    matrix_disk_text: str
    existing_paths: frozenset[str] = field(default_factory=frozenset)


# --- C2 normative surfaces (ECAP §2.C C2 / roadmap §3.1 duplication coverage) ---------------------------------

C2_NORMATIVE_SURFACES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("execution-dispatch", ("RF-EXE-01",)),
    ("governance", ("RF-GOV-01",)),
    ("state-persistence-recovery", ("RF-STA-01", "RF-RCV-01")),
    ("provider-backend-selection", ("RF-CAP-01", "RF-PRV-01")),
    ("configuration-profile", ("RF-CAP-01",)),
    ("registries", ("RF-REG-01", "RF-PLG-01")),
    ("tenant-context", ("RF-TEN-01",)),
    ("observability", ("RF-OBS-01",)),
    ("trace-evidence", ("RF-PRV-02",)),
    ("scheduler-background", ("RF-BGW-01",)),
    ("compatibility", ("RF-LEG-01",)),
    ("composition", ("RF-CMP-01", "RF-T3-01")),
)

RF_OWNER_DELTA: Final[dict[str, OwnerDeltaBinding]] = {
    "RF-EXE-01": OwnerDeltaBinding(owner_doc="docs/project/architecture/UNIFIED_EXECUTION_ARCHITECTURE.md"),
    "RF-BGW-01": OwnerDeltaBinding(owner_doc="docs/project/architecture/BACKGROUND_TASKS.md"),
    "RF-GOV-01": OwnerDeltaBinding(owner_doc="docs/project/architecture/GOVERNED_EXECUTION.md"),
    "RF-RCV-01": OwnerDeltaBinding(owner_doc="docs/project/architecture/RECOVERY_AND_COMPENSATION.md"),
    "RF-STA-01": OwnerDeltaBinding(
        owner_doc="docs/project/architecture/MEMORY_ARCHITECTURE.md",
        implementation_path="intergrax/runtime/persistence/",
    ),
    "RF-TEN-01": OwnerDeltaBinding(
        owner_doc="docs/project/architecture/UNIFIED_EXECUTION_ARCHITECTURE.md",
        implementation_path="intergrax/runtime/security/tenant_scope.py",
    ),
    "RF-CMP-01": OwnerDeltaBinding(discovery_check="dup_x.composition_root_union"),
    "RF-T3-01": OwnerDeltaBinding(discovery_check="dup_x.tier3_host_factory_census"),
}

ROLLUP_FAMILIES: Final[dict[str, str]] = {"VBN": "C3"}

TIER_IMPORT_EXCEPTIONS: Final[tuple[TierImportException, ...]] = (
    TierImportException(
        path="intergrax/cli/external_work.py",
        module_prefix="applications.governed_contractor_application.host.collaborative_work_local_fixture",
        evidence_doc="docs/project/maintainers/audits/AGENT_PLATFORM_AC1_AC6_FINAL_ARCHITECTURE_AUDIT.md",
    ),
)

C4_RULES: Final[tuple[C4Rule, ...]] = (
    C4Rule("docs/", (ChangeClass.DOCUMENTATION_ONLY,)),
    C4Rule("tests/", (ChangeClass.QUALIFICATION_INFRASTRUCTURE,)),
    C4Rule("scripts/qualification/", (ChangeClass.QUALIFICATION_INFRASTRUCTURE,)),
    C4Rule("scripts/ci/", (ChangeClass.QUALIFICATION_INFRASTRUCTURE,)),
    C4Rule(".github/", (ChangeClass.QUALIFICATION_INFRASTRUCTURE,)),
    C4Rule("pyproject.toml", (ChangeClass.COMPOSITION_CONFIGURATION,)),
    C4Rule("uv.lock", (ChangeClass.COMPOSITION_CONFIGURATION,)),
    C4Rule("intergrax/applications/", (ChangeClass.COMPOSITION_CONFIGURATION,)),
    C4Rule("intergrax/contracts/", (ChangeClass.CONTRACT,)),
    C4Rule("intergrax/integrations/registry/", (ChangeClass.OWNER_AUTHORITY,)),
    C4Rule("intergrax/integrations/providers/", (ChangeClass.VENDOR_BACKEND,)),
    C4Rule("intergrax/memory/stores/", (ChangeClass.STATE_RECOVERY,)),
    C4Rule("intergrax/runtime/security/", (ChangeClass.SECURITY_GOVERNANCE_TENANT,)),
    C4Rule("intergrax/runtime/governance/", (ChangeClass.SECURITY_GOVERNANCE_TENANT,)),
    C4Rule(
        "intergrax/runtime/nexus/tools/invoker.py",
        (ChangeClass.RUNTIME_BEHAVIOR, ChangeClass.SECURITY_GOVERNANCE_TENANT),
    ),
    C4Rule("intergrax/runtime/", (ChangeClass.RUNTIME_BEHAVIOR,)),
)

_TOOLS = "tests/unit/runtime/nexus/tools"
C4_CLASS_REGRESSIONS: Final[dict[ChangeClass, tuple[str, ...]]] = {
    ChangeClass.DOCUMENTATION_ONLY: (
        "tests/qualification/ebh_7/test_ebh_7_b6_f10_f11_corrective_gates.py::test_b6_f10_matrix_markdown_matches_generated",
    ),
    ChangeClass.QUALIFICATION_INFRASTRUCTURE: (
        "tests/qualification/ebh_7/test_ebh_7_p03_b5_immutable_gates.py",
        "tests/qualification/ebh_7/test_ebh_7_b6_f10_certification_adversarial.py",
    ),
    ChangeClass.CONTRACT: ("tests/unit/architecture/test_ebh_2a_public_contract_boundary_gate.py",),
    ChangeClass.OWNER_AUTHORITY: ("tests/unit/runtime/architecture/test_ee_final_arch_owner_uniqueness.py",),
    ChangeClass.COMPOSITION_CONFIGURATION: (
        "tests/qualification/config_x/test_config_x_final_reconciliation_gates.py",
        "tests/qualification/ebh_5/test_ebh_5_l2_integrity_gates.py",
    ),
    ChangeClass.RUNTIME_BEHAVIOR: (
        f"{_TOOLS}/test_runtime_tool_invoker_policy.py",
        f"{_TOOLS}/test_runtime_tool_invoker_dependency_admission.py",
        f"{_TOOLS}/test_tool_runtime_scope_enforcement.py",
    ),
    ChangeClass.SECURITY_GOVERNANCE_TENANT: (
        f"{_TOOLS}/test_runtime_tool_invoker_cross_tenant.py",
        "tests/unit/runtime/security/test_tenant_scope.py",
        "tests/qualification/ebh_7/tenant_platform_e2e/test_ebh7_integrated_tenant_boundary_proofs.py",
    ),
    ChangeClass.STATE_RECOVERY: (
        "tests/qualification/ebh_7/test_ebh_7_terminal_rpl_gates.py::test_ebh7_fixed_substrate_rpl_positive",
        "tests/qualification/ebh_7/test_ebh_7_terminal_rpl_gates.py::test_ebh7_fixed_substrate_rpl_negative",
    ),
    ChangeClass.VENDOR_BACKEND: (
        "tests/qualification/ebh_7/test_ebh_7_int2e_bundle_regression.py",
        "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py",
    ),
    ChangeClass.UNKNOWN: (),
}


# --- Parsers (pure; injectable text) ----------------------------------------------------------------------------


def _read(rel: str) -> str:
    path = _REPO / rel
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=_REPO, text=True).strip()


def _is_ancestor(ancestor: str, head: str) -> bool:
    proc = subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, head], cwd=_REPO, capture_output=True)
    return proc.returncode == 0


def parse_frz_families(checklist_text: str) -> tuple[str, ...]:
    return tuple(sorted({m.group(1) for m in _FRZ_ID_RE.finditer(checklist_text)}))


def parse_ecap_closing_stages(ecap_text: str) -> dict[str, tuple[str, ...]]:
    stages: dict[str, tuple[str, ...]] = {}
    current: str | None = None
    for line in ecap_text.splitlines():
        header = _ECAP_FAMILY_HEADER_RE.match(line)
        if header:
            current = header.group(1)
            continue
        closing = _ECAP_CLOSING_RE.match(line)
        if closing and current and current not in stages:
            stage_cell = closing.group(1).split(";", 1)[0]
            stages[current] = tuple(_CLOSING_STAGE_TOKEN_RE.findall(stage_cell))
    return stages


def parse_roadmap_stage_rows(roadmap_text: str) -> tuple[dict[str, int], frozenset[str]]:
    """Stage status table rows: ``| **STAGE** | name | scope | [ ]/[x] status |`` (first occurrence wins)."""
    order: dict[str, int] = {}
    closed: set[str] = set()
    for index, line in enumerate(roadmap_text.splitlines()):
        match = _ROADMAP_STAGE_ROW_RE.match(line)
        if not match:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        status = _ROADMAP_STATUS_MARK_RE.search(cells[-1]) if len(cells) == 4 else None
        stage = match.group(1)
        if status is None or stage in order:
            continue
        order[stage] = index
        if status.group(1) == "x":
            closed.add(stage)
    return order, frozenset(closed)


def parse_roadmap_vbn_domains(roadmap_text: str) -> tuple[tuple[int, str], ...]:
    lines = roadmap_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("#### Mandatory domain taxonomy")), None)
    if start is None:
        return ()
    domains: list[tuple[int, str]] = []
    for line in lines[start + 1 :]:
        if line.startswith("#### ") or line.startswith("Per-concern record fields"):
            break
        match = _ROADMAP_VBN_RE.match(line.strip())
        if match:
            domains.append((int(match.group(1)), match.group(2).strip()))
    return tuple(domains)


def parse_agents_tiers(agents_text: str) -> tuple[TierRecord, ...]:
    tiers: list[TierRecord] = []
    for line in agents_text.splitlines():
        found = _TIER_LINE_RE.findall(line)
        if len(found) >= 4:
            for tier_no, root in found:
                if tier_no in ("0", "1"):
                    forbidden = ("agents", "applications")
                elif tier_no == "2":
                    forbidden = ("applications",)
                else:
                    forbidden = ()
                tiers.append(TierRecord(f"Tier-{tier_no}", root, forbidden))
            break
    return tuple(tiers)


def parse_ebh7_family_matrix_ids(matrix_text: str) -> tuple[str, ...]:
    return tuple(m.group(1) for ln in matrix_text.splitlines() if (m := _EBH7_FAMILY_ROW_RE.match(ln)))


def parse_dup_class_doc_results(doc_text: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for ln in doc_text.splitlines() if (m := _DUP_CLASS_ROW_RE.match(ln))}


def scan_tier_import_violations(tiers: tuple[TierRecord, ...]) -> tuple[tuple[str, str], ...]:
    violations: list[tuple[str, str]] = []
    for tier in tiers:
        if not tier.forbidden_top_level_imports or tier.tier_id == "Tier-1":
            continue
        for path in sorted((_REPO / tier.root).rglob("*.py")):
            rel = path.relative_to(_REPO).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module]
                else:
                    continue
                for module in modules:
                    if module.split(".", 1)[0] in tier.forbidden_top_level_imports:
                        violations.append((rel, module))
    return tuple(sorted(set(violations)))


def profile_argv_from_command(command: str) -> tuple[str, ...]:
    parts = shlex.split(command, posix=False)
    if parts[:3] == ["uv", "run", "pytest"]:
        return tuple(parts[3:])
    if parts[:1] == ["pytest"]:
        return tuple(parts[1:])
    raise AssertionError(f"Unrecognized pytest profile command: {command!r}")


def _identity_allowlist() -> Callable[[str], bool]:
    spec = importlib.util.spec_from_file_location("_ebh7_p03_identity", _REPO / _IDENTITY_SCRIPT_REL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._path_allowed


def executable_identity_violations(test_execution_sha: str, head_sha: str) -> tuple[str, ...]:
    if test_execution_sha == head_sha:
        return ()
    if not _is_ancestor(test_execution_sha, head_sha):
        return (f"TEST_EXECUTION_SHA {test_execution_sha} is not an ancestor of executed HEAD {head_sha}",)
    allowed = _identity_allowlist()
    changed = _git("diff", "--name-only", test_execution_sha, head_sha).splitlines()
    return tuple(p for p in (c.strip() for c in changed) if p and not allowed(p))


# --- In-process validators (real execution @ HEAD) --------------------------------------------------------------


def _run_checks(checks: dict[str, Callable[[], None]]) -> dict[str, str]:
    failures: dict[str, str] = {}
    for name, check in checks.items():
        try:
            check()
        except AssertionError as exc:
            failures[name] = str(exc)[:300] or "assertion failed"
    return failures


def _in_process_checks() -> dict[str, Callable[[], None]]:
    from tests.qualification.dup_x._closed_world_inventory import (
        COMPOSITION_ROOT_HINTS,
        assert_dup_x_closed_world_integrity,
        discover_application_host_factory_paths,
        discover_integration_env_slug_reads_outside_factory,
        discover_normative_composition_root_union,
        discover_production_tools_base_imports,
    )
    from tests.qualification.ebh_5._inventory_closed_world_validation import inventory_validation_errors
    from tests.qualification.ebh_5._replaceability_matrix import undocumented_qualified_rows
    from tests.qualification.ebh_6._config_x_reconciliation import assert_ebh6_config_x_fully_reconciled
    from tests.qualification.ebh_6._findings_register import count_prohibited_unresolved, count_unclassified

    def _expect_empty(label: str, value: object) -> None:
        assert not value, f"{label}: {sorted(value)[:5] if isinstance(value, (set, frozenset)) else value}"

    def _composition_union() -> None:
        union = discover_normative_composition_root_union()
        assert COMPOSITION_ROOT_HINTS[0] in union, "composition root union lacks canonical hint"

    def _tier3_census() -> None:
        assert len(discover_application_host_factory_paths()) >= 9, "Tier-3 host factory census < 9"

    def _ebh6_unclassified() -> None:
        assert count_unclassified() == 0, f"EBH-6 unclassified={count_unclassified()}"

    def _ebh6_prohibited() -> None:
        assert count_prohibited_unresolved() == 0, f"EBH-6 PROHIBITED unresolved={count_prohibited_unresolved()}"

    return {
        "dup_x.closed_world_integrity": assert_dup_x_closed_world_integrity,
        "dup_x.composition_root_union": _composition_union,
        "dup_x.tier3_host_factory_census": _tier3_census,
        "dup_x.env_slug_reads_outside_factory": lambda: _expect_empty(
            "env slug reads outside factory", discover_integration_env_slug_reads_outside_factory()
        ),
        "dup_x.production_tools_base_imports": lambda: _expect_empty(
            "production tools_base imports", discover_production_tools_base_imports()
        ),
        "ebh5.inventory_closed_world": lambda: _expect_empty("inventory errors", inventory_validation_errors()),
        "ebh5.qualified_proof_integrity": lambda: _expect_empty("invalid QUALIFIED", undocumented_qualified_rows()),
        "ebh6.unclassified_zero": _ebh6_unclassified,
        "ebh6.prohibited_unresolved_zero": _ebh6_prohibited,
        "config_x.head_reconciled_54": assert_ebh6_config_x_fully_reconciled,
    }


def _dup_counters() -> dict[str, int]:
    from tests.qualification.dup_x import _findings_register as reg

    return {
        "unresolved_duplicate_blockers": reg.count_unresolved_duplicate_blockers(),
        "unresolved_shadow_authority": reg.count_unresolved_shadow_authority(),
        "open_gaps": reg.count_open_gaps(),
        "open_candidates": reg.count_open_candidates(),
        "unclassified_candidates": reg.count_unclassified_candidates(),
        "open_bypass": reg.count_open_bypass_unresolved(),
        "retained_parallel_without_dossier": len(reg.retained_parallel_without_dossier()),
        "unknown_reachability": len(reg.findings_with_unknown_reachability()),
    }


def _dup_records() -> tuple[DupRecord, ...]:
    from tests.qualification.dup_x._findings_register import DUP_X_CANDIDATE_RECORDS, DUP_X_FINDING_RECORDS

    rows = [DupRecord(r.finding_id, r.dup_classes, r.status.value) for r in DUP_X_FINDING_RECORDS]
    rows.extend(DupRecord(r.candidate_id, r.dup_classes, r.status.value) for r in DUP_X_CANDIDATE_RECORDS)
    return tuple(rows)


def _vendor_concerns() -> tuple[VendorConcern, ...]:
    from tests.qualification.ebh_5._vendor_backend_inventory import VENDOR_BACKEND_DEPENDENCY_INVENTORY

    return tuple(
        VendorConcern(
            concern_id=r.concern_id,
            domain=r.domain.value,
            semantic_owner=r.semantic_owner,
            composition_owner=r.composition_selection_owner,
            classification=r.extensibility_classification.value,
            structural_evidence=r.structural_replaceability_evidence_id,
            outstanding_debt=r.outstanding_debt,
        )
        for r in VENDOR_BACKEND_DEPENDENCY_INVENTORY
    )


def _domain_na() -> dict[str, DomainNaDisposition]:
    from tests.qualification.ebh_5._roadmap_domain_na_evidence import roadmap_domain_na_dispositions

    return {
        d.domain.value: DomainNaDisposition(d.domain.value, d.kind, d.discovery_paths, d.rationale)
        for d in roadmap_domain_na_dispositions()
    }


def _ebh6_families() -> tuple[FamilyRecord, ...]:
    from tests.qualification.ebh_6._family_catalog import EBH6_FAMILY_CATALOG

    return tuple(FamilyRecord(r.family_id, r.primary_historical_owner, r.gate_test_ids) for r in EBH6_FAMILY_CATALOG)


def _rf_families() -> tuple[ResponsibilityFamily, ...]:
    from tests.qualification.dup_x._closed_world_inventory import DUP_X_RESPONSIBILITY_FAMILIES

    return tuple(
        ResponsibilityFamily(r.family_id, r.canonical_owner_doc, r.implementation_owner_path, r.composition_root)
        for r in DUP_X_RESPONSIBILITY_FAMILIES
    )


def _path_resolves(rel: str) -> bool:
    if "*" in rel:
        return any(_REPO.glob(rel))
    return (_REPO / rel.split("#", 1)[0]).exists()


def _referenced_paths(inputs_paths: list[str]) -> frozenset[str]:
    return frozenset(p for p in inputs_paths if p and _path_resolves(p))


@lru_cache(maxsize=1)
def cached_canonical_f10_inputs() -> F10Inputs:
    """Single in-process build per pytest session; callers derive variants via ``dataclasses.replace``."""
    return build_canonical_f10_inputs()


def build_canonical_f10_inputs() -> F10Inputs:
    from tests.qualification.dup_x._findings_register import DUP_CLASSES
    from tests.qualification.ebh_5._vendor_backend_types import RoadmapVendorDomain
    from tests.qualification.ebh_7._constants import EBH7_AUTHORIZED_BASELINE_SHA, EBH7_PARENT_REPLAY_ACCEPTANCE_SHA

    manifest = json.loads(_read(MANIFEST_REL))
    test_sha = manifest["TEST_EXECUTION_SHA"]
    head = _git("rev-parse", "HEAD")
    roadmap = _read(ROADMAP_REL)
    order, closed = parse_roadmap_stage_rows(roadmap)
    tiers = parse_agents_tiers(_read(AGENTS_REL))
    rf_families = _rf_families()
    ebh6 = _ebh6_families()
    accepted = EBH7_PARENT_REPLAY_ACCEPTANCE_SHA
    ancestry = {
        (DUP_X_INDEPENDENT_ACCEPTANCE_SHA, test_sha): _is_ancestor(DUP_X_INDEPENDENT_ACCEPTANCE_SHA, test_sha),
        (accepted, EBH7_AUTHORIZED_BASELINE_SHA): _is_ancestor(accepted, EBH7_AUTHORIZED_BASELINE_SHA),
        (EBH7_AUTHORIZED_BASELINE_SHA, test_sha): _is_ancestor(EBH7_AUTHORIZED_BASELINE_SHA, test_sha),
        (accepted, test_sha): _is_ancestor(accepted, test_sha),
    }
    candidate_paths: list[str] = [t.root for t in tiers]
    candidate_paths += [e.evidence_doc for e in TIER_IMPORT_EXCEPTIONS]
    for rf in rf_families:
        candidate_paths += [rf.owner_doc, rf.implementation_path, rf.composition_root]
    for delta in RF_OWNER_DELTA.values():
        candidate_paths += [delta.owner_doc, delta.implementation_path]
    na = _domain_na()
    for disposition in na.values():
        candidate_paths += list(disposition.discovery_paths)
    nodes = [n for f in ebh6 for n in f.gate_nodes]
    nodes += [n for regs in C4_CLASS_REGRESSIONS.values() for n in regs]
    candidate_paths += [n.split("::", 1)[0] for n in nodes]
    return F10Inputs(
        head_sha=head,
        test_execution_sha=test_sha,
        identity_violations=executable_identity_violations(test_sha, head),
        profile_argv=profile_argv_from_command(manifest["pytest_profile"]["command"]),
        in_process_failures=_run_checks(_in_process_checks()),
        frz_families=parse_frz_families(_read(FRZ_CHECKLIST_REL)),
        ecap_closing_stages=parse_ecap_closing_stages(_read(ECAP_PROTOCOL_REL)),
        roadmap_stage_order=order,
        roadmap_closed_stages=closed,
        ebh6_families=ebh6,
        ebh7_family_matrix_ids=parse_ebh7_family_matrix_ids(_read(EBH7_FAMILY_MATRIX_REL)),
        rollup_families=dict(ROLLUP_FAMILIES),
        tiers=tiers,
        tier_import_violations=scan_tier_import_violations(tiers),
        tier_import_exceptions=TIER_IMPORT_EXCEPTIONS,
        dup_classes=DUP_CLASSES,
        dup_class_doc_results=parse_dup_class_doc_results(_read(DUP_X_CLASS_MATRIX_REL)),
        dup_records=_dup_records(),
        rf_families=rf_families,
        rf_owner_delta=dict(RF_OWNER_DELTA),
        c2_surfaces=C2_NORMATIVE_SURFACES,
        dup_counters=_dup_counters(),
        roadmap_vbn_domains=parse_roadmap_vbn_domains(roadmap),
        vbn_domains=tuple(d.value for d in RoadmapVendorDomain),
        vendor_concerns=_vendor_concerns(),
        domain_na=na,
        accepted_baseline_sha=accepted,
        authorized_baseline_sha=EBH7_AUTHORIZED_BASELINE_SHA,
        ancestry=ancestry,
        changed_paths=tuple(p for p in _git("diff", "--name-only", accepted, test_sha).splitlines() if p.strip()),
        c4_rules=C4_RULES,
        c4_class_regressions=dict(C4_CLASS_REGRESSIONS),
        report_text=_read(ECAP_REPORT_REL),
        matrix_disk_text=_read(F10_MATRIX_REL),
        existing_paths=_referenced_paths(candidate_paths),
    )
