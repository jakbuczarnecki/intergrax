#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E501
"""ARCH-FREEZE B1: closed-world production inventory @ exact HEAD (T-B1-ENGINE + strict qualify)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_BASELINE_SHA = "a0624e28ff77b061e83f3e10aab4c2a9083ee1b4"
CW09_DISPOSITION_ID = "CW09-DUP-EBH7-W7-DISPOSITION-2026-11-01"

LAB_PATH_MARKERS = (
    "debug_lab",
    "/lab/",
    "\\lab\\",
    "ee_debug",
    "platform_proofs/",
)
MANDATORY_SCOPE_PREFIXES = ("intergrax/", "agents/", "applications/")
SUPPLEMENTARY_ROOTS = ("scripts/",)


@dataclass(frozen=True)
class ModuleRow:
    module_id: str
    path: str
    classification: str
    semantic_owner: str
    reachability_basis: str
    frz_hint: str
    notes: str


def _head_sha() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


def _inventory_implementation_sha() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD:scripts/governance/arch_freeze_b1_closed_world_inventory.py"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


def _git_ls_py(prefix: str) -> list[str]:
    r = subprocess.run(
        ["git", "ls-files", f"{prefix}**/*.py"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def _manifest_node_ids() -> frozenset[str]:
    from testing_support.change_safety.data.loaders import load_evol_x_data_document

    doc = load_evol_x_data_document("evol_vl3_execution_manifest.v1.json")
    return frozenset(doc["pytest_node_ids"])


def _qual_suite_paths() -> frozenset[str]:
    from tests.qualification.evol_x.adapters.qual_x import QualXSuiteCatalogAdapter

    return frozenset(n.split("::", 1)[0] for n in QualXSuiteCatalogAdapter().collectable_node_ids())


def _compat_module_paths() -> frozenset[str]:
    from intergrax.contracts.migrations.registry import CONTRACT_SCHEMA_REGISTRY

    return frozenset(e.module_path.replace(".", "/") + ".py" for e in CONTRACT_SCHEMA_REGISTRY)


def _classify_path(
    path: str,
    *,
    in_evol_inventory: bool,
    reachable_from_strict: bool,
    is_test: bool,
) -> tuple[str, str]:
    norm = path.replace("\\", "/")
    if is_test or norm.startswith("tests/"):
        return "TEST_ONLY", "tests/ tree"
    if any(m in norm for m in LAB_PATH_MARKERS):
        return "LAB_ONLY", "lab/debug marker path"
    if norm.startswith("platform_proofs/"):
        return "LAB_ONLY", "platform proof harness (EVOL exclusion register)"
    if reachable_from_strict:
        return "PRODUCTION_EFFECTIVE", "strict production graph reachability (PROD-Q P1 roots)"
    if norm.startswith("testing_support/"):
        return (
            "PRODUCTION_ELIGIBLE_UNVERIFIED",
            "qualification/discovery infrastructure (not in strict PROD-Q reachable set)",
        )
    if not any(norm.startswith(p) for p in MANDATORY_SCOPE_PREFIXES):
        if norm.startswith(SUPPLEMENTARY_ROOTS):
            return (
                "PRODUCTION_ELIGIBLE_UNVERIFIED",
                "governance/execution entrypoint tree (supplementary; unverified effectiveness)",
            )
        return "PRODUCTION_ELIGIBLE_UNVERIFIED", "adjacent repo surface (supplementary scan)"
    if not in_evol_inventory:
        return (
            "PRODUCTION_ELIGIBLE_UNVERIFIED",
            "mandatory tier path @ HEAD not in EVOL closed-world module nodes",
        )
    return (
        "PRODUCTION_ELIGIBLE_UNVERIFIED",
        "mandatory tier module in EVOL universe without strict reachability proof",
    )


def _build_gate_inputs(head: str, inv, owner_rules, exclusion_register):
    from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
    from testing_support.change_safety.discovery.closed_world_probes import ProbeInputs, run_cw07, run_mutation_probes
    from testing_support.change_safety.discovery.extractors import reset_fact_cache

    cold_b = build_closed_world_inventory(source_commit_sha=head, use_cache=False)
    reset_fact_cache()
    incremental = build_closed_world_inventory(source_commit_sha=head, use_cache=True)
    probe_inputs = ProbeInputs(inv, owner_rules, exclusion_register, _manifest_node_ids())
    mutation = run_mutation_probes(probe_inputs)
    not_flipped = [r.probe_id for r in mutation if not r.flipped]
    if not_flipped:
        raise RuntimeError(f"CW-07 mutation probes did not flip: {not_flipped}")
    cw07_report = run_cw07(probe_inputs)
    return {
        "collected_node_ids": _manifest_node_ids(),
        "second_build": cold_b,
        "incremental_build": incremental,
        "qual_suite_paths": _qual_suite_paths(),
        "compat_module_paths": _compat_module_paths(),
        "disposition_id": CW09_DISPOSITION_ID,
        "cw07_report": cw07_report,
    }


def _load_dispositions(output_dir: Path) -> dict[str, str]:
    candidates = (
        output_dir / "arch_freeze_b1_mandatory_dispositions.json",
        _REPO / "docs/project/maintainers/qualification/arch_freeze_b1_mandatory_dispositions.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} must be a JSON object")
        return {str(k): str(v) for k, v in data.items()}
    return {}


def run_scan(
    *,
    output_dir: Path,
    audit_baseline_sha: str,
    strict_certify: bool,
) -> dict:
    head = _head_sha()
    output_dir.mkdir(parents=True, exist_ok=True)

    from testing_support.change_safety.data.loaders import load_evol_x_data_document
    from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
    from testing_support.change_safety.discovery.closed_world_gates import run_closed_world_gates
    from tests.qualification.prod_q._p0_inventory import build_closed_world_discovery_report
    from tests.qualification.prod_q._prod_q_r1_r7_q_p1_strict_reachability import (
        STRICT_PRODUCTION_ROOTS,
        analyze_strict_production_reachability,
    )

    inv = build_closed_world_inventory(source_commit_sha=head, use_cache=False)
    owner_rules = load_evol_x_data_document("owner_rules.v1.json")
    exclusion_register = load_evol_x_data_document("exclusion_register.v1.json")
    gate_kw = _build_gate_inputs(head, inv, owner_rules, exclusion_register)
    gate_report = run_closed_world_gates(
        inv,
        owner_rules=owner_rules,
        exclusion_register=exclusion_register,
        **gate_kw,
    )
    prod_p0 = build_closed_world_discovery_report()
    reach = analyze_strict_production_reachability()
    strict_path_files = {p.replace("\\", "/") for p, _ in reach.reachable_functions}

    inv_paths = {n.path for n in inv.nodes if n.category == "module" and n.path}
    all_py: set[str] = set(inv_paths)
    for prefix in MANDATORY_SCOPE_PREFIXES + SUPPLEMENTARY_ROOTS:
        all_py.update(_git_ls_py(prefix))

    rows: list[ModuleRow] = []
    for path in sorted(all_py):
        norm = path.replace("\\", "/")
        mid = norm.replace("/", ".").removesuffix(".py")
        owners = []
        for owner in owner_rules.get("owners", ()):
            for prefix in owner.get("paths_prefix", ()):
                p = str(prefix).rstrip("/")
                if norm == p or norm.startswith(f"{p}/"):
                    owners.append(str(owner["owner_id"]))
        owner_s = ",".join(sorted(set(owners))) or "UNASSIGNED"
        reachable = norm in strict_path_files
        cls, basis = _classify_path(
            path,
            in_evol_inventory=norm in inv_paths,
            reachable_from_strict=reachable,
            is_test=False,
        )
        frz = "FRZ-BND" if "intergrax/" in norm else "FRZ-PRD" if norm.startswith("applications/") else "FRZ-EXE"
        rows.append(
            ModuleRow(
                module_id=mid,
                path=norm,
                classification=cls,
                semantic_owner=owner_s,
                reachability_basis=basis,
                frz_hint=frz,
                notes="",
            )
        )

    unknown_nodes = [u for u in inv.unknowns if u.startswith("unknown:")]
    unresolved_mandatory = [
        (src, imp)
        for src, imp in inv.unresolved_imports
        if any(src.startswith(p.replace("/", ".")) for p in MANDATORY_SCOPE_PREFIXES)
        and imp.split(".")[0] not in {"docs", "exports", "build", "dist"}
    ]

    dispositions = _load_dispositions(output_dir)
    mandatory_keys = list(unknown_nodes) + [f"import:{src}:{imp}" for src, imp in unresolved_mandatory]
    undisposed = [k for k in mandatory_keys if k not in dispositions]

    counts = Counter(r.classification for r in rows)
    unknown_class_count = counts.get("UNKNOWN", 0)
    gate_failures = [g.gate_id for g in gate_report.checks if not g.passed]

    findings: list[dict] = []
    if unresolved_mandatory and undisposed:
        findings.append(
            {
                "FINDING_ID": "FIND-CW-UNRESOLVED-IMPORTS",
                "SEVERITY": "MAJ",
                "SOURCE_PATH": "closed_world.unresolved_imports",
                "REACHABILITY": f"{len(unresolved_mandatory)} import records",
                "SEMANTIC_OWNER": "COMPAT-X",
                "INVARIANT": "closed-world import resolution",
                "FRZ_ID": "FRZ-BND-03,FRZ-CTR-02",
                "IMPACT": "Static graph incomplete; behavioral proof required for affected paths",
                "REQUIRED_REMEDIATION": "T-B1-MANDATORY-DISPOSITION",
                "TEST": "CW-01..CW-06; strict reachability pytest",
                "ACCEPTANCE": "Zero mandatory-scope unresolved OR evidence-backed N/A",
            }
        )
    if unknown_nodes and any(k.startswith("unknown:") for k in undisposed):
        findings.append(
            {
                "FINDING_ID": "FIND-CW-EXTRACTOR-UNKNOWNS",
                "SEVERITY": "MAJ",
                "SOURCE_PATH": "closed_world.unknowns",
                "REACHABILITY": f"{len(unknown_nodes)} unknown nodes",
                "SEMANTIC_OWNER": "COMPAT-X",
                "INVARIANT": "extractor completeness",
                "FRZ_ID": "FRZ-BND-03,FRZ-CTR-02",
                "IMPACT": "Extractor unknown nodes without disposition",
                "REQUIRED_REMEDIATION": "T-B1-MANDATORY-DISPOSITION",
                "TEST": "test_evol_x_closed_world_gates.py",
                "ACCEPTANCE": "All mandatory unknowns dispositioned",
            }
        )
    if gate_failures:
        findings.append(
            {
                "FINDING_ID": "FIND-CW-GATE-FAIL",
                "SEVERITY": "BLOCKER" if "CW-01" in gate_failures else "MAJ",
                "SOURCE_PATH": "closed_world_gates",
                "REACHABILITY": ",".join(gate_failures),
                "SEMANTIC_OWNER": "COMPAT-X",
                "INVARIANT": "EVOL closed-world gates @ HEAD",
                "FRZ_ID": "FRZ-CMP-01",
                "IMPACT": "Closed-world inventory gates not green on current HEAD",
                "REQUIRED_REMEDIATION": "T-B1-ENGINE; T-B1-STRICT-QUALIFY",
                "TEST": "tests/qualification/evol_x/gates/test_evol_x_closed_world_gates.py",
                "ACCEPTANCE": "All CW gates PASS @ execution SHA",
            }
        )

    strict_fail_reasons: list[str] = []
    if strict_certify:
        if gate_failures:
            strict_fail_reasons.append(f"cw_gates:{','.join(gate_failures)}")
        if undisposed:
            strict_fail_reasons.append(f"undisposed_mandatory:{len(undisposed)}")

    payload = {
        "execution_sha": head,
        "audit_baseline_sha": audit_baseline_sha,
        "inventory_implementation_sha": _inventory_implementation_sha(),
        "inventory_digest": inv.inventory_digest,
        "module_count": len(rows),
        "classification_counts": dict(counts),
        "production_effective": counts.get("PRODUCTION_EFFECTIVE", 0),
        "production_eligible_unverified": counts.get("PRODUCTION_ELIGIBLE_UNVERIFIED", 0),
        "test_only": counts.get("TEST_ONLY", 0),
        "lab_only": counts.get("LAB_ONLY", 0),
        "dead_code": counts.get("DEAD_CODE", 0),
        "unknown_modules": unknown_class_count,
        "evol_module_nodes": len(inv_paths),
        "universe_paths": len(inv.universe_paths),
        "excluded_paths": len(inv.excluded_paths),
        "extractor_unknown_nodes": len(unknown_nodes),
        "unresolved_import_pairs": len(inv.unresolved_imports),
        "unresolved_mandatory_pairs": len(unresolved_mandatory),
        "mandatory_disposition_count": len(dispositions),
        "mandatory_undisposed_count": len(undisposed),
        "strict_production_reachable_modules": len(strict_path_files),
        "prod_p0_raw_candidates": len(prod_p0.raw_candidates),
        "prod_p0_unclassified": len(prod_p0.unclassified_candidate_ids),
        "closed_world_gates_pass": not gate_failures,
        "closed_world_gate_failures": gate_failures,
        "strict_certify": strict_certify,
        "strict_fail_reasons": strict_fail_reasons,
        "strict_pass": not strict_fail_reasons,
        "strict_production_roots": [{"path": p, "symbol": s} for p, s in STRICT_PRODUCTION_ROOTS],
        "findings": findings,
        "limitations": [
            "scripts/ and tests/ use supplementary git ls-files; EVOL DISCOVERY_ROOTS omit scripts/.",
            "Dynamic import/config-driven paths require behavioral validation per ECAP.",
            f"Extractor unknown nodes @ HEAD: {len(unknown_nodes)} (recorded in mandatory dispositions).",
        ],
    }

    (output_dir / "arch_freeze_b1_inventory_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "arch_freeze_b1_module_inventory.jsonl").write_text(
        "\n".join(json.dumps(asdict(r), sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    md_lines = [
        "# ARCH-FREEZE B1 closed-world inventory run",
        "",
        f"**EXECUTION_SHA:** `{head}`",
        f"**inventory_implementation_sha:** `{payload['inventory_implementation_sha']}`",
        f"**audit_baseline_sha:** `{audit_baseline_sha}`",
        f"**inventory_digest:** `{inv.inventory_digest}`",
        f"**strict_certify:** `{strict_certify}` · **strict_pass:** `{payload['strict_pass']}`",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for k, v in payload["classification_counts"].items():
        md_lines.append(f"| {k} | {v} |")
    for k in (
        "module_count",
        "production_effective",
        "production_eligible_unverified",
        "unknown_modules",
        "unresolved_mandatory_pairs",
        "mandatory_undisposed_count",
        "closed_world_gates_pass",
    ):
        md_lines.append(f"| {k} | {payload[k]} |")
    md_lines.append("")
    md_lines.append("## Findings")
    for f in findings:
        md_lines.append(f"- **{f['FINDING_ID']}** ({f['SEVERITY']}): {f['IMPACT']}")
    (output_dir / "ARCH_FREEZE_B1_INVENTORY_RUN.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / ".tmp/session/arch-freeze-b1-cert",
    )
    parser.add_argument(
        "--audit-baseline-sha",
        default=DEFAULT_AUDIT_BASELINE_SHA,
        help="Planning lock SHA (AUDIT_BASELINE_SHA)",
    )
    parser.add_argument(
        "--strict-certify",
        action="store_true",
        help="Fail closed (exit 1) when CW gates or mandatory dispositions fail",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Always exit 0 after writing artifacts (warnings on stderr)",
    )
    args = parser.parse_args()
    strict = args.strict_certify and not args.report_only
    payload = run_scan(
        output_dir=args.output_dir,
        audit_baseline_sha=args.audit_baseline_sha,
        strict_certify=strict,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "execution_sha": payload["execution_sha"],
                "strict_pass": payload["strict_pass"],
                **payload["classification_counts"],
            }
        )
    )
    if payload["closed_world_gate_failures"]:
        print("[WARN] closed_world_gate_failures:", payload["closed_world_gate_failures"], file=sys.stderr)
    if payload["strict_fail_reasons"]:
        print("[WARN] strict_fail_reasons:", payload["strict_fail_reasons"], file=sys.stderr)
    if strict and not payload["strict_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
