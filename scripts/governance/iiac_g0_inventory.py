#!/usr/bin/env python3
"""IIAC G0 authority impact inventory — governed path enumeration and classification."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = REPO / "docs/project/maintainers/qualification/IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json"
GOVERNANCE_IMPLEMENTATION_SHA = "2ae1014116d353bc7742985e62f2bccc2bfac798"

INCLUSION_RULES_DOCUMENTED: list[str] = [
    "All git-tracked paths at inventory_source_sha with suffix in "
    "{.md,.mdc,.py,.yaml,.yml,.json,.toml,.ini,.cfg,.txt,.sh,.ps1,.bpmn,.xml,.wsdl}",
    "Cursor rules (.mdc) and GitHub workflow YAML under .github/",
]

EXCLUSION_RULES_DOCUMENTED: list[str] = [
    "Prefixes: .tmp/, node_modules/, .venv/, __pycache__/, .git/",
    "Generated G0 output: docs/project/maintainers/qualification/IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json",
    "Binary/media suffixes: .pyc,.png,.jpg,.jpeg,.gif,.webp,.ico,.pdf,.zip,.whl,.exe,.dll,.so,.bin,.woff,.woff2",
    "Lock/vendor blobs: uv.lock (vendor pin — not authority-bearing source)",
]

INCLUDED_SUFFIXES = frozenset(
    {
        ".md",
        ".mdc",
        ".py",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".ini",
        ".cfg",
        ".txt",
        ".sh",
        ".ps1",
        ".bpmn",
        ".xml",
        ".wsdl",
    }
)

EXCLUDED_PREFIXES = (".tmp/", "node_modules/", ".venv/", "__pycache__/", ".git/")

EXCLUDED_EXACT = frozenset(
    {
        "docs/project/maintainers/qualification/IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json",
        "uv.lock",
    }
)

BINARY_SUFFIXES = frozenset(
    {
        ".pyc",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".zip",
        ".whl",
        ".exe",
        ".dll",
        ".so",
        ".bin",
        ".woff",
        ".woff2",
    }
)

MANDATORY_IIAC_GOVERNANCE_PATHS: frozenset[str] = frozenset(
    {
        "docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md",
        "docs/project/maintainers/qualification/IIAC_REQUIREMENT_MIGRATION_MATRIX.md",
        "docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json",
        "docs/project/maintainers/qualification/IIAC_CERTIFICATION_AGENT_PUBLICATION_CONTRACT.md",
        "docs/project/maintainers/qualification/IIAC_SESSION_B_INDEPENDENT_GOVERNANCE_AUDITOR_INSTRUCTIONS.md",
        "docs/project/maintainers/qualification/IIAC_SESSION_C_EVOL_INDEPENDENT_AUDITOR_INSTRUCTIONS.md",
        "docs/project/maintainers/qualification/IIAC_SESSION_D_ENT_INDEPENDENT_AUDITOR_INSTRUCTIONS.md",
        "docs/project/maintainers/qualification/IIAC_SESSION_HANDOFF_RECORD.template.md",
        "docs/project/maintainers/qualification/IIAC_SESSION_HANDOFF_RECORD_2026_11_05_SESSION_A_B.md",
        "docs/project/maintainers/qualification/schemas/iiac_audit_session_manifest.v1.schema.json",
        "docs/project/maintainers/qualification/examples/iiac_audit_session_manifest.example.json",
        "docs/project/maintainers/qualification/ent_audit_x/OD_ENT_02_THREAT_MODEL.md",
        "docs/project/maintainers/qualification/ent_audit_x/OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md",
        "governance/tickets/T-IGX-016-A.md",
        "scripts/governance/generate_iiac_authority_impact_matrix.py",
        "scripts/governance/iiac_g0_inventory.py",
        "scripts/governance/iiac_common.py",
        "scripts/governance/verify_iiac_audit_manifest.py",
        "scripts/governance/verify_iiac_policy_activation.py",
        "scripts/governance/verify_iiac_role_separation.py",
        "scripts/governance/verify_od_ent_02_compensating_control.py",
        "scripts/governance/verify_iiac_g0_inventory_completeness.py",
        "tests/unit/governance/test_iiac_negative_guards.py",
        "tests/unit/governance/test_iiac_g0_inventory.py",
    }
)

NORMATIVE_CATEGORY_REQUIREMENTS: dict[str, set[int]] = {
    "docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md": {
        1,
        2,
    },
}

PATTERN_RULES: list[tuple[int, re.Pattern[str], str, str]] = [
    (
        3,
        re.compile(
            r"SYS-INV-|runtime\s+auth|fail-closed|SCENARIO-GATE",
            re.IGNORECASE,
        ),
        "platform-runtime",
        "PRESERVE — no IIAC authority substitution",
    ),
    (
        4,
        re.compile(r"PROD-Q|production\s+deployment|FRZ-PRD-", re.IGNORECASE),
        "production-qual",
        "PRESERVE — external live qualification unchanged",
    ),
    (
        5,
        re.compile(r"third[- ]party|legal|vendor\s+contract", re.IGNORECASE),
        "external/legal",
        "PRESERVE",
    ),
    (
        6,
        re.compile(
            r"EBH-6|historical\s+evidence|category\s+6|human\s+Protocol\s+03\s+records",
            re.IGNORECASE,
        ),
        "historical",
        "PRESERVE as historical authority; mark authority=historical",
    ),
    (
        7,
        re.compile(
            r"ECAP-P4-EVOL-SCOPE-AUTO-B6|superseded|obsolete|NOT_IN_FORCE",
            re.IGNORECASE,
        ),
        "legacy-norm",
        "Supersede with IIAC migration matrix pointer",
    ),
    (
        1,
        re.compile(
            r"Protocol\s*03|independent\s+human|human\s+auditor|human-signed|"
            r"external\s+auditor|mandatory\s+external\s+human",
            re.IGNORECASE,
        ),
        "qualification/governance",
        "Amend under IIAC-B6; preserve Mode A/B/C semantics",
    ),
    (
        1,
        re.compile(r"\bB6\b|OD-EVOL-11|OD-ENT-03", re.IGNORECASE),
        "stage-closure",
        "Map to IIAC manifest + pass record; no fabricated human id",
    ),
    (
        1,
        re.compile(r"OD-[A-Z]+-\d+|FRZ-[A-Z]+-\d+", re.IGNORECASE),
        "qualification",
        "Cross-check IIAC_REQUIREMENT_MIGRATION_MATRIX.md",
    ),
    (
        1,
        re.compile(r"CLOSED\s*—\s*INDEPENDENTLY|stage\s+CLOSED", re.IGNORECASE),
        "closure-vocabulary",
        "Use IIAC closure canon when policy IN_FORCE",
    ),
    (
        2,
        re.compile(
            r"operator\s+(approval|FRZ|closure|promotion)|certifier|"
            r"governance\s+agent|independent-quality-auditor|IIAC",
            re.IGNORECASE,
        ),
        "governance",
        "Map to certifier gate / role registry; preserve separation",
    ),
]

CATEGORY_LEGEND: dict[str, str] = {
    "1": "Mandatory external human / stage closure → CHANGE under IIAC",
    "2": "Internal governance approval → certifier gate",
    "3": "Runtime auth/security → PRESERVE",
    "4": "Production deployment → PRESERVE",
    "5": "Third-party/legal → PRESERVE",
    "6": "Historical evidence → PRESERVE",
    "7": "Obsolete/conflicting → supersede via migration matrix",
}

DEFAULT_COVERAGE_ENTRY: tuple[int, str, str, str] = (
    2,
    "inventory/neutral",
    "G0 enumerate-only — no authority keyword match; PRESERVE substance pending review",
    "g0-enumeration-default",
)


def normalize_rel_path(path: str) -> str:
    return path.strip().replace("\\", "/")


def is_excluded(rel: str) -> bool:
    rel = normalize_rel_path(rel)
    if rel in EXCLUDED_EXACT:
        return True
    if any(rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return True
    suffix = PurePosixPath(rel).suffix.lower()
    if suffix in BINARY_SUFFIXES:
        return True
    return False


def is_included(rel: str) -> bool:
    rel = normalize_rel_path(rel)
    if is_excluded(rel):
        return False
    suffix = PurePosixPath(rel).suffix.lower()
    return suffix in INCLUDED_SUFFIXES


def git_merge_base_is_ancestor(ancestor: str, descendant: str) -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def git_rev_parse(ref: str = "HEAD") -> str:
    proc = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git rev-parse {ref} failed")
    return proc.stdout.strip()


def git_ls_tree_paths(ref: str) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git ls-tree failed for {ref}")
    return [normalize_rel_path(line) for line in proc.stdout.splitlines() if line.strip()]


def enumerate_governed_paths_at_ref(ref: str) -> list[str]:
    """Independent governed-file inventory (not derived from pattern search)."""
    resolved = git_rev_parse(ref)
    try:
        head = git_rev_parse("HEAD")
    except RuntimeError:
        head = ""
    if resolved == head:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        proc = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", resolved],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git file list failed for {ref}")
    paths = [normalize_rel_path(line) for line in proc.stdout.splitlines() if line.strip()]
    governed = sorted({p for p in paths if is_included(p)})
    return governed


def read_text_at_ref(
    resolved_sha: str,
    rel_path: str,
    *,
    head_sha: str,
) -> str:
    if resolved_sha == head_sha:
        path = REPO / rel_path
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    proc = subprocess.run(
        ["git", "show", f"{resolved_sha}:{rel_path}"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return ""
    return proc.stdout.decode("utf-8", errors="replace")


def generator_blob_sha_at_ref(ref: str) -> str:
    rel = "scripts/governance/iiac_g0_inventory.py"
    proc = subprocess.run(
        ["git", "hash-object", str(REPO / rel)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    return proc.stdout.strip()


def classify_text(rel_path: str, text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    preserve_only = False
    for cat, pattern, owner_hint, action in PATTERN_RULES:
        if cat in (3, 4, 5) and pattern.search(text):
            preserve_only = True
        if not pattern.search(text):
            continue
        key = (cat, rel_path)
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "category": cat,
                "path": rel_path,
                "owner_hint": owner_hint,
                "migration_action": action,
                "matched_pattern": pattern.pattern[:120],
            }
        )
    if preserve_only:
        entries = [e for e in entries if int(e["category"]) not in (1,)]
    if not entries:
        cat, owner_hint, action, pattern_id = DEFAULT_COVERAGE_ENTRY
        entries.append(
            {
                "category": cat,
                "path": rel_path,
                "owner_hint": owner_hint,
                "migration_action": action,
                "matched_pattern": pattern_id,
            }
        )
    return entries


def build_entries_for_ref(ref: str) -> tuple[list[dict[str, Any]], list[str]]:
    resolved = git_rev_parse(ref)
    head_sha = git_rev_parse("HEAD")
    governed = enumerate_governed_paths_at_ref(resolved)
    all_entries: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, str]] = set()
    for rel in governed:
        text = read_text_at_ref(resolved, rel, head_sha=head_sha)
        for entry in classify_text(rel, text):
            key = (int(entry["category"]), str(entry["path"]))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_entries.append(entry)
    for mandatory in sorted(MANDATORY_IIAC_GOVERNANCE_PATHS):
        if mandatory not in governed:
            continue
        covered = any(e["path"] == mandatory for e in all_entries)
        if covered:
            continue
        cat, owner_hint, action, pattern_id = DEFAULT_COVERAGE_ENTRY
        all_entries.append(
            {
                "category": cat,
                "path": mandatory,
                "owner_hint": owner_hint,
                "migration_action": action + " (mandatory IIAC path)",
                "matched_pattern": pattern_id,
            }
        )
    all_entries.sort(key=lambda e: (int(e["category"]), str(e["path"])))
    return all_entries, governed


def category_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {str(i): 0 for i in range(1, 8)}
    for entry in entries:
        cat = str(entry.get("category"))
        if cat in counts:
            counts[cat] += 1
    return counts


def unique_paths_in_entries(entries: list[dict[str, Any]]) -> set[str]:
    return {str(e["path"]) for e in entries}


def prior_inventory_paths_at_sha(sha: str) -> set[str]:
    proc = subprocess.run(
        [
            "git",
            "show",
            f"{sha}:docs/project/maintainers/qualification/IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return set()
    payload = json.loads(proc.stdout)
    return unique_paths_in_entries(payload.get("entries") or [])


def build_matrix_payload(
    inventory_source_ref: str = "HEAD",
    *,
    inventory_publication_sha: str | None = None,
) -> dict[str, Any]:
    source_sha = git_rev_parse(inventory_source_ref)
    entries, governed = build_entries_for_ref(source_sha)
    covered = unique_paths_in_entries(entries)
    prior_paths = prior_inventory_paths_at_sha(GOVERNANCE_IMPLEMENTATION_SHA)
    new_paths = sorted(covered - prior_paths)
    gen_sha = generator_blob_sha_at_ref(source_sha)
    return {
        "schema_version": "iiac-authority-impact.v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "governance_implementation_sha_historical": GOVERNANCE_IMPLEMENTATION_SHA,
        "inventory_source_sha": source_sha,
        "generator_sha": gen_sha,
        "inventory_publication_sha": inventory_publication_sha,
        "repository_head_at_generation": source_sha,
        "covered_path_count": len(covered),
        "entry_count": len(entries),
        "category_counts": category_counts(entries),
        "new_paths_since_governance_implementation_sha": new_paths,
        "inclusion_rules_documented": INCLUSION_RULES_DOCUMENTED,
        "exclusion_rules_documented": EXCLUSION_RULES_DOCUMENTED,
        "governed_path_count": len(governed),
        "category_legend": CATEGORY_LEGEND,
        "entries": entries,
    }


def load_matrix(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
