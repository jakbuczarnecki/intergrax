#!/usr/bin/env python3
"""One-shot inventory: classify authority references for IIAC migration (G0)."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "docs/project/maintainers/qualification/IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json"
GLOB = "*.{md,yml,yaml,py,mdc}"

PATTERN_RULES: list[tuple[int, str, str, str]] = [
    (
        1,
        r"Protocol\s*03|independent\s+human|human\s+auditor|human-signed|"
        r"external\s+auditor|mandatory\s+external\s+human",
        "qualification/governance",
        "Amend under IIAC-B6; preserve Mode A/B/C semantics",
    ),
    (
        1,
        r"\bB6\b|OD-EVOL-11|OD-ENT-03",
        "stage-closure",
        "Map to IIAC manifest + pass record; no fabricated human id",
    ),
    (
        2,
        r"operator\s+(approval|FRZ|closure|promotion)|certifier|"
        r"governance\s+agent|independent-quality-auditor",
        "governance",
        "Map to certifier gate / role registry; preserve separation",
    ),
    (
        3,
        r"SYS-INV-|runtime\s+auth|fail-closed|SCENARIO-GATE",
        "platform-runtime",
        "PRESERVE — no IIAC authority substitution",
    ),
    (
        4,
        r"PROD-Q|production\s+deployment|FRZ-PRD-",
        "production-qual",
        "PRESERVE — external live qualification unchanged",
    ),
    (
        5,
        r"third[- ]party|legal|vendor\s+contract",
        "external/legal",
        "PRESERVE",
    ),
    (
        6,
        r"EBH-6|historical\s+evidence|category\s+6|human\s+Protocol\s+03\s+records",
        "historical",
        "PRESERVE as historical authority; mark authority=historical",
    ),
    (
        7,
        r"ECAP-P4-EVOL-SCOPE-AUTO-B6|superseded|obsolete|NOT_IN_FORCE",
        "legacy-norm",
        "Supersede with IIAC migration matrix pointer",
    ),
    (
        1,
        r"OD-[A-Z]+-\d+|FRZ-[A-Z]+-\d+",
        "qualification",
        "Cross-check IIAC_REQUIREMENT_MIGRATION_MATRIX.md",
    ),
    (
        1,
        r"CLOSED\s*—\s*INDEPENDENTLY|stage\s+CLOSED",
        "closure-vocabulary",
        "Use IIAC closure canon when policy IN_FORCE",
    ),
]


def _repo_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _rg_list_files(pattern: str) -> list[str]:
    proc = subprocess.run(
        [
            "rg",
            "-l",
            "--glob",
            GLOB,
            "--glob",
            "!.git/**",
            "--glob",
            "!.venv/**",
            "--glob",
            "!.tmp/**",
            "--glob",
            "!node_modules/**",
            pattern,
            ".",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        return []
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="JSON output path (default: qualification IIAC matrix)",
    )
    args = parser.parse_args()

    entries: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()
    for cat, pattern, owner_hint, action in PATTERN_RULES:
        for rel in _rg_list_files(pattern):
            key = (cat, rel)
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "category": cat,
                    "path": rel,
                    "owner_hint": owner_hint,
                    "migration_action": action,
                    "matched_pattern": pattern[:120],
                }
            )

    entries.sort(key=lambda e: (int(e["category"]), str(e["path"])))

    payload = {
        "schema_version": "iiac-authority-impact.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "repository_head_at_generation": _repo_head(),
        "entry_count": len(entries),
        "category_legend": {
            "1": "Mandatory external human / stage closure → CHANGE under IIAC",
            "2": "Internal governance approval → certifier gate",
            "3": "Runtime auth/security → PRESERVE",
            "4": "Production deployment → PRESERVE",
            "5": "Third-party/legal → PRESERVE",
            "6": "Historical evidence → PRESERVE",
            "7": "Obsolete/conflicting → supersede via migration matrix",
        },
        "entries": entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"IIAC authority impact matrix: {len(entries)} entries → {args.output.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
