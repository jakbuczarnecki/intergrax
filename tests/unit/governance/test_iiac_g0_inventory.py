"""IIAC G0 inventory completeness and F01 regression tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.governance.iiac_g0_inventory import (
    GOVERNANCE_IMPLEMENTATION_SHA,
    MANDATORY_IIAC_GOVERNANCE_PATHS,
    build_matrix_payload,
    enumerate_governed_paths_at_ref,
    git_rev_parse,
    unique_paths_in_entries,
)
from scripts.governance.verify_iiac_g0_inventory_completeness import validate_matrix_dict

REPO = Path(__file__).resolve().parents[3]
MATRIX = REPO / "docs/project/maintainers/qualification/IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json"
FIXTURES = REPO / "tests/fixtures/iiac"
SCRIPTS = REPO / "scripts/governance"


def _run(script: str, *args: str) -> int:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode


def test_governed_enumeration_independent_of_generator_output() -> None:
    head = git_rev_parse("HEAD")
    governed = enumerate_governed_paths_at_ref(head)
    assert len(governed) > 3000
    assert (
        "docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md"
        in governed
    )
    assert "docs/project/maintainers/qualification/IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json" not in governed


def test_committed_matrix_passes_completeness_at_head() -> None:
    assert MATRIX.is_file()
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    errors = validate_matrix_dict(payload, expect_head="HEAD")
    assert not errors, errors


def test_mandatory_iiac_paths_covered_in_committed_matrix() -> None:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    covered = unique_paths_in_entries(payload["entries"])
    missing = sorted(MANDATORY_IIAC_GOVERNANCE_PATHS - covered)
    assert not missing, missing


def test_f01_stale_session_a_matrix_rejected() -> None:
    stale_path = FIXTURES / "stale_g0_matrix_f01.json"
    assert stale_path.is_file()
    stale = json.loads(stale_path.read_text(encoding="utf-8"))
    errors = validate_matrix_dict(stale, allow_stale_source=True)
    assert any("mandatory IIAC governance path missing" in e for e in errors)
    assert any("INTEGRAX_INTERNAL" in e or "verify_iiac" in e or "missing" in e for e in errors)


def test_stale_head_rejected_for_committed_matrix(tmp_path: Path) -> None:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    payload["inventory_source_sha"] = "0000000000000000000000000000000000000001"
    errors = validate_matrix_dict(payload, expect_head="HEAD")
    assert any("not ancestor of HEAD" in e for e in errors)


def test_generator_regression_via_completeness_script() -> None:
    rc = _run("verify_iiac_g0_inventory_completeness.py", str(MATRIX))
    assert rc == 0


def test_build_payload_covers_all_governed_paths() -> None:
    head = git_rev_parse("HEAD")
    payload = build_matrix_payload(head)
    governed = set(enumerate_governed_paths_at_ref(head))
    covered = unique_paths_in_entries(payload["entries"])
    assert governed == covered


def test_classify_prod_q_preserves_production_category_not_b6_gate() -> None:
    from scripts.governance.iiac_g0_inventory import classify_text

    text = "PROD-Q live qualification; FRZ-PRD-01 remains OPEN."
    entries = classify_text("docs/example.md", text)
    cats = {int(e["category"]) for e in entries}
    assert 4 in cats
    assert 1 not in cats


def test_governance_implementation_sha_preserved_in_matrix() -> None:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert payload.get("governance_implementation_sha_historical") == GOVERNANCE_IMPLEMENTATION_SHA
