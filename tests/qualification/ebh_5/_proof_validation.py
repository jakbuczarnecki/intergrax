# © Artur Czarnecki. All rights reserved.

"""Proof reference validation — reject pozorne REPLACEABLE — QUALIFIED (EBH5-A01/A04)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.qualification.ebh_5._behavioral_proof_registry import (
    QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE,
    QUALIFIED_CONCERN_IDS,
)
from tests.qualification.ebh_5._vendor_backend_types import (
    ExtensibilityClassification,
    ReplaceabilityMatrixRow,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_PYTEST_NODE_RE = re.compile(r"^(.+\.py)::(test_[\w]+)$")


def is_valid_git_sha40(value: str) -> bool:
    return bool(_SHA40_RE.match(value.strip().lower()))


def parse_pytest_node_id(node_id: str) -> tuple[str, str] | None:
    match = _PYTEST_NODE_RE.match(node_id.strip())
    if match is None:
        return None
    return match.group(1), match.group(2)


def pytest_test_function_exists(node_id: str) -> bool:
    parsed = parse_pytest_node_id(node_id)
    if parsed is None:
        return False
    rel_path, func_name = parsed
    path = _REPO_ROOT / rel_path
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return True
    return False


def proof_must_live_in_ebh5_qualification_suite(node_id: str) -> bool:
    parsed = parse_pytest_node_id(node_id)
    if parsed is None:
        return False
    rel_path, _ = parsed
    return rel_path.replace("\\", "/").startswith("tests/qualification/ebh_5/")


def validate_behavioral_proof_node_for_concern(concern_id: str, behavioral_proof_ref: str) -> str | None:
    """Return error message or None when valid."""
    if behavioral_proof_ref in ("", "PENDING", "PENDING — EBH_5 Package C"):
        return "behavioral_proof_ref is pending or empty"
    if "@" in behavioral_proof_ref and not parse_pytest_node_id(behavioral_proof_ref):
        return "historical narrative proof ref is not a resolvable pytest node"
    if not parse_pytest_node_id(behavioral_proof_ref):
        return "behavioral_proof_ref is not a canonical pytest node id"
    if not proof_must_live_in_ebh5_qualification_suite(behavioral_proof_ref):
        return "proof must live under tests/qualification/ebh_5/"
    if not pytest_test_function_exists(behavioral_proof_ref):
        return "pytest node does not resolve to an existing test function"
    expected = QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE.get(concern_id)
    if expected is None:
        return f"concern_id {concern_id!r} is not in QUALIFIED registry"
    if behavioral_proof_ref != expected:
        return f"proof bound to wrong concern (expected {expected!r})"
    return None


def validate_qualified_matrix_row(row: ReplaceabilityMatrixRow) -> str | None:
    if row.rpl_h != ExtensibilityClassification.REPLACEABLE_QUALIFIED:
        return None
    err = validate_behavioral_proof_node_for_concern(row.concern_id, row.behavioral_proof_ref)
    if err:
        return f"{row.concern_id}: {err}"
    if row.concern_id not in QUALIFIED_CONCERN_IDS:
        return f"{row.concern_id}: QUALIFIED row not in canonical registry"
    from tests.qualification.ebh_5._rpl_f_evidence_registry import rpl_f_disposition

    rpl_f = rpl_f_disposition(row.concern_id)
    if rpl_f is None:
        return f"{row.concern_id}: missing RPL-F disposition"
    if row.rpl_f != rpl_f.gate:
        return f"{row.concern_id}: matrix rpl_f {row.rpl_f} != registry {rpl_f.gate}"
    return None


def qualified_rows_with_invalid_proofs(
    rows: tuple[ReplaceabilityMatrixRow, ...],
) -> tuple[str, ...]:
    bad: list[str] = []
    for row in rows:
        err = validate_qualified_matrix_row(row)
        if err:
            bad.append(err)
    return tuple(bad)


def undocumented_qualified_rows(
    rows: tuple[ReplaceabilityMatrixRow, ...],
) -> tuple[str, ...]:
    """Legacy name — now full proof integrity (not non-empty string check)."""
    return tuple(
        row.concern_id
        for row in rows
        if row.rpl_h == ExtensibilityClassification.REPLACEABLE_QUALIFIED
        and validate_qualified_matrix_row(row) is not None
    )
