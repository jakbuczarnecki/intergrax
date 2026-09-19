# © Artur Czarnecki. All rights reserved.

"""MP-6E-C1 — contract-driven read composition and canonical documentation gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.collaborative_work.collaborative_activity_composition import (
    build_collaborative_activity_read_service,
    build_sqlite_collaborative_activity_read_store,
)
from intergrax.contracts.collaborative_activity import CollaborativeActivityReadPort
from intergrax.contracts.collaborative_activity_read import (
    CollaborativeActivityReadAuthorizationPolicyInput,
    CollaborativeActivityReadDenied,
    CollaborativeActivityReadDenialReason,
    fail_closed_collaborative_activity_read_decision,
)
from tests.unit.collaborative_work.test_mp6e_collaborative_activity_read import (
    _read_request,
    _seed_authority,
)
from tests.unit.docs._ee_canonical_documentation_support import (
    MOJIBAKE_PATTERN,
    repair_mojibake,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSITION = (
    _REPO_ROOT / "intergrax" / "collaborative_work" / "collaborative_activity_composition.py"
)
_READ_STORE = _REPO_ROOT / "intergrax" / "collaborative_work" / "collaborative_activity_read_store.py"

_BOUNDED_MP6_DOCS: tuple[Path, ...] = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "technical"
    / "adr"
    / "entries"
    / "2026-09-18"
    / "ADR-MP-007.md",
    _REPO_ROOT / "docs" / "project" / "architecture" / "COLLABORATIVE_WORK.md",
    _REPO_ROOT / "docs" / "project" / "capabilities" / "architecture" / "MULTIPLAYER_AI.md",
    _REPO_ROOT / "docs" / "project" / "capabilities" / "plan" / "MULTIPLAYER_AI.md",
    _REPO_ROOT / "docs" / "project" / "maintainers" / "plans" / "COLLABORATIVE_WORK.md",
)

_MP6D_ROW = re.compile(
    r"^\|\s*MP-6D\s*\|",
    re.MULTILINE,
)

_MP6E_C1_PG_QUAL_EVIDENCE = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "MP-6E-C1_POSTGRESQL_READ_PROVIDER_QUALIFICATION.md"
)

_MP6E_C1_Q1_REQUIRED_MARKERS: tuple[str, ...] = (
    "MP-6E-C1-Q1",
    "implementation_sha",
    "eb586ccefea80ff05e0dd4f4f1e784ed1c398350",
    "mp6e_c1_correction_sha",
    "3c7b45d63e3c2d9dfd1afd22101506983bf25ce0",
    "qualification_sha",
    "fe79e01769853eb3655bfd90e9f26e1be6e1dcdd",
    "HEAD == origin/development",
    "test_postgresql_collaborative_activity_read_port_contract",
    "test_postgresql_collaborative_activity_read_isolation",
    "test_postgresql_collaborative_activity_read_late_occurred_at_ordering",
    "passed: 3",
    "skipped: 0",
    "xfailed: 0",
    "failed: 0",
    "collaborative_activity_read_port_contract",
    "authorization is upstream of provider",
    "READ COMMITTED",
    "MP-6E — CLOSED / RECERTIFIED",
    "BLOCKING FINDINGS: NONE",
)


def _annotation_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Subscript):
        base = _annotation_name(node.value)
        if isinstance(node.slice, ast.Tuple):
            parts = ", ".join(_annotation_name(elt) for elt in node.slice.elts)
            return f"{base}[{parts}]"
        return f"{base}[{_annotation_name(node.slice)}]"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return f"{_annotation_name(node.left)} | {_annotation_name(node.right)}"
    return ast.unparse(node)


def test_mp6e_c1_composition_read_policy_parameter_uses_contract_protocol() -> None:
    tree = ast.parse(_COMPOSITION.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_collaborative_activity_read_service":
            for arg in node.args.kwonlyargs:
                if arg.arg == "read_authorization_policy":
                    assert arg.annotation is not None
                    assert (
                        _annotation_name(arg.annotation)
                        == "CollaborativeActivityReadAuthorizationPolicy | None"
                    )
                    return
    pytest.fail("build_collaborative_activity_read_service missing read_authorization_policy")


def test_mp6e_c1_custom_policy_injected_through_composition_root(tmp_path: Path) -> None:
    class DenyPolicy:
        policy_id = "composition-deny"

        def evaluate(
            self,
            policy_input: CollaborativeActivityReadAuthorizationPolicyInput,
        ) -> object:
            return fail_closed_collaborative_activity_read_decision(
                policy_id=self.policy_id,
                denial_reason=CollaborativeActivityReadDenialReason.POLICY_AMBIGUITY,
            )

    spy = MagicMock(spec=CollaborativeActivityReadPort)
    fixture = _seed_authority()
    service = build_collaborative_activity_read_service(
        authority_resolver=fixture.resolver,
        read_port=spy,
        read_authorization_policy=DenyPolicy(),
    )
    with pytest.raises(CollaborativeActivityReadDenied):
        service.read_page(_read_request(fixture))
    spy.query.assert_not_called()


def test_mp6e_c1_default_policy_fallback_via_composition_root(tmp_path: Path) -> None:
    fixture = _seed_authority()
    read_port = build_sqlite_collaborative_activity_read_store(str(tmp_path / "c1.sqlite"))
    service = build_collaborative_activity_read_service(
        authority_resolver=fixture.resolver,
        read_port=read_port,
    )
    page = service.read_page(_read_request(fixture))
    assert page.activities == ()


def test_mp6e_c1_read_store_sql_uses_keyset_without_offset() -> None:
    source = _READ_STORE.read_text(encoding="utf-8-sig")
    assert " ORDER BY append_position ASC" in source
    assert "append_position > " in source
    assert "tenant_id" in source
    assert "workspace_id" in source
    assert " OFFSET " not in source.upper()


def test_mp6e_c1_bounded_mp6_docs_have_no_mojibake() -> None:
    for path in _BOUNDED_MP6_DOCS:
        text = path.read_text(encoding="utf-8-sig")
        assert not MOJIBAKE_PATTERN.search(text), (
            f"{path.relative_to(_REPO_ROOT)}: mojibake remains"
        )


def test_mp6e_c1_canonical_roadmap_has_single_mp6d_row() -> None:
    multiplayer_arch = (
        _REPO_ROOT / "docs" / "project" / "capabilities" / "architecture" / "MULTIPLAYER_AI.md"
    )
    text = multiplayer_arch.read_text(encoding="utf-8-sig")
    mp6_start = text.find("| Slice | Purpose | Status |")
    assert mp6_start >= 0
    mp6_end = text.find("### MP-7", mp6_start)
    table = text[mp6_start:mp6_end] if mp6_end > mp6_start else text[mp6_start : mp6_start + 2500]
    matches = _MP6D_ROW.findall(table)
    assert len(matches) == 1, f"expected one MP-6D roadmap row, found {len(matches)}"


def test_mp6e_c1_mp6e_closed_requires_qualification_artifact() -> None:
    assert _MP6E_C1_PG_QUAL_EVIDENCE.is_file(), (
        "MP-6E CLOSED requires PostgreSQL read qualification artifact"
    )
    body = _MP6E_C1_PG_QUAL_EVIDENCE.read_text(encoding="utf-8-sig")
    missing = [marker for marker in _MP6E_C1_Q1_REQUIRED_MARKERS if marker not in body]
    assert not missing, f"MP-6E-C1-Q1 evidence: missing markers: {missing}"
    assert "qualification_execution_base_sha" not in body, (
        "MP-6E-C1-Q1 evidence: ambiguous qualification_execution_base_sha must not remain"
    )


def test_mp6e_c1_adr_repair_helper_documents_sequences() -> None:
    sample = "scope â€” arrow â†’ section Â§ middle Â·"
    repaired = repair_mojibake(sample)
    assert "â" not in repaired
    assert "Â" not in repaired
    assert "—" in repaired
    assert "→" in repaired
    assert "§" in repaired
    assert "·" in repaired
