# © Artur Czarnecki. All rights reserved.

"""MP-7D — cross-slice architecture gates for Multiplayer Tier-3 consumability.

Aggregates predecessor evidence/status consistency and regression invariants.
Does not duplicate full MP-7A/B/C assertion suites.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MP7D = Path(__file__).resolve().parent
_MP7C = _MP7D.parent / "mp7c"
_MP7B = _MP7D.parent / "mp7b"
_LKW_APP = _REPO_ROOT / "applications" / "local_workspace_application"

_CERT_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "MP-7D_FINAL_REFERENCE_CONSUMER_BOUNDARY_ENTERPRISE_CERTIFICATION.md"
)

_PREDECESSOR_DOCS = {
    "MP-7A": _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "MP-7A_LKW_MULTIPLAYER_ADOPTION_ARCHITECTURE_GATE.md",
    "MP-7B": _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "MP-7B_TIER3_MULTIPLAYER_CONSUMER_BOUNDARY_QUALIFICATION.md",
    "MP-7C": _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "MP-7C_TIER3_HOST_COMPOSITION_BOUNDARY_E2E_QUALIFICATION.md",
}

_CONTRACT_EVALUATOR = (
    _REPO_ROOT / "intergrax" / "contracts" / "meaningful_side_effect_policy.py"
)
_CONTRACT_AUTH = (
    _REPO_ROOT / "intergrax" / "contracts" / "meaningful_side_effect_authorization.py"
)
_WIRING = (
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "harness_meaningful_side_effect_authorization_wiring.py"
)
_CONSUMER = _MP7B / "consumer.py"

_STATUS_DOCS = {
    "multiplayer_architecture": _REPO_ROOT
    / "docs"
    / "project"
    / "capabilities"
    / "architecture"
    / "MULTIPLAYER_AI.md",
    "multiplayer_plan": _REPO_ROOT
    / "docs"
    / "project"
    / "capabilities"
    / "plan"
    / "MULTIPLAYER_AI.md",
    "lkw_architecture": _LKW_APP / "docs" / "ARCHITECTURE.md",
    "lkw_plan": _LKW_APP / "docs" / "IMPLEMENTATION_PLAN.md",
}

_LKW_COMPOSITION_ALLOWLIST: frozenset[str] = frozenset()

_FORBIDDEN_PRIVATE_PREFIXES = ("intergrax.collaborative_work",)
_FORBIDDEN_PROVIDER_TOKENS = (
    "PostgreSQLCollaborativeWorkStore",
    "SQLiteCollaborativeWorkStore",
    "InMemoryCollaborativeWorkStore",
)

# Final predecessor binder required by MP-7D entry criteria.
_R1_BINDER_SHA = "e600f624bc79364195e92a8466f17854f22db0da"


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _imports_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


def _lkw_production_python_files() -> list[Path]:
    if not _LKW_APP.is_dir():
        return []
    skip_parts = {"tests", "docker", "__pycache__", ".proof_docs", "build"}
    return [
        path
        for path in _LKW_APP.rglob("*.py")
        if not any(part in skip_parts for part in path.parts)
    ]


def _git_is_ancestor(commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def test_mp7d_certification_doc_closed() -> None:
    text = _CERT_DOC.read_text(encoding="utf-8-sig")
    assert "ENTERPRISE CERTIFIED / CLOSED" in text or (
        "FINAL REFERENCE-CONSUMER BOUNDARY ENTERPRISE CERTIFICATION PASSED" in text
    )
    assert "MP-7 — ENTERPRISE BOUNDARY CERTIFIED / CLOSED" in text
    assert "Tier-3 consumability boundary" in text or "Tier-3 consumability" in text.lower()
    assert "not LKW product adoption" in text.lower() or (
        "not full product adoption" in text.lower()
    )
    assert "BLOCKING ARCHITECTURE GAPS: NONE" in text
    assert "BLOCKING FINDINGS: NONE" in text
    assert "PRODUCTION CHANGES" in text and "NONE" in text
    assert "PLATFORM OPERATES ON CONTRACTS" in text or (
        "contracts, not implementations" in text.lower()
    )
    assert "independent" in text.lower() and "audit" in text.lower()
    assert "AUDITED_SHA" in text
    assert "MeaningfulSideEffectPolicyEvaluator" in text
    assert "intergrax.contracts" in text


def test_predecessor_qualification_docs_exist_and_closed() -> None:
    expected_markers = {
        "MP-7A": ("CLOSED / CERTIFIED",),
        "MP-7B": ("TIER-3 MULTIPLAYER CONSUMER BOUNDARY QUALIFIED / CLOSED",),
        "MP-7C": (
            "TIER-3 HOST COMPOSITION & BOUNDARY E2E QUALIFIED / CLOSED",
            "MP-7C-C1-R1",
            "CLOSED / CERTIFIED",
        ),
    }
    for name, path in _PREDECESSOR_DOCS.items():
        assert path.is_file(), f"missing predecessor evidence: {_rel(path)}"
        text = path.read_text(encoding="utf-8-sig")
        for marker in expected_markers[name]:
            assert marker in text, f"{name}: missing {marker!r}"


def test_r1_binder_is_ancestor_of_head() -> None:
    assert _git_is_ancestor(_R1_BINDER_SHA), (
        f"R1 binder {_R1_BINDER_SHA} must be ancestor of HEAD"
    )


def test_mp7d_status_markers_in_ssot_docs() -> None:
    required = (
        "MP-7D",
        "ENTERPRISE BOUNDARY CERTIFIED",
        "MP-7A",
        "MP-7B",
        "MP-7C",
    )
    for name, path in _STATUS_DOCS.items():
        text = path.read_text(encoding="utf-8-sig")
        missing = [marker for marker in required if marker not in text]
        assert not missing, f"{name}: missing markers: {missing}"
        assert "MP-7 — IN PROGRESS" not in text, (
            f"{name}: MP-7 must not remain IN PROGRESS after MP-7D"
        )


def test_canonical_evaluator_contract_in_neutral_layer() -> None:
    from intergrax.contracts.meaningful_side_effect_policy import (
        MeaningfulSideEffectPolicyEvaluator,
    )

    assert MeaningfulSideEffectPolicyEvaluator.__module__ == (
        "intergrax.contracts.meaningful_side_effect_policy"
    )
    source = _CONTRACT_EVALUATOR.read_text(encoding="utf-8")
    assert "class MeaningfulSideEffectPolicyEvaluator" in source
    assert "intergrax.collaborative_work" not in source
    assert "applications." not in source
    assert "intergrax.runtime." not in source


def test_exactly_one_production_evaluator_definition() -> None:
    definitions: list[str] = []
    for path in (_REPO_ROOT / "intergrax").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if "class MeaningfulSideEffectPolicyEvaluator" in path.read_text(encoding="utf-8"):
            definitions.append(_rel(path))
    assert definitions == ["intergrax/contracts/meaningful_side_effect_policy.py"]


def test_no_old_evaluator_import_from_enforcement_gate_in_production() -> None:
    pattern = re.compile(
        r"from\s+intergrax\.collaborative_work\.enforcement_gate\s+import\s+"
        r".*MeaningfulSideEffectPolicyEvaluator"
    )
    violations: list[str] = []
    for root_name in ("intergrax", "applications"):
        root = _REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts or "docker" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                violations.append(_rel(path))
    assert not violations, "old evaluator import path:\n" + "\n".join(violations)


def test_host_composition_imports_public_contracts() -> None:
    mods = _imports_in_file(_WIRING)
    assert "intergrax.contracts.meaningful_side_effect_authorization" in mods
    assert "intergrax.contracts.meaningful_side_effect_policy" in mods
    source = _WIRING.read_text(encoding="utf-8")
    assert "MeaningfulSideEffectAuthorizationPort" in source
    assert "MeaningfulSideEffectPolicyEvaluator" in source
    assert "resolve_harness_host_meaningful_side_effect_authorization_wiring" in source


def test_consumer_remains_public_contracts_only() -> None:
    mods = _imports_in_file(_CONSUMER)
    for mod in mods:
        assert not mod.startswith("intergrax.collaborative_work"), mod
        assert "harness_meaningful_side_effect_authorization_wiring" not in mod
        assert "repository" not in mod
        assert "persistence_provider" not in mod
    assert any(m.startswith("intergrax.contracts.") for m in mods)
    source = _CONSUMER.read_text(encoding="utf-8")
    assert "getattr(" not in source
    assert "hasattr(" not in source
    assert "setattr(" not in source
    assert "isinstance(" not in source
    assert "# type: ignore" not in source
    assert "_inner" not in source
    assert "authorization: Any" not in source
    assert "authorization: object" not in source


def test_no_semantic_monkeypatch_in_mp7_qualification_tree() -> None:
    """Forbid semantic engine replacement; allow doc/gate string assertions."""
    tree = _MP7D.parent
    hits: list[str] = []
    self_path = Path(__file__).resolve()
    for path in tree.rglob("*.py"):
        if path.resolve() == self_path:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            if "patch(RuntimePolicyEngine)" not in line and (
                "harness_meaningful_side_effect_authorization_wiring.RuntimePolicyEngine"
                not in line
            ):
                continue
            # Architecture gates assert the forbidden string is absent from fixtures.
            window = "\n".join(lines[max(0, lineno - 4) : lineno])
            if "assert" in window:
                continue
            hits.append(f"{_rel(path)}:{lineno}")
    assert not hits, "semantic RuntimePolicyEngine monkeypatch:\n" + "\n".join(hits)


def test_no_duplicate_consumer_or_host_composition_in_mp7d() -> None:
    assert list(_MP7D.glob("**/consumer.py")) == []
    assert list(_MP7D.glob("**/host_composition.py")) == []
    assert list(_MP7D.glob("**/custom_ports.py")) == []
    assert list(_MP7D.glob("**/composition.py")) == []
    # Predecessor fixtures remain the reuse points.
    assert _CONSUMER.is_file()
    assert (_MP7C / "host_composition.py").is_file()


def test_lkw_production_forbids_private_collaborative_work_imports() -> None:
    violations: list[str] = []
    for path in _lkw_production_python_files():
        rel = _rel(path)
        if rel in _LKW_COMPOSITION_ALLOWLIST:
            continue
        for mod in _imports_in_file(path):
            for prefix in _FORBIDDEN_PRIVATE_PREFIXES:
                if mod == prefix or mod.startswith(prefix + "."):
                    violations.append(f"{rel}: import {mod}")
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_PROVIDER_TOKENS:
            if re.search(rf"\b{re.escape(token)}\b", source):
                violations.append(f"{rel}: token {token}")
    assert not violations, "LKW private Multiplayer leakage:\n" + "\n".join(violations)


def test_contracts_layer_has_no_lkw_or_application_package_types() -> None:
    contracts = _REPO_ROOT / "intergrax" / "contracts"
    for path in contracts.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\bclass Lkw[A-Z]", text), _rel(path)
        assert "from applications." not in text
        assert "import applications." not in text


def test_authorization_port_contract_exists() -> None:
    source = _CONTRACT_AUTH.read_text(encoding="utf-8")
    assert "class MeaningfulSideEffectAuthorizationPort" in source
    assert "intergrax.collaborative_work" not in source
