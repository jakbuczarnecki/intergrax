# © Artur Czarnecki. All rights reserved.

"""MP-FINAL-4 — cross-cutting enterprise recertification architecture gates."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACTS = _REPO_ROOT / "intergrax" / "contracts"
_CW_PROD = _REPO_ROOT / "intergrax" / "collaborative_work"
_QUAL = _REPO_ROOT / "docs" / "project" / "maintainers" / "qualification"

_MP_FINAL4_EVIDENCE = (
    _QUAL / "MP-FINAL-4_MULTIPLAYER_ENTERPRISE_CORE_FINAL_RECERTIFICATION.md"
)
_MP_FINAL4_R1_EVIDENCE = (
    _QUAL / "MP-FINAL-4-R1_FINAL_CERTIFICATION_EVIDENCE_CHAIN_INTEGRITY_CORRECTION.md"
)
_MP_FINAL4_R1_B1_BINDER = _QUAL / "MP-FINAL-4-R1-B1_FINAL_R1_EVIDENCE_BINDER.md"

_R1_QUALIFICATION_SHA = "17be264454ffaf1de1a3c9af2f0928a6e9647a43"
_R1_EVIDENCE_SHA = "fcf7e869e1d3552b29cacb90dcb8fbcf904546c9"

_PREDECESSOR_EVIDENCE = {
    "MP-7D": _QUAL / "MP-7D_FINAL_REFERENCE_CONSUMER_BOUNDARY_ENTERPRISE_CERTIFICATION.md",
    "MP-FINAL-1-R1": _QUAL
    / "MP-FINAL-1-R1_VISUAL_COMPOSITION_FLOW_EVIDENCE_PROVENANCE_CORRECTION.md",
    "MP-FINAL-2-C1": _QUAL
    / "MP-FINAL-2-C1_OPERATOR_FACING_DIAGNOSTICS_CONTRACT_BOUNDARY_QUALIFICATION.md",
    "MP-FINAL-3": _QUAL / "MP-FINAL-3_CAPABILITY_WIDE_BACKEND_E2E_CERTIFICATION.md",
}

_PREDECESSOR_SHAS = (
    "ab0c21b44bc4ee7c4faee074f31021495de475cf",  # MP-7D AUDITED_SHA
    "92663e44e1ad4d6349ecb710a565cbb6a484d676",  # MP-7D CERTIFICATION_SHA
    "133941393ad95bcfcf8e383a0fbf4068296296ae",  # MP-7D EVIDENCE_SHA
    "dfd2c9a1f67a8ab798765ed6f44a77f266bb6b57",  # MP-7D binder/closure
    "fd805578f4ab924b350cc6f19160ce702e88cfa7",  # MP-FINAL-1 base
    "8ada8d72dd3d78f89048aaef177488f255fb3a64",  # MP-FINAL-1-R1 correction/evidence
    "d1f0631cf7ebb5546e99099de2810e146bc9d9b0",  # MP-FINAL-2-C1 CORRECTION_SHA
    "9c304e70ace27b4f477269d425e6516090d16140",  # MP-FINAL-2-C1 QUALIFICATION_SHA
    "ea6c4ca7376021b9250a810f6c238e388b44dfc9",  # MP-FINAL-2-C1 EVIDENCE_SHA
    "0eda5cdd4bc6d6f723a754468f5e34d14bfbb443",  # MP-FINAL-2-C1 BINDER_SHA
    "559af7bcbe5bda320dae490316bd8f4b0786d14d",  # MP-FINAL-3 QUALIFICATION_SHA
    "61c9ae04f65b2052d9ce040986182af7ee1f4ce8",  # MP-FINAL-3 EVIDENCE_SHA
    "76c64c6f9cadc52a69cdea82e560975afd3230e3",  # MP-FINAL-3 BINDER_SHA
    "acdfc1d2b3f99244ee2776d3f286541fe3d43c93",  # MP-FINAL-4 QUALIFICATION_SHA
    "7dcf7a5cd10288419cf23b476bad6e81dc3373d9",  # MP-FINAL-4 EVIDENCE_SHA
    "d9ba436ac9a7ffa369507555e626c4dab2407393",  # MP-FINAL-4 BINDER_SHA
)

_CANONICAL_PROVENANCE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("MP-7D", "AUDITED_SHA", "ab0c21b44bc4ee7c4faee074f31021495de475cf"),
    ("MP-7D", "CERTIFICATION_SHA", "92663e44e1ad4d6349ecb710a565cbb6a484d676"),
    ("MP-7D", "EVIDENCE_SHA", "133941393ad95bcfcf8e383a0fbf4068296296ae"),
    ("MP-7D", "BINDER_SHA", "dfd2c9a1f67a8ab798765ed6f44a77f266bb6b57"),
    ("MP-FINAL-1-R1", "BASE_MP_FINAL_1_SHA", "fd805578f4ab924b350cc6f19160ce702e88cfa7"),
    ("MP-FINAL-1-R1", "CORRECTION_SHA", "8ada8d72dd3d78f89048aaef177488f255fb3a64"),
    ("MP-FINAL-2-C1", "CORRECTION_SHA", "d1f0631cf7ebb5546e99099de2810e146bc9d9b0"),
    ("MP-FINAL-2-C1", "QUALIFICATION_SHA", "9c304e70ace27b4f477269d425e6516090d16140"),
    ("MP-FINAL-2-C1", "EVIDENCE_SHA", "ea6c4ca7376021b9250a810f6c238e388b44dfc9"),
    ("MP-FINAL-2-C1", "BINDER_SHA", "0eda5cdd4bc6d6f723a754468f5e34d14bfbb443"),
    ("MP-FINAL-3", "QUALIFICATION_SHA", "559af7bcbe5bda320dae490316bd8f4b0786d14d"),
    ("MP-FINAL-3", "EVIDENCE_SHA", "61c9ae04f65b2052d9ce040986182af7ee1f4ce8"),
    ("MP-FINAL-3", "BINDER_SHA", "76c64c6f9cadc52a69cdea82e560975afd3230e3"),
    ("MP-FINAL-4", "QUALIFICATION_SHA", "acdfc1d2b3f99244ee2776d3f286541fe3d43c93"),
    ("MP-FINAL-4", "EVIDENCE_SHA", "7dcf7a5cd10288419cf23b476bad6e81dc3373d9"),
    ("MP-FINAL-4", "BINDER_SHA", "d9ba436ac9a7ffa369507555e626c4dab2407393"),
)

_FORBIDDEN_CONTRACT_IMPORT_PREFIXES = (
    "intergrax.runtime",
    "intergrax.applications",
    "intergrax.collaborative_work",
    "applications.",
)

_FORBIDDEN_ORCHESTRATORS = (
    "MultiplayerOrchestrator",
    "MultiplayerWorkflowEngine",
    "MultiplayerE2EOrchestrator",
)

_SSOT_DOCS = (
    _REPO_ROOT / "docs" / "project" / "capabilities" / "architecture" / "MULTIPLAYER_AI.md",
    _REPO_ROOT / "docs" / "project" / "capabilities" / "plan" / "MULTIPLAYER_AI.md",
)


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _git_is_ancestor(commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _git_object_exists(sha: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _imports_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


def test_mp_final4_evidence_closed_and_independent_audit() -> None:
    assert _MP_FINAL4_EVIDENCE.is_file()
    text = _MP_FINAL4_EVIDENCE.read_text(encoding="utf-8-sig")
    assert "MP-FINAL-4 — FINAL ENTERPRISE CORE RECERTIFICATION PASSED / CLOSED" in text
    assert "MULTIPLAYER ENTERPRISE CORE — FINAL CERTIFIED / CLOSED" in text
    assert "MP-FINAL-4 TASK PRODUCTION CHANGES = NONE" in text
    assert "BLOCKING ARCHITECTURE FINDINGS: NONE" in text
    assert "independent" in text.lower() and "audit" in text.lower()
    assert "START_HEAD" in text
    assert "QUALIFICATION_SHA" in text
    assert "EVIDENCE_SHA" in text
    assert "CURRENT_HEAD" not in text


def test_predecessor_evidence_exists_and_closed() -> None:
    markers = {
        "MP-7D": ("CLOSED / ENTERPRISE CERTIFIED", "AUDITED_SHA"),
        "MP-FINAL-1-R1": ("R1", "CLOSED"),
        "MP-FINAL-2-C1": ("CLOSED / CERTIFIED", "intergrax.contracts.diagnostics"),
        "MP-FINAL-3": ("CAPABILITY-WIDE BACKEND E2E CERTIFIED / CLOSED",),
    }
    for name, path in _PREDECESSOR_EVIDENCE.items():
        assert path.is_file(), f"missing {_rel(path)}"
        text = path.read_text(encoding="utf-8-sig")
        for marker in markers[name]:
            assert marker in text, f"{name}: missing {marker!r}"


def test_predecessor_shas_are_ancestors_and_fetchable() -> None:
    for sha in _PREDECESSOR_SHAS:
        assert _git_object_exists(sha), f"missing git object {sha}"
        assert _git_is_ancestor(sha), f"{sha} must be ancestor of HEAD"


def test_mp_final4_evidence_canonical_provenance_chain() -> None:
    text = _MP_FINAL4_EVIDENCE.read_text(encoding="utf-8-sig")
    assert "Canonical final provenance chain" in text
    assert "BINDER_SHA        = d9ba436ac9a7ffa369507555e626c4dab2407393" in text
    for slice_name, role_label, sha in _CANONICAL_PROVENANCE_ROWS:
        assert sha in text, f"missing SHA {sha} for {slice_name}"
        assert role_label in text, f"missing role label {role_label} for {slice_name}"
        row = re.search(
            rf"\|\s*{re.escape(slice_name)}\s*\|[^\n]*{re.escape(sha)}",
            text,
        )
        assert row is not None, f"{slice_name} row must bind SHA {sha} in provenance table"


def test_mp_final4_r1_evidence_closed_and_no_mutable_head() -> None:
    assert _MP_FINAL4_R1_EVIDENCE.is_file()
    text = _MP_FINAL4_R1_EVIDENCE.read_text(encoding="utf-8-sig")
    assert (
        "MP-FINAL-4-R1 — FINAL CERTIFICATION EVIDENCE CHAIN INTEGRITY CORRECTION CLOSED / CERTIFIED"
        in text
    )
    assert "MP-FINAL-4-R1 TASK PRODUCTION CHANGES = NONE" in text
    assert "CURRENT_HEAD =" not in text
    assert "CURRENT_HEAD_AT" not in text
    assert "BLOCKING PROVENANCE FINDINGS: NONE" in text


def test_mp_final4_r1_b1_binder_immutable_evidence_binding() -> None:
    assert _MP_FINAL4_R1_B1_BINDER.is_file()
    text = _MP_FINAL4_R1_B1_BINDER.read_text(encoding="utf-8-sig")
    assert (
        "MP-FINAL-4-R1-B1 — FINAL R1 EVIDENCE BINDER CLOSED / CERTIFIED" in text
    )
    assert _R1_QUALIFICATION_SHA in text
    assert _R1_EVIDENCE_SHA in text
    assert "This binder does not introduce new certification content." in text
    assert "CURRENT_HEAD" not in text
    assert "git log -1" not in text
    assert "BINDER_SHA        =" not in text


def test_contracts_layer_has_no_implementation_imports() -> None:
    violations: list[str] = []
    for path in _CONTRACTS.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for mod in _imports_in_file(path):
            for prefix in _FORBIDDEN_CONTRACT_IMPORT_PREFIXES:
                if mod == prefix or mod.startswith(prefix + "."):
                    violations.append(f"{_rel(path)}: {mod}")
    assert not violations, "illegal contract imports:\n" + "\n".join(violations)


def test_collaborative_work_production_has_no_mega_orchestrator() -> None:
    hits: list[str] = []
    for path in _CW_PROD.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for name in _FORBIDDEN_ORCHESTRATORS:
            if name in text:
                hits.append(f"{_rel(path)}: {name}")
    assert not hits, "forbidden orchestrator symbols:\n" + "\n".join(hits)


def test_collaborative_work_has_no_second_decision_repository() -> None:
    for path in _CW_PROD.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        assert "class DecisionStore" not in text, _rel(path)
        assert "InMemoryDecisionRepository" not in text, _rel(path)


def test_meaningful_side_effect_contracts_owned_in_neutral_layer() -> None:
    policy = _CONTRACTS / "meaningful_side_effect_policy.py"
    auth = _CONTRACTS / "meaningful_side_effect_authorization.py"
    for path in (policy, auth):
        source = path.read_text(encoding="utf-8")
        assert "intergrax.collaborative_work" not in source
        assert "intergrax.runtime." not in source
        assert "applications." not in source


def test_diagnostics_operator_contract_module_exists() -> None:
    op = _CONTRACTS / "diagnostics" / "functional_operator_projection.py"
    assert op.is_file()
    source = op.read_text(encoding="utf-8")
    assert "class FunctionalDiagnosticOperatorProjection" in source
    assert "intergrax.runtime" not in source


def test_ssot_mp_final4_closed_mp8_mp9_future() -> None:
    for path in _SSOT_DOCS:
        text = path.read_text(encoding="utf-8-sig")
        assert "MP-FINAL-4" in text
        assert "FINAL RECERTIFICATION PASSED / CLOSED" in text or (
            "FINAL CERTIFIED / CLOSED" in text and "MP-FINAL-4" in text
        )
        assert "MP-FINAL-3 — CLOSED / CERTIFIED" in text or "MP-FINAL-3 — CLOSED" in text
        assert "MP-8" in text and "PLANNED / NOT STARTED" in text
        assert "MP-9" in text
        assert "MP-FINAL-4 — Final Enterprise Core Recertification** (NEXT)" not in text
        stale = re.search(r"MP-FINAL-[123]\s+NEXT", text)
        assert stale is None, f"{_rel(path)}: stale NEXT marker {stale!r}"


def test_lkw_reference_consumer_not_multiplayer_owner() -> None:
    lkw_arch = (
        _REPO_ROOT / "applications" / "local_workspace_application" / "docs" / "ARCHITECTURE.md"
    )
    if not lkw_arch.is_file():
        pytest.skip("LKW architecture doc absent")
    text = lkw_arch.read_text(encoding="utf-8-sig").lower()
    assert "reference consumer" in text or "tier-3" in text
    assert "owner of multiplayer" not in text
    assert "owns collaborative persistence" not in text
