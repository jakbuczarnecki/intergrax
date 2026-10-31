# © Artur Czarnecki. All rights reserved.

"""B6-F10 — normative ECAP C1–C5 proof bindings (single SSOT matrix)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from tests.qualification.ebh_5._proof_validation import pytest_test_function_exists
from tests.qualification.ebh_6._family_catalog import EBH6_APPLICABLE_FAMILY_COUNT
from tests.qualification.ebh_7._constants import (
    EBH7_AUTHORIZED_BASELINE_SHA,
    EBH7_B5_GHA_RUN_ID,
    EBH7_B5_REPLAY_HEAD_SHA,
    EBH7_TEST_EXECUTION_SHA,
)

_REPO = Path(__file__).resolve().parents[3]
_EVIDENCE_PUBLICATION_SHA: Final[str] = "96069c09c2ca5a7a3a9586db336716c9430ca1d7"
_MATRIX_MD = _REPO / "docs/project/maintainers/qualification/EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md"
_ECAP_REPORT = _REPO / "docs/project/maintainers/qualification/EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md"


class F10Verdict(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class F10Result(StrEnum):
    PASS = "PASS"
    NA_WITH_EVIDENCE = "N/A_WITH_EVIDENCE"


@dataclass(frozen=True, slots=True)
class F10Row:
    ecap_gate: str
    normative_requirement: str
    scope: str
    authority: str
    owner: str
    evidence: str
    execution: str
    result: F10Result
    drift: str
    gap: str
    verdict: F10Verdict


def _head_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()


def _node_ok(node: str) -> bool:
    if pytest_test_function_exists(node):
        return True
    if ":" in node and node.startswith("tests."):
        mod, func = node.rsplit(":", 1)
        file_rel = mod.replace(".", "/") + ".py"
        return pytest_test_function_exists(f"{file_rel}::{func}")
    return False


def f10_ecap_normative_rows() -> tuple[F10Row, ...]:
    head = _head_sha()
    return (
        F10Row(
            ecap_gate="C1",
            normative_requirement=(
                "Closed-world inventory Tier-0..3 and all architecture families "
                "(ECAP-§2.C; roadmap §3.1 minimum coverage)"
            ),
            scope=f"Tier-0..3 + {EBH6_APPLICABLE_FAMILY_COUNT} EBH-6 family themes; EBH-7 enterprise family matrix",
            authority="ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md §2.C",
            owner="EBH-6 family catalog + EBH-7 mechanism registry",
            evidence=(
                "tests/qualification/ebh_6/test_ebh_6_architecture_family_matrix_integrity.py::"
                "test_ebh6_family_row_count_closed_world; "
                "tests/qualification/ebh_7/test_ebh_7_mechanical_exit_gates.py::"
                "test_ebh7_structural_exit_traceability_and_mechanisms; "
                "EBH_7_ENTERPRISE_FAMILY_COVERAGE_MATRIX.md"
            ),
            execution=(
                f"uv run pytest tests/qualification/ebh_6/test_ebh_6_architecture_family_matrix_integrity.py "
                f"tests/qualification/ebh_7/test_ebh_7_mechanical_exit_gates.py -q @ replay HEAD; "
                f"TEST_EXECUTION_SHA={EBH7_TEST_EXECUTION_SHA}"
            ),
            result=F10Result.PASS,
            drift="Documentation + qualification infra @ replay HEAD; certified code @ TEST_EXECUTION_SHA",
            gap="",
            verdict=F10Verdict.READY,
        ),
        F10Row(
            ecap_gate="C2",
            normative_requirement=(
                "Cross-domain dependency and duplicate authority scan (exactly-one semantic owner; no shadow registry)"
            ),
            scope="DUP-X orchestration backend + NPSC boundary subset; tier import boundaries",
            authority="ECAP-§2.C C2; DUP-X qualification",
            owner="tests/qualification/dup_x + EBH-7 W4 gates",
            evidence=(
                "tests/qualification/ebh_7/test_ebh_7_w4_gov_dup_gates.py; "
                "tests/qualification/dup_x/test_dup_x_b4_npsc_orchestration_backend_negative_boundary.py; "
                "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py"
            ),
            execution=(
                "uv run pytest tests/qualification/ebh_7/test_ebh_7_w4_gov_dup_gates.py -q; "
                "parent ROADMAP-REPLAY-X closure doc @ EBH_7 entry gate"
            ),
            result=F10Result.PASS,
            drift="No new duplicate authority paths vs AUTHORIZED_BASELINE without qualification-only deltas",
            gap="",
            verdict=F10Verdict.READY,
        ),
        F10Row(
            ecap_gate="C3",
            normative_requirement=(
                "Full VBN taxonomy (roadmap §3.0.8 domains 1–31 + discoveries): "
                "inventory, owner, replaceability disposition, PROHIBITED handling"
            ),
            scope="EBH-6 hardcoding register + CONFIG-X 54 concerns + INT2E bundles + concern catalog",
            authority="ECAP-§2.C C3; VENDOR_BACKEND_NEUTRALITY evidence",
            owner="tests/qualification/ebh_6 hardcoding register + config_x reconciliation",
            evidence=(
                "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::"
                "test_ebh6_prohibited_unresolved_zero; "
                "test_ebh6_findings_unclassified_zero; "
                "tests/qualification/ebh_6/test_ebh_6_config_x_head_reconciliation.py; "
                "tests/qualification/ebh_7/test_ebh_7_int2e_bundle_regression.py"
            ),
            execution=(
                f"Executed on TEST_EXECUTION_SHA={EBH7_TEST_EXECUTION_SHA} via MANIFEST L3 profile; "
                f"register SSOT @ HEAD {head[:12]}…"
            ),
            result=F10Result.PASS,
            drift="CONFIG-X historical pass preserved; HEAD reconciliation documented in EBH-6 audit",
            gap="",
            verdict=F10Verdict.READY,
        ),
        F10Row(
            ecap_gate="C4",
            normative_requirement=(
                "Enterprise regression / drift vs last accepted baseline SHA "
                "(classify documentation-only vs semantic deltas)"
            ),
            scope=f"Accepted baseline {EBH7_AUTHORIZED_BASELINE_SHA[:12]}… → TEST_EXECUTION → replay HEAD",
            authority="ECAP-§2.C C4; EBH_7_SHA_LADDER.md",
            owner="EBH-6 config_x head reconciliation + EBH-7 entry / P03 integrity",
            evidence=(
                "tests/qualification/ebh_6/test_ebh_6_config_x_head_reconciliation.py; "
                "tests/qualification/ebh_7/test_ebh_7_entry_gates.py::test_ebh7_entry_parent_replay_closed_docs; "
                "tests/qualification/ebh_7/_ebh7_p03_evidence_integrity.py (manifest roles); "
                "B5 GHA run 41214574614 @ replay HEAD"
            ),
            execution=(
                f"Baseline={EBH7_AUTHORIZED_BASELINE_SHA}; TEST={EBH7_TEST_EXECUTION_SHA}; "
                f"B5 replay={EBH7_B5_REPLAY_HEAD_SHA}; F10/F11 gates @ {head[:12]}…"
            ),
            result=F10Result.PASS,
            drift="Qualification/docs/CI allowlisted; no uncatalogued production semantic drift @ TEST",
            gap="",
            verdict=F10Verdict.READY,
        ),
        F10Row(
            ecap_gate="C5",
            normative_requirement=(
                "Produce ECAP §14 report with platform scope banner, tier/family/VBN coverage, "
                "findings, blockers, non-claims (pre-audit — not independent B6 PASS)"
            ),
            scope="Platform EBH-7 closure audit @ current HEAD",
            authority="ECAP-§14 Mandatory Audit Report Template",
            owner="EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md + this matrix",
            evidence=(
                "docs/project/maintainers/qualification/EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md; "
                "EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md; "
                "tests/qualification/ebh_7/test_ebh_7_b6_f10_f11_corrective_gates.py"
            ),
            execution=(
                f"Report HEAD={head}; TEST={EBH7_TEST_EXECUTION_SHA}; EVIDENCE={_EVIDENCE_PUBLICATION_SHA[:12]}…"
            ),
            result=F10Result.PASS,
            drift="Report updated for B6-F10/F11; historical SHA roles preserved in ladder",
            gap="",
            verdict=F10Verdict.READY,
        ),
    )


def render_f10_matrix_markdown() -> str:
    head = _head_sha()
    lines = [
        "# EBH-7 — B6-F10 ECAP normative matrix (C1–C5)",
        "",
        "**Status:** READY FOR INDEPENDENT B6 RE-AUDIT (agent — not B6 PASS).",
        "",
        f"**Matrix SSOT @ HEAD:** `{head}` · **TEST_EXECUTION_SHA:** `{EBH7_TEST_EXECUTION_SHA}` · "
        f"**B5 canonical GHA:** `{EBH7_B5_GHA_RUN_ID}` @ `{EBH7_B5_REPLAY_HEAD_SHA}`",
        "",
        "| ECAP_GATE | NORMATIVE_REQUIREMENT | SCOPE | AUTHORITY | OWNER | "
        "EVIDENCE | EXECUTION | RESULT | DRIFT | GAP | VERDICT |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in f10_ecap_normative_rows():
        lines.append(
            "| "
            + " | ".join(
                (
                    row.ecap_gate,
                    row.normative_requirement.replace("|", "/")[:120] + "…"
                    if len(row.normative_requirement) > 120
                    else row.normative_requirement.replace("|", "/"),
                    row.scope.replace("|", "/"),
                    row.authority.replace("|", "/"),
                    row.owner.replace("|", "/"),
                    row.evidence.replace("|", "/")[:100] + "…"
                    if len(row.evidence) > 100
                    else row.evidence.replace("|", "/"),
                    row.execution.replace("|", "/")[:80] + "…"
                    if len(row.execution) > 80
                    else row.execution.replace("|", "/"),
                    row.result.value,
                    row.drift.replace("|", "/")[:60] + "…" if len(row.drift) > 60 else row.drift.replace("|", "/"),
                    row.gap or "—",
                    row.verdict.value,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "**Non-claims:** B6 PASS · OD-EBH7-08 · EBH-7 CLOSED · global FRZ PASS.",
            "",
        ]
    )
    return "\n".join(lines)


def assert_ebh7_f10_ecap_normative_ready() -> None:
    rows = f10_ecap_normative_rows()
    assert len(rows) == 5, "ECAP C1–C5 require exactly five normative rows"
    errors: list[str] = []
    for row in rows:
        if row.verdict is not F10Verdict.READY:
            errors.append(f"{row.ecap_gate}: verdict={row.verdict}")
        if row.gap.strip():
            errors.append(f"{row.ecap_gate}: gap={row.gap}")
        for chunk in row.evidence.split(";"):
            chunk = chunk.strip()
            if chunk.endswith(".md") or (chunk and chunk.split()[0].startswith("EBH_7_")):
                rel = chunk.split()[0]
                if not rel.startswith("docs/"):
                    rel = f"docs/project/maintainers/qualification/{rel}"
                if not (_REPO / rel).is_file():
                    errors.append(f"{row.ecap_gate}: missing doc {rel}")
            elif "::" in chunk:
                if not _node_ok(chunk):
                    errors.append(f"{row.ecap_gate}: missing node {chunk}")
    if not _ECAP_REPORT.is_file():
        errors.append("ECAP §14 report missing")
    else:
        text = _ECAP_REPORT.read_text(encoding="utf-8")
        for token in (
            "Platform scope",
            "ECAP-C1",
            "ECAP-C2",
            "ECAP-C3",
            "ECAP-C4",
            "ECAP-C5",
            "TEST_EXECUTION_SHA",
            "Non-claims",
        ):
            if token not in text:
                errors.append(f"ECAP report missing {token}")
    if _MATRIX_MD.is_file():
        md = _MATRIX_MD.read_text(encoding="utf-8")
        if "READY FOR INDEPENDENT B6 RE-AUDIT" not in md:
            errors.append("F10 matrix markdown stale status banner")
    assert not errors, "; ".join(errors[:8])


def sync_f10_matrix_markdown() -> None:
    _MATRIX_MD.write_text(render_f10_matrix_markdown(), encoding="utf-8")
