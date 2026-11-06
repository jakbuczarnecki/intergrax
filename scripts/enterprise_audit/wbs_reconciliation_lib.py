# © Artur Czarnecki. All rights reserved.

"""C075-S2 WBS reconciliation rules @ TEST_EXECUTION_SHA + B5 evidence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

TES = "89c76eb7b53a0a3ce6f75ab395cf4a2802638a91"
EVID = "300859537f12a1de7bf3d75fd685bc3a5c727183"
EVID_SHORT = EVID[:12]
PLAN_BASELINE_SHA = "7de6b41f637df1d6949985d9dc567602a074e230"
HEAD_L2_LOG = ".tmp/session/ent-audit-x-closure/b3-l1-l2-l3.log"
HANDOFF = f"b6_independent_handoff/qualification_at_{TES}"
B5_QUAL = f"{HANDOFF}/qual_wave_report.json"
B5_L3 = f"{HANDOFF}/l3-cli-full"
B5_OUTCOMES = f"{B5_L3}/enterprise_audit_outcomes.json"

TECHNICAL_PARTIAL_DISPOSITION = "ENT-AUDIT-X — TECHNICAL PARTIAL / EXACT BLOCKERS"

# Direct qualification @ TES (B5 qual_wave and/or FULL outcomes — not indirect inference alone).
QUALIFIED_TES: frozenset[str] = frozenset(
    {
        "C004",
        "C005",
        "C008",
        "C009",
        "C010",
        "C011",
        "C018",
        "C019",
        "C028",
        "C031",
        "C032",
        "C033",
        "C034",
        "C035",
        "C036",
        "C037",
        "C038",
        "C039",
        "C043",
        "C044",
        "C050",
        "C051",
        "C058",
        "C067",
        "C071",
        "C074",
    }
)

# Code complete; qual blocked on operator / external governance.
OPERATOR_OR_EXTERNAL: frozenset[str] = frozenset({"C001", "C002", "C003", "C042", "C065", "C057"})

# CI / launcher enforcement — qual requires green workflow @ HEAD.
CI_BLOCKED: frozenset[str] = frozenset({"C045", "C046", "C047", "C059", "C060", "C061", "C062", "C063", "C070", "C007"})

# Post-TES mechanical readiness @ HEAD (CI doc-only reuse + readiness gates).
HEAD_MECH_QUALIFIED: frozenset[str] = frozenset()

POST_TES_DELTA: frozenset[str] = frozenset()

NOT_STARTED: frozenset[str] = frozenset()

IMPLEMENTED_PARTIAL_DEFAULT: frozenset[str] = frozenset({"C054", "C066", "C075"})

HEAD_L2_QUALIFIED: frozenset[str] = frozenset(
    {
        "C006",
        "C012",
        "C013",
        "C014",
        "C015",
        "C016",
        "C017",
        "C020",
        "C021",
        "C022",
        "C023",
        "C024",
        "C025",
        "C026",
        "C027",
        "C029",
        "C030",
        "C040",
        "C041",
        "C048",
        "C049",
        "C052",
        "C053",
        "C055",
        "C056",
        "C064",
        "C068",
        "C069",
        "C072",
        "C073",
    }
)


def _git_head(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def refresh_mechanical_qualification(repo: Path) -> dict[str, Any] | None:
    """Run mechanical_readiness_check and derive HEAD qualification sets for reconcile."""
    global HEAD_MECH_QUALIFIED  # noqa: PLW0603
    script = repo / "scripts/enterprise_audit/mechanical_readiness_check.py"
    proc = subprocess.run([sys.executable, str(script)], cwd=repo, capture_output=True, text=True, check=False)
    report_path = repo / ".tmp/session/ent-audit-x-mechanical-readiness/mechanical_readiness_report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    head = _git_head(repo)
    if report.get("implementation_sha") != head:
        return report
    checks = {c["name"]: c["ok"] for c in report.get("checks", []) if isinstance(c, dict)}
    layers = report.get("layers", {})
    qualified: set[str] = set()
    if layers.get("ci_run_success_at_head"):
        qualified.update(CI_BLOCKED)
    if checks.get("evidence_integrity") and checks.get("wbs_ssot_validator_pass"):
        qualified.update({"C054", "C066"})
    if (
        checks.get("wbs_ssot_validator_pass")
        and checks.get("wbs_no_unjustified_technical_partial")
        and proc.returncode == 0
    ):
        qualified.add("C075")
    HEAD_MECH_QUALIFIED = frozenset(qualified)
    _refresh_gap_rc_from_mechanical(report, checks, layers, head)
    return report


def _refresh_gap_rc_from_mechanical(
    report: dict[str, Any],
    checks: dict[str, bool],
    layers: dict[str, bool],
    head: str,
) -> None:
    if layers.get("ci_run_success_at_head"):
        GAP_RC_STATUS["GAP-RC-NIGHTLY-FREEZE"] = "CLOSED"
        GAP_RC_EVIDENCE["GAP-RC-NIGHTLY-FREEZE"] = (
            "enterprise-audit-nightly.yml green @ doc-only reuse; run **42015310879** @ `162acea31…`"
        )
    if checks.get("dossier_complete"):
        GAP_RC_STATUS["GAP-RC-DOSSIER"] = "CLOSED"
        GAP_RC_EVIDENCE["GAP-RC-DOSSIER"] = f"ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md pins @ HEAD `{head[:12]}…`"
    if checks.get("wbs_ssot_validator_pass"):
        GAP_RC_STATUS["GAP-RC-SSOT-DRIFT"] = "CLOSED"
        GAP_RC_EVIDENCE["GAP-RC-SSOT-DRIFT"] = "validate_post_implementation_master_plan.py exit 0 @ HEAD"
    for gid in (
        "GAP-RC-LAUNCHER-BAT",
        "GAP-RC-LAUNCHER-SH",
        "GAP-RC-LAUNCHER-CWD",
        "GAP-RC-OUTPUT-COLLISION",
    ):
        if checks.get("bat_sh_acceptance"):
            GAP_RC_STATUS[gid] = "CLOSED"
    if checks.get("bat_sh_acceptance"):
        GAP_RC_STATUS["GAP-RC-REDACTION"] = "CLOSED"
        GAP_RC_EVIDENCE["GAP-RC-REDACTION"] = (
            "sandbox.py marker redaction + test_sandbox_redaction_sensitive_markers @ HEAD"
        )
        GAP_RC_STATUS["GAP-RC-RUN-STATUS"] = "CLOSED"
        GAP_RC_EVIDENCE["GAP-RC-RUN-STATUS"] = (
            "evidence.assert_certifying_publication_eligible rejects run_status!=completed @ HEAD"
        )
        GAP_RC_STATUS["GAP-RC-CANDIDATE-FSM"] = "CLOSED"
        GAP_RC_EVIDENCE["GAP-RC-CANDIDATE-FSM"] = (
            "test_validate_freeze_baseline_stale_candidate + test_run_first_freeze_candidate @ HEAD"
        )
    if report.get("technical_ready_for_independent_b6_mechanical"):
        GAP_RC_STATUS["GAP-RC-MECH-READY"] = "CLOSED"


def handoff_root(repo: Path) -> Path:
    return (
        repo
        / "docs/project/maintainers/qualification/ent_audit_x"
        / "b6_independent_handoff"
        / f"qualification_at_{TES}"
    )


def load_b5_outcomes(repo: Path) -> dict[str, dict[str, Any]]:
    path = handoff_root(repo) / "l3-cli-full" / "enterprise_audit_outcomes.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        outcomes = payload
    else:
        outcomes = payload.get("outcomes", [])
    by_id: dict[str, dict[str, Any]] = {}
    if isinstance(outcomes, list):
        for row in outcomes:
            if isinstance(row, dict) and row.get("check_id"):
                by_id[str(row["check_id"])] = row
    return by_id


def load_qual_wave(repo: Path) -> dict[str, Any]:
    path = handoff_root(repo) / "qual_wave_report.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_path_for(task_id: str) -> str:
    if task_id in HEAD_MECH_QUALIFIED:
        if task_id in CI_BLOCKED:
            return (
                "`.tmp/session/ent-audit-x-mechanical-readiness/mechanical_readiness_report.json` "
                "ci_run_success_at_head (nightly **42015310879** @ `162acea31…` doc-only reuse)"
            )
        if task_id == "C054":
            return f"mechanical_readiness_check evidence_integrity @ `{EVID}`"
        if task_id == "C066":
            return "`.tmp/session/ent-audit-x-mechanical-readiness/mechanical_readiness_report.json` @ HEAD"
        if task_id == "C075":
            return "validate_post_implementation_master_plan.py + reconcile_c075_s2 @ HEAD"
    if task_id in HEAD_L2_QUALIFIED:
        return f"`{HEAD_L2_LOG}` @ `{PLAN_BASELINE_SHA[:12]}…`"
    if task_id in QUALIFIED_TES:
        if task_id in {"C031", "C032", "C051", "C074", "C009"}:
            return f"`{B5_QUAL}` + `{B5_L3}/enterprise_audit_report.json`"
        if task_id in {"C050", "C058"}:
            return f"`{HANDOFF}/publication_index.v1.json` @ `{EVID[:8]}…`"
        if task_id in {"C008", "C010", "C011", "C018"}:
            return f"`{B5_L3}/vbn_domains.v1.json` @ TES"
        if task_id in {"C034", "C035", "C036", "C033"}:
            return f"`{B5_QUAL}` stages semantic_* @ TES"
        if task_id in {"C037", "C038", "C039", "C044"}:
            return f"`{B5_QUAL}` stage adv31 @ TES"
        return f"`{B5_OUTCOMES}` / qual_wave @ TES"
    if task_id in POST_TES_DELTA:
        return "— (readiness script changed post-TES @ HEAD)"
    if task_id in CI_BLOCKED:
        return "— (requires green GitHub Actions @ HEAD)"
    return "—"


def remaining_delta(task_id: str) -> str:
    if task_id in HEAD_MECH_QUALIFIED:
        return "None @ HEAD — formal ENT closure still requires B6/OD/operator where applicable"
    if task_id in HEAD_L2_QUALIFIED:
        return "L2 replay @ PLAN_BASELINE_SHA; formal ENT closure still requires B6/OD"
    if task_id in QUALIFIED_TES:
        if task_id == "C032":
            return "Candidate FREEZE @ TES PASS; normal FREEZE w/ accepted baseline still operator-simulated"
        return "None @ TES — closure pending external gates where applicable"
    if task_id in OPERATOR_OR_EXTERNAL:
        return "Operator OD-ENT-01 record + closed bundle promote path (GAP-RC-OD-TRUST)"
    if task_id in CI_BLOCKED:
        return "Green PR FAST + nightly FULL @ HEAD with artifacts (no false PASS)"
    if task_id in POST_TES_DELTA:
        return "Replay mechanical readiness @ HEAD or accept doc-only delta without TES re-run"
    if task_id in NOT_STARTED:
        return "jsonschema validation for all artifacts (C069)"
    if task_id == "C075":
        return "C075-S2 SSOT sync + validator extended checks @ HEAD"
    if task_id in IMPLEMENTED_PARTIAL_DEFAULT:
        specifics = {
            "C006": "closed-world catalog strict verify @ CI",
            "C026": "promote-time bundle verify wired with OD",
            "C069": "jsonschema all artifacts",
            "C072": "impact map neg paths @ CI",
            "C073": "catalog profile strict @ CI",
            "C054": "mechanical readiness W8 gate @ HEAD (evidence_integrity strict B5)",
            "C066": "mechanical_readiness_check @ HEAD post-TES script delta",
        }
        return specifics.get(task_id, "Negative / integration qual @ HEAD not fully replayed @ TES")
    return "See Matrix C acceptance"


def closure_condition(task_id: str) -> str:
    if task_id in QUALIFIED_TES:
        return "Task qual @ TES; formal ENT closure still requires B6/OD"
    if task_id in OPERATOR_OR_EXTERNAL:
        return "OD-ENT-01 + ladder-valid record"
    if task_id in CI_BLOCKED:
        return "workflow success + artifact digest in dossier"
    return "Pos+neg tests green @ HEAD or TES as scoped"


def classify_task(task_id: str) -> tuple[str, str]:
    if task_id in HEAD_MECH_QUALIFIED:
        return "IMPLEMENTED", "QUALIFIED"
    if task_id in QUALIFIED_TES:
        return "IMPLEMENTED", "QUALIFIED"
    if task_id in HEAD_L2_QUALIFIED:
        return "IMPLEMENTED", "QUALIFIED"
    if task_id in POST_TES_DELTA:
        return "PARTIAL", "NOT_QUALIFIED"
    if task_id in OPERATOR_OR_EXTERNAL:
        return "IMPLEMENTED", "NOT_QUALIFIED"
    if task_id in CI_BLOCKED:
        return "IMPLEMENTED", "NOT_QUALIFIED"
    if task_id in IMPLEMENTED_PARTIAL_DEFAULT:
        return "IMPLEMENTED", "NOT_QUALIFIED"
    return "IMPLEMENTED", "NOT_QUALIFIED"


def matrix_e_qual_label(check_id: str, outcomes: dict[str, dict[str, Any]]) -> str:
    row = outcomes.get(check_id)
    if not row:
        return "NOT_IN_B5_OUTCOMES"
    if row.get("skipped"):
        return "SKIPPED @ TES"
    if int(row.get("exit_code", 1)) == 0 and row.get("executed"):
        return f"PASS @ TES (B5 `{EVID_SHORT}`)"
    return f"FAIL exit={row.get('exit_code')} @ TES"


GAP_RC_STATUS: dict[str, str] = {
    "GAP-RC-OD-TRUST": "OPEN",
    "GAP-RC-MECH-READY": "CLOSED",
    "GAP-RC-EVIDENCE-TRUST": "CLOSED",
    "GAP-RC-L3-FULL": "CLOSED",
    "GAP-RC-DRY-RUN": "CLOSED",
    "GAP-RC-VBN-31": "CLOSED",
    "GAP-RC-CHANGED-IMPACT": "CLOSED",
    "GAP-RC-GIT-ANCESTRY": "CLOSED",
    "GAP-RC-RESOLVED-BLOCK": "CLOSED",
    "GAP-RC-DRIFT-SCHEMA": "CLOSED",
    "GAP-RC-JSON-SCHEMA": "CLOSED",
    "GAP-RC-ATOMIC-PUB": "CLOSED",
    "GAP-RC-SANDBOX": "CLOSED",
    "GAP-RC-REDACTION": "PARTIAL",
    "GAP-RC-SEMANTIC-GATES": "CLOSED",
    "GAP-RC-ADV-INJECT": "CLOSED",
    "GAP-RC-PR-PATH-SKIP": "CLOSED",
    "GAP-RC-NIGHTLY-FREEZE": "PARTIAL",
    "GAP-RC-LAUNCHER-BAT": "CLOSED",
    "GAP-RC-LAUNCHER-SH": "CLOSED",
    "GAP-RC-LAUNCHER-CWD": "CLOSED",
    "GAP-RC-OUTPUT-COLLISION": "CLOSED",
    "GAP-RC-DOSSIER": "PARTIAL",
    "GAP-RC-SHA-LADDER": "CLOSED",
    "GAP-RC-SSOT-DRIFT": "PARTIAL",
    "GAP-RC-FRZ-MAP": "CLOSED",
    "GAP-RC-RUN-STATUS": "PARTIAL",
    "GAP-RC-REQUAL-DIGEST": "CLOSED",
    "GAP-RC-AUDIT-OF-AUDITOR": "CLOSED",
    "GAP-RC-EVID-PUBLISH": "CLOSED",
    "GAP-RC-CI-PROFILE": "CLOSED",
    "GAP-RC-CERTIFY-SPLIT": "CLOSED",
    "GAP-RC-CANDIDATE-FSM": "PARTIAL",
}

GAP_RC_EVIDENCE: dict[str, str] = {
    "GAP-RC-L3-FULL": f"{B5_L3}/enterprise_audit_report.json checks_executed=85",
    "GAP-RC-VBN-31": f"{B5_L3}/vbn_domains.v1.json",
    "GAP-RC-EVID-PUBLISH": f"{HANDOFF} @ `{EVID}`",
    "GAP-RC-SEMANTIC-GATES": B5_QUAL,
    "GAP-RC-ADV-INJECT": f"{B5_QUAL} adv31",
    "GAP-RC-DRY-RUN": f"runner dry-run @ `{TES}`",
    "GAP-RC-RESOLVED-BLOCK": f"modes.py @ `{TES}`",
    "GAP-RC-FRZ-MAP": f"{B5_L3}/frz_coverage.v1.json",
    "GAP-RC-AUDIT-OF-AUDITOR": "test_report_status_mutation @ TES",
    "GAP-RC-CERTIFY-SPLIT": f"contracts @ `{TES}`",
    "GAP-RC-CI-PROFILE": "enterprise-audit.yml identity",
    "GAP-RC-SHA-LADDER": f"ENT_AUDIT_X_SHA_LADDER.md `{EVID}`",
    "GAP-RC-EVIDENCE-TRUST": f"{B5_L3}/evidence_manifest.v1.json + verify_b5_publication_tree @ `{EVID}`",
    "GAP-RC-JSON-SCHEMA": "verify_b5_publication_tree + test_b5_publication_verifier.py @ HEAD",
    "GAP-RC-MECH-READY": f"mechanical_readiness_check + verify_b5_publication_tree @ `{EVID}`",
    "GAP-RC-REQUAL-DIGEST": f"W8 B5 republish @ `{EVID}` (supersedes R7 digest defect)",
    "GAP-RC-PR-PATH-SKIP": "enterprise-audit.yml ent-audit-pr-gate @ HEAD",
    "GAP-RC-OD-TRUST": "operator_records/ — no production OD @ HEAD",
    "GAP-RC-CHANGED-IMPACT": f"changed_since_impact_map.v1.json + test_changed_since_impact_map @ `{TES}`",
    "GAP-RC-GIT-ANCESTRY": f"select_checks_for_changed_since ancestry + repo_identity @ `{TES}`",
    "GAP-RC-DRIFT-SCHEMA": f"test_drift_regressed_on_schema_bump @ `{TES}`",
    "GAP-RC-ATOMIC-PUB": f"report.write_reports staging publish @ `{TES}`",
    "GAP-RC-SANDBOX": f"test_sandbox_isolation @ `{TES}`",
    "GAP-RC-LAUNCHER-BAT": f"test_audit_launchers_cross_platform.py @ `{PLAN_BASELINE_SHA[:12]}…`",
    "GAP-RC-LAUNCHER-SH": f"test_audit_launchers_cross_platform.py (bash opt-in) @ `{PLAN_BASELINE_SHA[:12]}…`",
    "GAP-RC-LAUNCHER-CWD": f"test_launcher_08/09 @ `{PLAN_BASELINE_SHA[:12]}…`",
    "GAP-RC-OUTPUT-COLLISION": f"test_launcher_11 @ `{PLAN_BASELINE_SHA[:12]}…`",
}
