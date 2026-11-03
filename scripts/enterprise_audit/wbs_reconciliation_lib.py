# © Artur Czarnecki. All rights reserved.

"""C075-S2 WBS reconciliation rules @ TEST_EXECUTION_SHA + B5 evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TES = "b8bb88c9db1060153ec0a6a29bbb79ba16e00078"
EVID = "894b0e1aff071d19636d23bea406b95dcffc79e1"
EVID_SHORT = EVID[:12]
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
CI_BLOCKED: frozenset[str] = frozenset({"C045", "C046", "C047", "C059", "C060", "C061", "C063", "C070", "C007"})

# Post-TES changes to readiness scripts — not re-run @ TES.
POST_TES_DELTA: frozenset[str] = frozenset({"C054", "C066"})

NOT_STARTED: frozenset[str] = frozenset()

IMPLEMENTED_PARTIAL_DEFAULT: frozenset[str] = frozenset(
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
        "C075",
    }
)


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
    if task_id in NOT_STARTED:
        return "NOT_STARTED", "NOT_QUALIFIED"
    if task_id in QUALIFIED_TES:
        return "IMPLEMENTED", "QUALIFIED"
    if task_id in {"C004", "C005", "C067", "C071"}:
        return "IMPLEMENTED", "QUALIFIED"
    if task_id in POST_TES_DELTA:
        return "PARTIAL", "NOT_QUALIFIED"
    if task_id in OPERATOR_OR_EXTERNAL:
        return "PARTIAL", "NOT_QUALIFIED"
    if task_id in CI_BLOCKED:
        return "PARTIAL", "NOT_QUALIFIED"
    if task_id in IMPLEMENTED_PARTIAL_DEFAULT:
        return "PARTIAL", "NOT_QUALIFIED"
    return "PARTIAL", "NOT_QUALIFIED"


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
    "GAP-RC-MECH-READY": "PARTIAL",
    "GAP-RC-EVIDENCE-TRUST": "CLOSED",
    "GAP-RC-L3-FULL": "CLOSED",
    "GAP-RC-DRY-RUN": "CLOSED",
    "GAP-RC-VBN-31": "CLOSED",
    "GAP-RC-CHANGED-IMPACT": "OPEN",
    "GAP-RC-GIT-ANCESTRY": "PARTIAL",
    "GAP-RC-RESOLVED-BLOCK": "CLOSED",
    "GAP-RC-DRIFT-SCHEMA": "PARTIAL",
    "GAP-RC-JSON-SCHEMA": "CLOSED",
    "GAP-RC-ATOMIC-PUB": "PARTIAL",
    "GAP-RC-SANDBOX": "OPEN",
    "GAP-RC-REDACTION": "PARTIAL",
    "GAP-RC-SEMANTIC-GATES": "CLOSED",
    "GAP-RC-ADV-INJECT": "CLOSED",
    "GAP-RC-PR-PATH-SKIP": "OPEN",
    "GAP-RC-NIGHTLY-FREEZE": "PARTIAL",
    "GAP-RC-LAUNCHER-BAT": "PARTIAL",
    "GAP-RC-LAUNCHER-SH": "PARTIAL",
    "GAP-RC-LAUNCHER-CWD": "PARTIAL",
    "GAP-RC-OUTPUT-COLLISION": "PARTIAL",
    "GAP-RC-DOSSIER": "PARTIAL",
    "GAP-RC-SHA-LADDER": "CLOSED",
    "GAP-RC-SSOT-DRIFT": "PARTIAL",
    "GAP-RC-FRZ-MAP": "CLOSED",
    "GAP-RC-RUN-STATUS": "PARTIAL",
    "GAP-RC-REQUAL-DIGEST": "PARTIAL",
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
    "GAP-RC-EVIDENCE-TRUST": f"{B5_L3}/evidence_manifest.v1.json",
    "GAP-RC-JSON-SCHEMA": "test_validators_all_artifacts.py + validate_b5_handoff_bundle @ HEAD",
    "GAP-RC-OD-TRUST": "operator_records/ — no production OD @ HEAD",
}
