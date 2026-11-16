"""D-3: ENT formal-closure preflight - Gate A / Gate B under IIAC.

Trees are built in ``tmp_path``; ancestry is stubbed for the file-level cases and real (a temporary git repo)
for the ordering cases. Every negative case must make the gate fail closed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import scripts.qualification.ent_audit_x_formal_closure_preflight as pre

POLICY = "IIAC-2026-11-05.1"
QUAL = Path("docs/project/maintainers/qualification")
ENT = QUAL / "ent_audit_x"
D64 = "d" * 64
TES, EVID, B6, BASE, FORMAL = ("1" * 40, "2" * 40, "3" * 40, "4" * 40, "5" * 40)
AUDITED, AUDIT_EXEC = "a" * 40, "b" * 40
AUDIT_DIR = QUAL / "iiac_evidence/ENT-AUDIT-X/audit"
_AUDIT_FILES = {
    "IIAC_ENT_AUDIT_REPORT.md": "# audit report\n",
    "audit_scope.json": "{}\n",
    "test_matrix.txt": "matrix\n",
    "findings.json": "[]\n",
}


def _w(root: Path, rel: Path | str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _digest(root: Path, rel: Path | str) -> str:
    return hashlib.sha256((root / rel).read_bytes()).hexdigest()


def _frz(
    status10: str = "PASS — COMPENSATING (SCOPED)", open_rows: tuple[int, ...] = (), skip: int | None = None
) -> str:
    rows = ["| ID | Family | Requirement | Owner | Status | Note |", "| --- | --- | --- | --- | --- | --- |"]
    for n in range(1, 11):
        if n == skip:
            continue
        status = "OPEN" if n in open_rows else (status10 if n == 10 else "PASS")
        rows.append(f"| FRZ-AUD-{n:02d} | Continuous enterprise audit | requirement | ENT-AUDIT-X | {status} | — |")
    return "\n".join(rows) + "\n"


def make_tree(root: Path, gate: str = "A", **opt: Any) -> Path:  # noqa: C901 - one fixture, many switches
    pins = {"B6": B6, "BASE": BASE, "FORMAL": FORMAL if gate == "B" else None}
    pins.update(opt.get("pins", {}))
    fmt = lambda sha: f"`{sha}`" if sha else "*unset*"  # noqa: E731
    _w(
        root,
        pre._LADDER,
        "\n".join(
            [
                "| Role | SHA | Note |",
                "| --- | --- | --- |",
                f"| TEST_EXECUTION_SHA | `{TES}` | tes |",
                f"| EVIDENCE_PUBLICATION_SHA | `{EVID}` | evid |",
                f"| B6_ACCEPTANCE_PUBLICATION_SHA | {fmt(pins['B6'])} | b6 |",
                f"| ACCEPTED_BASELINE_SHA | {fmt(pins['BASE'])} | baseline |",
                f"| FORMAL_CLOSURE_PUBLICATION_SHA | {fmt(pins['FORMAL'])} | closure |",
                "",
            ]
        ),
    )
    _w(root, pre._ACTIVATION, json.dumps({"policy_version": POLICY, "status": "IN_FORCE"}))
    evol = {
        "parent_evol_x_status": "CLOSED_INDEPENDENTLY_ACCEPTED_AUTOMATION_FIRST_SCOPED_PUBLISHED",
        "protocol_03": "SUPERSEDED_BY_IIAC_B6_ACCEPTANCE",
        "w7_06_handoff_publication_sha": "7" * 40,
    }
    evol.update(opt.get("evol", {}))
    _w(root, pre._EVOL_STATUS, json.dumps(evol))
    status = {
        "formal_b6_status": "ACCEPTED_VIA_IIAC_B6",
        "b6_acceptance_publication_sha": pins["B6"],
        "accepted_baseline_sha": pins["BASE"],
        "frz_aud_status": "PASS_SCOPED",
        "od_ent_02_mode": "compensating",
        "formal_closure_publication_sha": pins["FORMAL"],
    }
    status.update(opt.get("status", {}))
    _w(root, pre._STATUS, json.dumps(status))
    audited, audit_exec = opt.get("audited", AUDITED), opt.get("audit_exec", AUDIT_EXEC)
    if opt.get("pass_record", True):
        default_text = f"# IIAC pass record\nverdict PASS_SCOPED at {audit_exec}\n"
        _w(root, pre._PASS_RECORD, opt.get("pass_text", default_text))
    for name, text in _AUDIT_FILES.items():
        _w(root, AUDIT_DIR / name, text)
    index_files = {
        name: {"sha256": _digest(root, AUDIT_DIR / name), "bytes": (root / AUDIT_DIR / name).stat().st_size}
        for name in _AUDIT_FILES
    }
    if opt.get("index_tamper"):
        index_files["findings.json"]["sha256"] = "0" * 64
    _w(root, AUDIT_DIR / "EVIDENCE_INDEX.json", json.dumps({"schema_version": 1, "files": index_files}))
    manifest = {
        "stage_id": "ENT-AUDIT-X",
        "verdict": "PASS_SCOPED",
        "policy_version": POLICY,
        "implementation_session_id": "I",
        "auditor_session_id": "A",
        "certifier_session_id": "C",
        "audited_implementation_sha": audited,
        "audit_execution_sha": audit_exec,
        "audit_report_path": (AUDIT_DIR / "IIAC_ENT_AUDIT_REPORT.md").as_posix(),
        "evidence_index_path": (AUDIT_DIR / "EVIDENCE_INDEX.json").as_posix(),
        "pass_record_path": pre._PASS_RECORD.as_posix(),
        "audit_report_digest": _digest(root, AUDIT_DIR / "IIAC_ENT_AUDIT_REPORT.md"),
        "audit_scope_manifest_digest": _digest(root, AUDIT_DIR / "audit_scope.json"),
        "test_matrix_digest": _digest(root, AUDIT_DIR / "test_matrix.txt"),
        "findings_digest": _digest(root, AUDIT_DIR / "findings.json"),
        "limitations": ["LIM-1 process_enforced"],
        "assurance_level": "process_enforced",
        "test_execution_sha": TES,
        "evidence_publication_sha": EVID,
    }
    manifest.update(opt.get("manifest", {}))
    if opt.get("manifest_file", True):
        _w(root, pre._MANIFEST, json.dumps(manifest))
    _w(root, pre._FRZ, _frz(**opt.get("frz", {})))
    headline = "" if gate == "A" else pre.HEADLINE
    _w(root, pre._ROADMAP, opt.get("roadmap", f"| ENT-AUDIT-X | {headline} |\n"))
    return root


def run(root: Path, gate: str = "A", **kw: Any) -> list[str]:
    kw.setdefault("iiac_in_force", True)
    kw.setdefault("is_ancestor", lambda _sha: True)
    kw.setdefault("compensating_verifier", lambda _repo: [])
    kw.setdefault("live_verifier", lambda: [])
    kw.setdefault("manifest_verifier", lambda _repo, _manifest: [])
    kw.setdefault("branch_rules_reader", lambda: ([{"type": "deletion"}, {"type": "non_fast_forward"}], None))
    return pre.evaluate(gate, root, **kw)


def test_gate_a_passes_on_a_complete_tree(tmp_path: Path) -> None:
    assert run(make_tree(tmp_path)) == []


def test_gate_b_passes_on_a_complete_tree(tmp_path: Path) -> None:
    assert run(make_tree(tmp_path, "B"), "B") == []


def test_iiac_not_in_force_fails_closed(tmp_path: Path) -> None:
    errors = run(make_tree(tmp_path), iiac_in_force=False)
    assert errors and "not IN_FORCE" in errors[0]


GATE_A = {
    "b6_pin_unset_star_unset_cell": ({"pins": {"B6": None}}, "B6_ACCEPTANCE_PUBLICATION_SHA is unset"),
    "baseline_pin_unset": ({"pins": {"BASE": None}}, "ACCEPTED_BASELINE_SHA is unset"),
    "status_b6_differs_from_ladder": (
        {"status": {"b6_acceptance_publication_sha": "9" * 40}},
        "does not equal the SHA ladder",
    ),
    "status_baseline_differs_from_ladder": (
        {"status": {"accepted_baseline_sha": "9" * 40}},
        "does not equal the SHA ladder",
    ),
    "formal_b6_pending": ({"status": {"formal_b6_status": "PENDING_AUTHORITY"}}, "formal_b6_status"),
    "frz_aud_status_open": ({"status": {"frz_aud_status": "OPEN"}}, "frz_aud_status still OPEN"),
    "formal_pin_already_set": ({"status": {"formal_closure_publication_sha": FORMAL}}, "already set before closure"),
    "pass_record_missing": (
        {"pass_record": False},
        "missing docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_IIAC",
    ),
    "manifest_missing": ({"manifest_file": False}, "ENT IIAC audit manifest"),
    "manifest_verdict_fail": ({"manifest": {"verdict": "FAIL"}}, "verdict must be PASS"),
    "manifest_wrong_stage": ({"manifest": {"stage_id": "EVOL-X"}}, "stage_id"),
    "manifest_policy_mismatch": ({"manifest": {"policy_version": "IIAC-0"}}, "policy_version"),
    "manifest_sessions_not_distinct": ({"manifest": {"auditor_session_id": "I"}}, "pairwise distinct"),
    "manifest_digest_missing": ({"manifest": {"findings_digest": ""}}, "findings_digest"),
    "manifest_no_limitations": ({"manifest": {"limitations": []}}, "limitations"),
    "manifest_no_assurance_level": ({"manifest": {"assurance_level": ""}}, "assurance_level"),
    "manifest_tes_differs_from_ladder": ({"manifest": {"test_execution_sha": "8" * 40}}, "test_execution_sha"),
    "manifest_audited_sha_not_a_commit": (
        {"manifest": {"audited_implementation_sha": "xyz"}},
        "audited_implementation_sha",
    ),
    "manifest_report_digest_forged": ({"manifest": {"audit_report_digest": "f" * 64}}, "audit_report_digest differs"),
    "manifest_findings_digest_forged": ({"manifest": {"findings_digest": "e" * 64}}, "not the sha256 of any file"),
    "manifest_report_path_missing": ({"manifest": {"audit_report_path": "nope/missing.md"}}, "audit_report_path"),
    "manifest_report_path_escapes_repo": ({"manifest": {"audit_report_path": "../outside.md"}}, "audit_report_path"),
    "manifest_index_missing": ({"manifest": {"evidence_index_path": "nope/index.json"}}, "evidence_index_path"),
    "evidence_index_digest_tampered": ({"index_tamper": True}, "does not match the file"),
    "pass_record_one_line_fabrication": ({"pass_text": "fabricated\n"}, "must cite the manifest verdict"),
    "evol_not_closed": ({"evol": {"parent_evol_x_status": "OPEN"}}, "EVOL predecessor"),
    "evol_still_protocol_03": (
        {"evol": {"protocol_03": "PENDING", "formal_ecap_independent_b6": "PENDING"}},
        "EVOL predecessor",
    ),
    "evol_handoff_pin_missing": ({"evol": {"w7_06_handoff_publication_sha": None}}, "EVOL predecessor"),
    "od02_mode_unset": ({"status": {"od_ent_02_mode": None}}, "od_ent_02_mode"),
    "od02_mode_unknown": ({"status": {"od_ent_02_mode": "equivalent"}}, "od_ent_02_mode"),
    "frz_row_open": ({"frz": {"open_rows": (3,)}}, "FRZ-AUD-03 still OPEN"),
    "frz_row_missing": ({"frz": {"skip": 7}}, "FRZ-AUD-07 row missing"),
    "frz_row10_not_compensating": ({"frz": {"status10": "PASS"}}, "COMPENSATING"),
    "roadmap_premature_headline": ({"roadmap": pre.HEADLINE + "\n"}, "premature"),
    "roadmap_legacy_headline": ({"roadmap": pre.LEGACY_HEADLINE + "\n"}, "superseded"),
}


@pytest.mark.parametrize(("opt", "match"), GATE_A.values(), ids=GATE_A.keys())
def test_gate_a_rejects(tmp_path: Path, opt: dict[str, Any], match: str) -> None:
    errors = run(make_tree(tmp_path, **opt))
    assert any(match in error for error in errors), errors


def test_pin_that_is_not_an_ancestor_is_rejected(tmp_path: Path) -> None:
    errors = run(make_tree(tmp_path), is_ancestor=lambda sha: sha != B6)
    assert any("B6_ACCEPTANCE_PUBLICATION_SHA is not an ancestor" in e for e in errors), errors


def test_compensating_verifier_failure_blocks(tmp_path: Path) -> None:
    errors = run(make_tree(tmp_path), compensating_verifier=lambda _r: ["no auditor acceptance"])
    assert any("OD-ENT-02 compensating: no auditor acceptance" in e for e in errors), errors


def test_live_mode_requires_live_verifier_and_row10_live(tmp_path: Path) -> None:
    tree = make_tree(tmp_path, status={"od_ent_02_mode": "live"}, frz={"status10": "PASS — LIVE"})
    assert run(tree) == []
    errors = run(tree, live_verifier=lambda: ["Missing required status check context"])
    assert any("OD-ENT-02 live:" in e for e in errors), errors


def test_compensating_row10_must_not_claim_live(tmp_path: Path) -> None:
    errors = run(make_tree(tmp_path, frz={"status10": "PASS — LIVE"}))
    assert any("COMPENSATING" in e for e in errors), errors


GATE_B = {
    "headline_missing": ({"roadmap": "| ENT-AUDIT-X | pending |\n"}, "exactly once"),
    "headline_twice": ({"roadmap": (pre.HEADLINE + "\n") * 2}, "exactly once"),
    "legacy_headline": ({"roadmap": pre.HEADLINE + "\n" + pre.LEGACY_HEADLINE + "\n"}, "superseded"),
    "formal_pin_unset": ({"pins": {"FORMAL": None}}, "FORMAL_CLOSURE_PUBLICATION_SHA is unset"),
    "formal_pin_differs_from_status": ({"status": {"formal_closure_publication_sha": "9" * 40}}, "does not equal"),
}


@pytest.mark.parametrize(("opt", "match"), GATE_B.values(), ids=GATE_B.keys())
def test_gate_b_rejects(tmp_path: Path, opt: dict[str, Any], match: str) -> None:
    errors = run(make_tree(tmp_path, "B", **opt), "B")
    assert any(match in error for error in errors), errors


def test_ladder_pin_parser_never_reports_unset_as_set() -> None:
    table = "| X | *unset* | n |\n| Y | `" + "a" * 40 + "` | n |\n| Y | **PENDING** | n |\n| Z | `abc` | n |\n"
    assert pre.ladder_pin(table, "X") is None
    assert pre.ladder_pin(table, "Y") is None  # last row wins and it is not a sha
    assert pre.ladder_pin(table, "Z") is None  # malformed
    assert pre.ladder_pin(table, "MISSING") is None
    assert pre.ladder_pin("| Y | `" + "a" * 40 + "` | n |\n", "Y") == "a" * 40


def test_main_default_gate_fails_on_the_real_repo_when_iiac_is_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pre, "_iiac_in_force", lambda: True)
    assert pre.main() != 0


# ---- Gate B ordering (real git) -------------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--allow-empty", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _git_tree(root: Path, *, same_commit: bool, record_after_headline: bool = False) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "user.email", "t@example.invalid")
    _w(root, "seed.txt", "x")
    _commit(root, "seed")
    pass_text = "# IIAC pass record\n"
    headline_line = f"| ENT-AUDIT-X | {pre.HEADLINE} |\n"
    if same_commit:
        _w(root, pre._PASS_RECORD, pass_text)
        _w(root, pre._ROADMAP, headline_line)
        b6 = formal = _commit(root, "record and headline together")
    elif record_after_headline:
        _w(root, pre._ROADMAP, headline_line)
        formal = _commit(root, "headline first")
        _w(root, pre._PASS_RECORD, pass_text)
        b6 = _commit(root, "record later")
    else:
        _w(root, pre._PASS_RECORD, pass_text)
        b6 = _commit(root, "record")
        _w(root, pre._ROADMAP, headline_line)
        formal = _commit(root, "headline")
    seed_sha = _git(root, "rev-list", "--max-parents=0", "HEAD")
    make_tree(
        root, "B", pins={"B6": b6, "BASE": b6, "FORMAL": formal}, roadmap=headline_line, audited=seed_sha,
        audit_exec=seed_sha,
    )  # fmt: skip
    _commit(root, "publish status, ladder, manifest")


def _real_git_run(root: Path) -> list[str]:
    return pre.evaluate(
        "B",
        root,
        iiac_in_force=True,
        compensating_verifier=lambda _r: [],
        live_verifier=lambda: [],
        manifest_verifier=lambda _repo, _manifest: [],
    )


def test_gate_b_accepts_pass_record_published_before_the_headline(tmp_path: Path) -> None:
    _git_tree(tmp_path, same_commit=False)
    assert _real_git_run(tmp_path) == []


def test_gate_b_rejects_record_and_headline_in_one_commit(tmp_path: Path) -> None:
    _git_tree(tmp_path, same_commit=True)
    assert any("earlier commit than the closure headline" in e for e in _real_git_run(tmp_path))


def test_gate_b_rejects_record_published_after_the_headline(tmp_path: Path) -> None:
    _git_tree(tmp_path, same_commit=False, record_after_headline=True)
    assert any("earlier commit than the closure headline" in e for e in _real_git_run(tmp_path))


def test_audit_sha_that_is_not_an_ancestor_is_rejected(tmp_path: Path) -> None:
    errors = run(make_tree(tmp_path), is_ancestor=lambda sha: sha != AUDIT_EXEC)
    assert any("audit_execution_sha is not a commit that is an ancestor" in e for e in errors), errors


def test_manifest_verifier_failures_block(tmp_path: Path) -> None:
    errors = run(make_tree(tmp_path), manifest_verifier=lambda _repo, _manifest: ["jsonschema: boom"])
    assert any("IIAC manifest verifier: jsonschema: boom" in e for e in errors), errors


def test_default_manifest_verifier_rejects_an_incomplete_manifest(tmp_path: Path) -> None:
    tree = make_tree(tmp_path)
    errors = pre._default_manifest_verifier(tree, tree / pre._MANIFEST)
    assert errors, "the repository manifest verifier must reject a manifest that lacks required fields"
