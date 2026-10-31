# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E501

"""EBH-7 L3 JUnit evidence: strict totals, skipped-test disposition verification, register rendering.

Subcommands (all read-only with respect to the published dossier; outputs go to ``--out``):

* ``summary``          strict JUnit parse -> totals (tests/passed/failed/errors/skipped/xfailed), fail-closed.
* ``skip-disposition`` every skipped node in the real JUnit must have an ACCEPTED_NA register record whose
                       evidence nodes are PASS in the *same* JUnit; unregistered/stale skips fail.
* ``meta-lines``       ``key=value`` lines for ``meta.txt`` built from the two JSON outputs (never zeros on error).
* ``render-register``  render (or ``--check``) the human-readable skip disposition register document.

No JUnit total is hardcoded: all values are read from the file.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from tests.qualification.ebh_7._ebh7_skip_disposition_register import (  # noqa: E402
    SKIP_DISPOSITION_REGISTER,
    SkipAuditStatus,
    SkipDisposition,
    SkipDispositionRecord,
)

REGISTER_DOC_REL = (
    "docs/project/maintainers/qualification/ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md"
)
_F07_MODULE = "test_ebh_5_dual_slug_rpl_e_operational_gates"
_XFAIL_TYPE = "pytest.xfail"


class JunitError(Exception):
    """Raised for missing, malformed, or internally inconsistent JUnit XML."""


@dataclass(frozen=True, slots=True)
class JunitCase:
    node_id: str
    outcome: str  # passed | failed | error | skipped | xfailed
    skip_message: str = ""


@dataclass(frozen=True, slots=True)
class JunitReport:
    root_tag: str
    suites: int
    tests: int
    passed: int
    failed: int
    errors: int
    skipped: int
    xfailed: int
    cases: tuple[JunitCase, ...]

    def totals(self) -> dict[str, Any]:
        return {
            "root_tag": self.root_tag,
            "suites": self.suites,
            "tests": self.tests,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "xfailed": self.xfailed,
            "f07_operational_gate_nodes": sum(1 for c in self.cases if _F07_MODULE in c.node_id),
        }


def junit_node_id(classname: str, name: str, repo: Path = _REPO) -> str:
    """Convert JUnit ``classname``/``name`` to a pytest node id (longest existing module path wins)."""
    parts = classname.split(".")
    for i in range(len(parts), 0, -1):
        rel = "/".join(parts[:i]) + ".py"
        if (repo / rel).is_file():
            cls = ".".join(parts[i:])
            return f"{rel}::{cls + '::' if cls else ''}{name}"
    return f"{classname}::{name}"


def _declared(suite: ET.Element, attr: str, label: str) -> int:
    raw = suite.get(attr)
    if raw is None:
        raise JunitError(f"suite '{label}' missing required statistic '{attr}'")
    try:
        value = int(raw)
    except ValueError as exc:
        raise JunitError(f"suite '{label}' statistic '{attr}' is not an integer: {raw!r}") from exc
    if value < 0:
        raise JunitError(f"suite '{label}' statistic '{attr}' negative: {value}")
    return value


def parse_junit(path: Path, repo: Path = _REPO) -> JunitReport:
    """Strict, fail-closed JUnit parse (root ``testsuite`` or ``testsuites``; leaf suites counted once)."""
    path = Path(path)
    if not path.is_file():
        raise JunitError(f"JUnit file missing: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise JunitError(f"JUnit XML malformed: {exc}") from exc
    if root.tag not in ("testsuite", "testsuites"):
        raise JunitError(f"unsupported JUnit root element: {root.tag!r}")

    if root.tag == "testsuites":
        for child in root:
            if child.tag == "testcase":
                raise JunitError("testcase outside any testsuite")
    leaf_suites = [s for s in root.iter("testsuite") if any(c.tag == "testcase" for c in s)]
    if not leaf_suites:
        raise JunitError("JUnit contains no testcase elements")

    cases: list[JunitCase] = []
    seen: set[str] = set()
    sum_tests = sum_failures = sum_errors = sum_skipped = 0
    for idx, suite in enumerate(leaf_suites):
        label = suite.get("name") or f"#{idx}"
        direct = [c for c in suite if c.tag == "testcase"]
        n_fail = n_err = n_skip = 0
        for tc in direct:
            classname, name = tc.get("classname"), tc.get("name")
            if not name:
                raise JunitError(f"testcase without name in suite '{label}'")
            node_id = junit_node_id(classname or "", name, repo)
            if node_id in seen:
                raise JunitError(f"duplicate testcase (repeated statistics): {node_id}")
            seen.add(node_id)
            failure, error, skipped = tc.find("failure"), tc.find("error"), tc.find("skipped")
            if failure is not None:
                n_fail += 1
            if error is not None:
                n_err += 1
            if skipped is not None:
                n_skip += 1
            if error is not None:
                outcome, msg = "error", ""
            elif failure is not None:
                outcome, msg = "failed", ""
            elif skipped is not None:
                is_xfail = (skipped.get("type") or "") == _XFAIL_TYPE
                outcome, msg = ("xfailed" if is_xfail else "skipped"), (skipped.get("message") or "")
            else:
                outcome, msg = "passed", ""
            cases.append(JunitCase(node_id=node_id, outcome=outcome, skip_message=msg))
        for attr, actual in (
            ("tests", len(direct)),
            ("failures", n_fail),
            ("errors", n_err),
            ("skipped", n_skip),
        ):
            declared = _declared(suite, attr, label)
            if declared != actual:
                raise JunitError(f"suite '{label}' inconsistent '{attr}': declared {declared} != actual {actual}")
        sum_tests += len(direct)
        sum_failures += n_fail
        sum_errors += n_err
        sum_skipped += n_skip

    if root.tag == "testsuites":
        for attr, actual in (
            ("tests", sum_tests),
            ("failures", sum_failures),
            ("errors", sum_errors),
            ("skipped", sum_skipped),
        ):
            raw = root.get(attr)
            if raw is None:
                continue
            try:
                declared = int(raw)
            except ValueError as exc:
                raise JunitError(f"root testsuites '{attr}' is not an integer: {raw!r}") from exc
            if declared != actual:
                raise JunitError(f"root testsuites inconsistent '{attr}': declared {declared} != computed {actual}")

    by_outcome = {
        k: sum(1 for c in cases if c.outcome == k) for k in ("passed", "failed", "error", "skipped", "xfailed")
    }
    total = len(cases)
    if sum(by_outcome.values()) != total or total != sum_tests:
        raise JunitError("internal classification inconsistency")
    return JunitReport(
        root_tag=root.tag,
        suites=len(leaf_suites),
        tests=total,
        passed=by_outcome["passed"],
        failed=by_outcome["failed"],
        errors=by_outcome["error"],
        skipped=by_outcome["skipped"],
        xfailed=by_outcome["xfailed"],
        cases=tuple(cases),
    )


def verify_skip_disposition(
    report: JunitReport,
    register: tuple[SkipDispositionRecord, ...] = SKIP_DISPOSITION_REGISTER,
) -> dict[str, Any]:
    """Compare skipped nodes in a real JUnit to the register; return per-record results and status."""
    outcome_by_node = {c.node_id: c for c in report.cases}
    skipped_nodes = {c.node_id: c for c in report.cases if c.outcome == "skipped"}
    by_node = {r.node_id: r for r in register}
    problems: list[str] = []
    if len(by_node) != len(register):
        problems.append("register contains duplicate node ids")

    unregistered = sorted(set(skipped_nodes) - set(by_node))
    for node in unregistered:
        problems.append(f"UNREGISTERED skip (no disposition): {node}")
    stale = sorted(n for n, r in by_node.items() if n not in skipped_nodes and not r.environment_dependent)
    for node in stale:
        problems.append(f"STALE register record (node did not skip): {node}")

    records: list[dict[str, Any]] = []
    for rec in register:
        rec_problems: list[str] = []
        case = skipped_nodes.get(rec.node_id)
        if case is None:
            skipped_here = False
        else:
            skipped_here = True
            if case.skip_message != rec.skip_reason:
                rec_problems.append(f"skip reason mismatch: junit={case.skip_message!r} register={rec.skip_reason!r}")
        if rec.disposition is not SkipDisposition.ACCEPTED_NA:
            rec_problems.append(f"disposition {rec.disposition.value} is not ACCEPTED_NA")
        if rec.audit_status is not SkipAuditStatus.READY:
            rec_problems.append(f"audit status {rec.audit_status.value} is not READY")
        if rec.mandatory and not rec.evidence_nodes:
            rec_problems.append("mandatory skip without alternative evidence node")
        evidence_results: list[dict[str, str]] = []
        for ev in rec.evidence_nodes:
            ev_case = outcome_by_node.get(ev)
            ev_outcome = ev_case.outcome if ev_case is not None else "MISSING"
            evidence_results.append({"node_id": ev, "junit_outcome": ev_outcome})
            if ev_outcome != "passed":
                rec_problems.append(f"evidence node not PASS in JUnit ({ev_outcome}): {ev}")
        records.append(
            {
                "NODE_ID": rec.node_id,
                "SKIP_REASON": rec.skip_reason,
                "SKIP_SOURCE": rec.skip_source,
                "REQUIREMENT": rec.requirement,
                "MANDATORY": "YES" if rec.mandatory else "NO",
                "DISPOSITION": rec.disposition.value,
                "EVIDENCE": evidence_results,
                "EVIDENCE_DOCUMENTS": list(rec.evidence_documents),
                "SOURCE_SHA": rec.source_sha,
                "JUSTIFICATION": rec.justification,
                "AUDIT_STATUS": "READY" if not rec_problems else "BLOCKED",
                "skipped_in_junit": skipped_here,
                "problems": rec_problems,
            }
        )
        problems.extend(f"{rec.node_id}: {p}" for p in rec_problems)

    return {
        "protocol": "EBH-7-L3-SKIP-DISPOSITION",
        "junit_tests": report.tests,
        "junit_skipped": report.skipped,
        "register_records": len(register),
        "skipped_in_junit": len(skipped_nodes),
        "unregistered_skips": unregistered,
        "stale_records": stale,
        "records": records,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }


def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_summary(junit: Path, out: Path | None) -> int:
    try:
        report = parse_junit(junit)
    except JunitError as exc:
        _write_json(out, {"protocol": "EBH-7-L3-JUNIT-SUMMARY", "status": "FAIL", "reason": str(exc)})
        print(f"junit_summary=FAIL reason={exc}")
        return 1
    payload = {"protocol": "EBH-7-L3-JUNIT-SUMMARY", "status": "PASS", **report.totals()}
    _write_json(out, payload)
    print(
        "junit_summary=PASS "
        f"tests={report.tests} passed={report.passed} failed={report.failed} errors={report.errors} "
        f"skipped={report.skipped} xfailed={report.xfailed}"
    )
    return 0


def cmd_skip_disposition(junit: Path, out: Path | None) -> int:
    try:
        report = parse_junit(junit)
    except JunitError as exc:
        _write_json(out, {"protocol": "EBH-7-L3-SKIP-DISPOSITION", "status": "FAIL", "problems": [str(exc)]})
        print(f"skip_disposition=FAIL reason={exc}")
        return 1
    payload = verify_skip_disposition(report)
    _write_json(out, payload)
    print(
        f"skip_disposition={payload['status']} skipped={payload['skipped_in_junit']} "
        f"registered={payload['register_records']} problems={len(payload['problems'])}"
    )
    for problem in payload["problems"][:20]:
        print(f"  {problem}")
    return 0 if payload["status"] == "PASS" else 1


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def meta_lines(summary_path: Path | None, skip_path: Path | None) -> list[str]:
    """``meta.txt`` lines. Missing/failed inputs yield ``unknown`` + FAIL — never zero statistics."""
    summary = _load_json(summary_path)
    skip = _load_json(skip_path)
    ok_summary = summary is not None and summary.get("status") == "PASS"
    ok_skip = skip is not None and skip.get("status") == "PASS"

    def val(key: str) -> str:
        return str(summary[key]) if ok_summary and summary is not None and key in summary else "unknown"

    lines = [
        f"junit_parse={'PASS' if ok_summary else 'FAIL'}",
        f"junit_tests={val('tests')}",
        f"junit_passed={val('passed')}",
        f"junit_failed={val('failed')}",
        f"junit_errors={val('errors')}",
        f"junit_skipped={val('skipped')}",
        f"junit_xfailed={val('xfailed')}",
        f"junit_suites={val('suites')}",
        f"junit_f07_operational_gate_nodes={val('f07_operational_gate_nodes')}",
        f"skip_disposition={'PASS' if ok_skip else 'FAIL'}",
        f"skip_disposition_records={skip['register_records'] if ok_skip and skip is not None else 'unknown'}",
        f"junit_evidence_status={'PASS' if ok_summary and ok_skip else 'FAIL'}",
    ]
    return lines


def render_register_markdown(register: tuple[SkipDispositionRecord, ...] = SKIP_DISPOSITION_REGISTER) -> str:
    out: list[str] = [
        "# EBH-7 — L3 skipped-test disposition register",
        "",
        "**Status:** individual dispositions READY for independent B6 review · **not** a B6 PASS.",
        "",
        "Source of truth: [`tests/qualification/ebh_7/_ebh7_skip_disposition_register.py`]"
        "(../../../../../tests/qualification/ebh_7/_ebh7_skip_disposition_register.py). "
        "Verified against the real L3 `junit.xml` by `scripts/qualification/ebh7_junit_evidence.py skip-disposition` "
        "(artifact `skip-disposition.json`): every skipped node must be registered, every evidence node must be "
        "PASS in the same JUnit, unregistered or stale skips fail the replay. No JUnit total is hardcoded.",
        "",
        "Disposition vocabulary: `ACCEPTED_NA` = approved `N/A — WITH EVIDENCE` (Path B; no live-provider test implied).",
        "",
        "| # | NODE_ID | MANDATORY | DISPOSITION | AUDIT_STATUS | SOURCE_SHA |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(register, 1):
        out.append(
            f"| {i} | `{r.node_id}` | {'YES' if r.mandatory else 'NO'} | {r.disposition.value} | "
            f"{r.audit_status.value} | `{r.source_sha[:12]}` |"
        )
    out.append("")
    for i, r in enumerate(register, 1):
        out.extend(
            [
                f"## {i}. `{r.scenario_id or 'EBH-7 W9'}`",
                "",
                f"- **NODE_ID:** `{r.node_id}`",
                f"- **SKIP_REASON:** {r.skip_reason}",
                f"- **Skip source:** `{r.skip_source}`",
                f"- **REQUIREMENT:** {r.requirement}",
                f"- **MANDATORY:** {'YES' if r.mandatory else 'NO'}",
                f"- **DISPOSITION:** {r.disposition.value} (`N/A — WITH EVIDENCE`)",
                "- **EVIDENCE (must be PASS in the same JUnit):**",
                *[f"  - `{n}`" for n in r.evidence_nodes],
                "- **Evidence documents:**",
                *[f"  - `{d}`" for d in r.evidence_documents],
                f"- **SOURCE_SHA:** `{r.source_sha}` (last commit touching the skip-bearing test file @ pre-reconciliation TEST `496016c0c`)",
                f"- **JUSTIFICATION:** {r.justification}",
                f"- **AUDIT_STATUS:** {r.audit_status.value}",
                "",
            ]
        )
    out.extend(
        [
            "## Non-claims",
            "",
            "B6 PASS, `INDEPENDENT_ACCEPTANCE_BASELINE_SHA`, OD-EBH7-08, EBH-7 CLOSED, EVOL-X, global FRZ PASS.",
            "",
        ]
    )
    return "\n".join(out)


def cmd_render_register(out: Path | None, check: bool) -> int:
    text = render_register_markdown()
    target = out or (_REPO / REGISTER_DOC_REL)
    if check:
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            print(f"register_doc=STALE {target}")
            return 1
        print("register_doc=PASS")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"register_doc=WRITTEN {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_sum = sub.add_parser("summary")
    p_sum.add_argument("--junit", required=True, type=Path)
    p_sum.add_argument("--out", type=Path)
    p_skip = sub.add_parser("skip-disposition")
    p_skip.add_argument("--junit", required=True, type=Path)
    p_skip.add_argument("--out", type=Path)
    p_meta = sub.add_parser("meta-lines")
    p_meta.add_argument("--summary", type=Path)
    p_meta.add_argument("--skip-disposition", type=Path, dest="skip_disposition")
    p_doc = sub.add_parser("render-register")
    p_doc.add_argument("--out", type=Path)
    p_doc.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.cmd == "summary":
        return cmd_summary(args.junit, args.out)
    if args.cmd == "skip-disposition":
        return cmd_skip_disposition(args.junit, args.out)
    if args.cmd == "meta-lines":
        print("\n".join(meta_lines(args.summary, args.skip_disposition)))
        return 0
    return cmd_render_register(args.out, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
