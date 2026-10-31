# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E501

"""EBH-7 final B6 reconciliation: JUnit metadata parser, skip disposition integrity (read-only gates)."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
from tests.qualification.ebh_7._ebh7_skip_disposition_register import (
    SKIP_DISPOSITION_REGISTER,
    SkipAuditStatus,
    SkipDisposition,
)
from tests.qualification.ebh_7._semantic_disposition_registry import disposition_for

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts/qualification/ebh7_junit_evidence.py"
_WORKFLOW = _REPO / ".github/workflows/unit-tests.yml"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ebh7_junit_evidence_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_script()


def _split(node_id: str) -> tuple[str, str]:
    file_part, name = node_id.split("::", 1)
    return file_part[: -len(".py")].replace("/", "."), name


def _testcase(node_id: str, *, kind: str = "passed", message: str = "", skip_type: str = "pytest.skip") -> str:
    classname, name = _split(node_id)
    head = f'<testcase classname="{classname}" name="{name}" time="0.001"'
    if kind == "passed":
        return head + "/>"
    if kind == "skipped":
        return f'{head}><skipped type="{skip_type}" message="{message}">loc: {message}</skipped></testcase>'
    if kind == "xfailed":
        return f'{head}><skipped type="pytest.xfail" message="xfail"/></testcase>'
    if kind == "failed":
        return f'{head}><failure message="boom">trace</failure></testcase>'
    if kind == "error":
        return f'{head}><error message="boom">trace</error></testcase>'
    raise AssertionError(kind)


def _suite(cases: list[str], *, tests: int, failures: int = 0, errors: int = 0, skipped: int = 0) -> str:
    return (
        f'<testsuite name="pytest" errors="{errors}" failures="{failures}" skipped="{skipped}" '
        f'tests="{tests}" time="1.0">' + "".join(cases) + "</testsuite>"
    )


def _write(tmp_path: Path, xml: str, name: str = "junit.xml") -> Path:
    path = tmp_path / name
    path.write_text('<?xml version="1.0" encoding="utf-8"?>' + xml, encoding="utf-8")
    return path


def _filler(n: int) -> list[str]:
    return [_testcase(f"tests/qualification/ebh_7/test_ebh_7_filler.py::test_filler_{i}") for i in range(n)]


def _register_cases(*, evidence_pass: bool = True) -> list[str]:
    """9 skipped register nodes + all evidence nodes passed (590-case scale built by caller)."""
    cases = [_testcase(r.node_id, kind="skipped", message=r.skip_reason) for r in SKIP_DISPOSITION_REGISTER]
    evidence = sorted({n for r in SKIP_DISPOSITION_REGISTER for n in r.evidence_nodes})
    cases.extend(_testcase(n, kind="passed" if evidence_pass else "failed") for n in evidence)
    return cases


def _full_590(tmp_path: Path, *, wrap_in_testsuites: bool) -> Path:
    reg = _register_cases()
    filler = _filler(590 - len(reg))
    cases = reg + filler
    assert len(cases) == 590
    suite = _suite(cases, tests=590, skipped=9)
    xml = f'<testsuites name="pytest tests">{suite}</testsuites>' if wrap_in_testsuites else suite
    return _write(tmp_path, xml)


# --- 1/2/3: roots and 590/581/9 -------------------------------------------------------------------


def test_junit_root_testsuites_590_581_9(tmp_path: Path) -> None:
    report = ev.parse_junit(_full_590(tmp_path, wrap_in_testsuites=True))
    assert (report.root_tag, report.tests, report.passed, report.skipped) == ("testsuites", 590, 581, 9)
    assert (report.failed, report.errors, report.xfailed, report.suites) == (0, 0, 0, 1)


def test_junit_root_testsuite_590_581_9(tmp_path: Path) -> None:
    report = ev.parse_junit(_full_590(tmp_path, wrap_in_testsuites=False))
    assert (report.root_tag, report.tests, report.passed, report.skipped) == ("testsuite", 590, 581, 9)


def test_junit_testsuites_root_without_tests_attribute_is_not_zero(tmp_path: Path) -> None:
    """Regression for meta.txt junit_tests=0 (root <testsuites> carries no ``tests`` attribute)."""
    path = _full_590(tmp_path, wrap_in_testsuites=True)
    assert '<testsuites name="pytest tests">' in path.read_text(encoding="utf-8")
    assert ev.parse_junit(path).tests == 590


def test_junit_no_double_counting_with_declared_root_sums(tmp_path: Path) -> None:
    cases = _filler(5) + [
        _testcase("tests/qualification/ebh_7/test_ebh_7_filler.py::test_s", kind="skipped", message="m")
    ]
    suite = _suite(cases, tests=6, skipped=1)
    xml = f'<testsuites tests="6" failures="0" errors="0" skipped="1">{suite}</testsuites>'
    report = ev.parse_junit(_write(tmp_path, xml))
    assert (report.tests, report.passed, report.skipped) == (6, 5, 1)


def test_junit_two_leaf_suites_summed_once(tmp_path: Path) -> None:
    a = _suite(_filler(3), tests=3)
    b_cases = [
        _testcase("tests/qualification/ebh_7/test_ebh_7_other.py::test_a"),
        _testcase("tests/qualification/ebh_7/test_ebh_7_other.py::test_b", kind="skipped", message="x"),
    ]
    b = _suite(b_cases, tests=2, skipped=1)
    report = ev.parse_junit(_write(tmp_path, f"<testsuites>{a}{b}</testsuites>"))
    assert (report.tests, report.passed, report.skipped, report.suites) == (5, 4, 1, 2)


# --- classification --------------------------------------------------------------------------------


def test_junit_classifies_passed_failed_error_skipped_xfailed(tmp_path: Path) -> None:
    base = "tests/qualification/ebh_7/test_ebh_7_filler.py::"
    cases = [
        _testcase(base + "p"),
        _testcase(base + "f", kind="failed"),
        _testcase(base + "e", kind="error"),
        _testcase(base + "s", kind="skipped", message="m"),
        _testcase(base + "x", kind="xfailed"),
    ]
    # pytest counts xfail inside ``skipped``
    suite = _suite(cases, tests=5, failures=1, errors=1, skipped=2)
    report = ev.parse_junit(_write(tmp_path, f"<testsuites>{suite}</testsuites>"))
    assert (report.passed, report.failed, report.errors, report.skipped, report.xfailed) == (1, 1, 1, 1, 1)
    assert report.passed + report.failed + report.errors + report.skipped + report.xfailed == report.tests


def test_skipped_is_never_counted_as_passed(tmp_path: Path) -> None:
    cases = [_testcase("tests/qualification/ebh_7/test_ebh_7_filler.py::s", kind="skipped", message="m")]
    report = ev.parse_junit(_write(tmp_path, f"<testsuites>{_suite(cases, tests=1, skipped=1)}</testsuites>"))
    assert (report.passed, report.skipped) == (0, 1)


# --- 4/5: malformed + missing ----------------------------------------------------------------------


def test_junit_malformed_xml_fails(tmp_path: Path) -> None:
    path = _write(tmp_path, "<testsuites><testsuite></testsuites>")
    with pytest.raises(ev.JunitError, match="malformed"):
        ev.parse_junit(path)


def test_junit_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(ev.JunitError, match="missing"):
        ev.parse_junit(tmp_path / "absent.xml")


def test_junit_unsupported_root_fails(tmp_path: Path) -> None:
    with pytest.raises(ev.JunitError, match="unsupported"):
        ev.parse_junit(_write(tmp_path, "<report/>"))


def test_junit_zero_testcases_is_failure_not_success(tmp_path: Path) -> None:
    with pytest.raises(ev.JunitError, match="no testcase"):
        ev.parse_junit(
            _write(
                tmp_path,
                '<testsuites><testsuite name="pytest" tests="0" failures="0" errors="0" skipped="0"/></testsuites>',
            )
        )


def test_junit_testcase_outside_suite_fails(tmp_path: Path) -> None:
    with pytest.raises(ev.JunitError, match="outside"):
        ev.parse_junit(_write(tmp_path, f"<testsuites>{_filler(1)[0]}</testsuites>"))


# --- 6/7/8: inconsistent counters, hidden failures, repeated/omitted statistics --------------------


def test_junit_inconsistent_tests_counter_fails(tmp_path: Path) -> None:
    xml = f"<testsuites>{_suite(_filler(3), tests=4)}</testsuites>"
    with pytest.raises(ev.JunitError, match="inconsistent 'tests'"):
        ev.parse_junit(_write(tmp_path, xml))


def test_junit_inconsistent_skipped_counter_fails(tmp_path: Path) -> None:
    cases = [_testcase("tests/qualification/ebh_7/test_ebh_7_filler.py::s", kind="skipped", message="m")]
    with pytest.raises(ev.JunitError, match="inconsistent 'skipped'"):
        ev.parse_junit(_write(tmp_path, f"<testsuites>{_suite(cases, tests=1, skipped=0)}</testsuites>"))


def test_junit_failure_hidden_behind_clean_summary_fails(tmp_path: Path) -> None:
    cases = _filler(2) + [_testcase("tests/qualification/ebh_7/test_ebh_7_filler.py::f", kind="failed")]
    xml = f"<testsuites>{_suite(cases, tests=3, failures=0)}</testsuites>"
    with pytest.raises(ev.JunitError, match="inconsistent 'failures'"):
        ev.parse_junit(_write(tmp_path, xml))


def test_junit_error_hidden_behind_clean_summary_fails(tmp_path: Path) -> None:
    cases = _filler(2) + [_testcase("tests/qualification/ebh_7/test_ebh_7_filler.py::e", kind="error")]
    with pytest.raises(ev.JunitError, match="inconsistent 'errors'"):
        ev.parse_junit(_write(tmp_path, f"<testsuites>{_suite(cases, tests=3, errors=0)}</testsuites>"))


def test_junit_failed_and_error_reported_not_hidden(tmp_path: Path) -> None:
    cases = _filler(2) + [_testcase("tests/qualification/ebh_7/test_ebh_7_filler.py::f", kind="failed")]
    report = ev.parse_junit(_write(tmp_path, f"<testsuites>{_suite(cases, tests=3, failures=1)}</testsuites>"))
    assert (report.failed, report.passed) == (1, 2)


def test_junit_omitted_statistics_fail(tmp_path: Path) -> None:
    body = "".join(_filler(2))
    xml = f'<testsuites><testsuite name="pytest" tests="2" failures="0" skipped="0">{body}</testsuite></testsuites>'
    with pytest.raises(ev.JunitError, match="missing required statistic 'errors'"):
        ev.parse_junit(_write(tmp_path, xml))


def test_junit_repeated_testcases_fail(tmp_path: Path) -> None:
    one = _filler(1)[0]
    with pytest.raises(ev.JunitError, match="duplicate testcase"):
        ev.parse_junit(_write(tmp_path, f"<testsuites>{_suite([one, one], tests=2)}</testsuites>"))


def test_junit_duplicated_suite_block_fails(tmp_path: Path) -> None:
    suite = _suite(_filler(2), tests=2)
    with pytest.raises(ev.JunitError, match="duplicate testcase"):
        ev.parse_junit(_write(tmp_path, f"<testsuites>{suite}{suite}</testsuites>"))


def test_junit_root_declared_sum_mismatch_fails(tmp_path: Path) -> None:
    suite = _suite(_filler(2), tests=2)
    with pytest.raises(ev.JunitError, match="root testsuites inconsistent 'tests'"):
        ev.parse_junit(_write(tmp_path, f'<testsuites tests="4">{suite}</testsuites>'))


def test_junit_non_integer_statistic_fails(tmp_path: Path) -> None:
    body = "".join(_filler(1))
    xml = (
        f'<testsuites><testsuite name="p" tests="x" failures="0" errors="0" skipped="0">{body}</testsuite></testsuites>'
    )
    with pytest.raises(ev.JunitError, match="not an integer"):
        ev.parse_junit(_write(tmp_path, xml))


# --- CLI + meta lines: never zero-on-error ---------------------------------------------------------


def test_cli_summary_and_meta_lines_for_590(tmp_path: Path) -> None:
    junit = _full_590(tmp_path, wrap_in_testsuites=True)
    summary, skip = tmp_path / "s.json", tmp_path / "k.json"
    assert ev.main(["summary", "--junit", str(junit), "--out", str(summary)]) == 0
    assert ev.main(["skip-disposition", "--junit", str(junit), "--out", str(skip)]) == 0
    lines = dict(ln.split("=", 1) for ln in ev.meta_lines(summary, skip))
    assert lines["junit_tests"] == "590"
    assert lines["junit_passed"] == "581"
    assert lines["junit_skipped"] == "9"
    assert lines["junit_failed"] == "0" and lines["junit_errors"] == "0" and lines["junit_xfailed"] == "0"
    assert lines["junit_parse"] == "PASS" and lines["skip_disposition"] == "PASS"
    assert lines["junit_evidence_status"] == "PASS"


def test_cli_summary_failure_exit_nonzero_and_meta_not_zero(tmp_path: Path) -> None:
    bad = _write(tmp_path, "<testsuites><testsuite>")
    summary, skip = tmp_path / "s.json", tmp_path / "k.json"
    assert ev.main(["summary", "--junit", str(bad), "--out", str(summary)]) == 1
    assert ev.main(["skip-disposition", "--junit", str(bad), "--out", str(skip)]) == 1
    lines = dict(ln.split("=", 1) for ln in ev.meta_lines(summary, skip))
    assert lines["junit_parse"] == "FAIL" and lines["junit_evidence_status"] == "FAIL"
    assert lines["junit_tests"] == "unknown" and lines["junit_skipped"] == "unknown"


def test_meta_lines_without_inputs_is_fail_unknown(tmp_path: Path) -> None:
    lines = dict(ln.split("=", 1) for ln in ev.meta_lines(tmp_path / "none.json", None))
    assert lines["junit_tests"] == "unknown"
    assert lines["junit_evidence_status"] == "FAIL"
    assert "0" not in {lines[k] for k in ("junit_tests", "junit_passed", "junit_skipped")}


# --- skip disposition verification (real register against synthetic JUnit) -------------------------


def test_skip_disposition_passes_for_exact_register(tmp_path: Path) -> None:
    report = ev.parse_junit(_full_590(tmp_path, wrap_in_testsuites=True))
    result = ev.verify_skip_disposition(report)
    assert result["status"] == "PASS", result["problems"]
    assert result["skipped_in_junit"] == result["register_records"] == len(SKIP_DISPOSITION_REGISTER)
    assert all(r["AUDIT_STATUS"] == "READY" for r in result["records"])


def test_skip_disposition_unregistered_skip_fails(tmp_path: Path) -> None:
    cases = _register_cases() + [
        _testcase(
            "tests/qualification/ebh_7/test_ebh_7_w9_dossier_gates.py::test_surprise", kind="skipped", message="why"
        ),
    ]
    suite = _suite(cases, tests=len(cases), skipped=10)
    report = ev.parse_junit(_write(tmp_path, f"<testsuites>{suite}</testsuites>"))
    result = ev.verify_skip_disposition(report)
    assert result["status"] == "FAIL"
    assert result["unregistered_skips"] == ["tests/qualification/ebh_7/test_ebh_7_w9_dossier_gates.py::test_surprise"]


def test_skip_disposition_evidence_failed_fails(tmp_path: Path) -> None:
    cases = _register_cases(evidence_pass=False)
    suite = _suite(cases, tests=len(cases), failures=len(cases) - 9, skipped=9)
    report = ev.parse_junit(_write(tmp_path, f"<testsuites>{suite}</testsuites>"))
    result = ev.verify_skip_disposition(report)
    assert result["status"] == "FAIL"
    assert any("evidence node not PASS" in p for p in result["problems"])


def test_skip_disposition_evidence_missing_fails(tmp_path: Path) -> None:
    cases = [_testcase(r.node_id, kind="skipped", message=r.skip_reason) for r in SKIP_DISPOSITION_REGISTER]
    suite = _suite(cases, tests=len(cases), skipped=9)
    report = ev.parse_junit(_write(tmp_path, f"<testsuites>{suite}</testsuites>"))
    result = ev.verify_skip_disposition(report)
    assert result["status"] == "FAIL"
    assert any("MISSING" in p for p in result["problems"])


def test_skip_disposition_reason_mismatch_fails(tmp_path: Path) -> None:
    cases = _register_cases()
    first = SKIP_DISPOSITION_REGISTER[0]
    cases[0] = _testcase(first.node_id, kind="skipped", message="some other reason")
    suite = _suite(cases, tests=len(cases), skipped=9)
    report = ev.parse_junit(_write(tmp_path, f"<testsuites>{suite}</testsuites>"))
    result = ev.verify_skip_disposition(report)
    assert result["status"] == "FAIL"
    assert any("skip reason mismatch" in p for p in result["problems"])


def test_skip_disposition_stale_register_record_fails(tmp_path: Path) -> None:
    cases = _register_cases()
    dropped = SKIP_DISPOSITION_REGISTER[0]
    cases[0] = _testcase(dropped.node_id, kind="passed")  # no longer skipped
    suite = _suite(cases, tests=len(cases), skipped=8)
    report = ev.parse_junit(_write(tmp_path, f"<testsuites>{suite}</testsuites>"))
    result = ev.verify_skip_disposition(report)
    assert result["status"] == "FAIL"
    assert result["stale_records"] == [dropped.node_id]


def test_skip_disposition_blocked_or_unready_record_fails(tmp_path: Path) -> None:
    report = ev.parse_junit(_full_590(tmp_path, wrap_in_testsuites=True))
    blocked = replace(SKIP_DISPOSITION_REGISTER[0], disposition=SkipDisposition.BLOCKED)
    unready = replace(SKIP_DISPOSITION_REGISTER[1], audit_status=SkipAuditStatus.BLOCKED)
    register = (blocked, unready) + SKIP_DISPOSITION_REGISTER[2:]
    result = ev.verify_skip_disposition(report, register)
    assert result["status"] == "FAIL"


def test_skip_disposition_mandatory_without_evidence_fails(tmp_path: Path) -> None:
    report = ev.parse_junit(_full_590(tmp_path, wrap_in_testsuites=True))
    no_evidence = replace(SKIP_DISPOSITION_REGISTER[0], evidence_nodes=())
    result = ev.verify_skip_disposition(report, (no_evidence,) + SKIP_DISPOSITION_REGISTER[1:])
    assert result["status"] == "FAIL"
    assert any("mandatory skip without alternative evidence" in p for p in result["problems"])


# --- static register integrity (repository @ HEAD) -------------------------------------------------


def _function_defined(file_rel: str, name: str) -> bool:
    tree = ast.parse((_REPO / file_rel).read_text(encoding="utf-8"))
    return any(isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == name for n in ast.walk(tree))


def _split_node(node: str) -> tuple[str, str]:
    file_rel, name = node.split("::", 1)
    return file_rel, re.sub(r"\[.*\]$", "", name)


def test_register_has_nine_individual_unique_records() -> None:
    nodes = [r.node_id for r in SKIP_DISPOSITION_REGISTER]
    assert len(nodes) == len(set(nodes)) == 9


def test_register_records_are_complete_and_ready() -> None:
    for r in SKIP_DISPOSITION_REGISTER:
        assert r.disposition is SkipDisposition.ACCEPTED_NA, r.node_id
        assert r.audit_status is SkipAuditStatus.READY, r.node_id
        assert re.fullmatch(r"[0-9a-f]{40}", r.source_sha), r.node_id
        assert r.skip_reason.strip() and r.justification.strip() and r.requirement.strip(), r.node_id
        assert "Path B" in r.justification or "no obligation is dropped" in r.justification.lower(), r.node_id
        if r.mandatory:
            assert r.evidence_nodes, r.node_id
        assert r.evidence_documents, r.node_id


def test_register_node_ids_resolve_to_real_tests_and_skip_statements() -> None:
    for r in SKIP_DISPOSITION_REGISTER:
        file_rel, fn = _split_node(r.node_id)
        assert _function_defined(file_rel, fn), r.node_id
        source = (_REPO / file_rel).read_text(encoding="utf-8")
        assert "pytest.skip(" in source, r.node_id
        # reason literal present exactly as the skip message (first segment before any f-string join)
        assert r.skip_reason in source, f"skip reason not found in source for {r.node_id}"
        line = int(r.skip_source.split(":")[1].split(" ")[0])
        assert "pytest.skip(" in source.splitlines()[line - 1], f"skip site line mismatch for {r.node_id}"


def test_register_evidence_nodes_exist_and_are_not_themselves_skipped() -> None:
    skipped = {r.node_id for r in SKIP_DISPOSITION_REGISTER}
    for r in SKIP_DISPOSITION_REGISTER:
        for ev_node in r.evidence_nodes:
            file_rel, fn = _split_node(ev_node)
            assert _function_defined(file_rel, fn), ev_node
            assert ev_node not in skipped
            assert "pytest.skip(" not in _function_source(file_rel, fn), ev_node


def _function_source(file_rel: str, name: str) -> str:
    text = (_REPO / file_rel).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == name:
            return ast.get_source_segment(text, n) or ""
    raise AssertionError(name)


def test_register_scenarios_are_qualified_and_equivalents_match_semantic_ssot() -> None:
    for r in SKIP_DISPOSITION_REGISTER:
        if r.scenario_id is None:
            continue
        d = disposition_for(r.scenario_id)
        assert d is not None and d.executed_equivalent_node_id, r.scenario_id
        assert d.executed_equivalent_node_id in r.evidence_nodes, r.scenario_id


def test_register_covers_every_routed_skip_path_in_gate_sources() -> None:
    """Every conditional pytest.skip in the W6/W7/W9 gates has a register record (no silent routes)."""
    routed_scenarios = {r.scenario_id for r in SKIP_DISPOSITION_REGISTER if r.scenario_id}
    from tests.qualification.ebh_7._cross_domain_scenarios import cross_domain_qualified_scenarios
    from tests.qualification.ebh_7._tenant_e2e_scenarios import tenant_e2e_qualified_scenarios

    expected: set[str] = {"EBH7-TEN-E2E-04", "EBH7-TEN-E2E-11"}
    expected |= {s.scenario_id for s in tenant_e2e_qualified_scenarios() if s.proof_needs_tmp_path}
    expected |= {s.scenario_id for s in cross_domain_qualified_scenarios() if s.proof_needs_tmp_path}
    assert routed_scenarios == expected


def test_register_document_matches_renderer() -> None:
    assert ev.render_register_markdown() == (_REPO / ev.REGISTER_DOC_REL).read_text(encoding="utf-8")


def test_register_document_render_is_read_only_to_dossier(tmp_path: Path) -> None:
    out = tmp_path / "REGISTER.md"
    assert ev.cmd_render_register(out, check=False) == 0
    assert out.read_text(encoding="utf-8") == ev.render_register_markdown()
    assert ev.cmd_render_register(out, check=True) == 0
    out.write_text("tampered", encoding="utf-8")
    assert ev.cmd_render_register(out, check=True) == 1


def test_register_document_states_non_claims() -> None:
    text = (_REPO / ev.REGISTER_DOC_REL).read_text(encoding="utf-8")
    assert "not** a B6 PASS" in text and "Non-claims" in text


# --- workflow guard --------------------------------------------------------------------------------


def test_workflow_uses_strict_junit_evidence_not_legacy_parser() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "r.get('tests','0')" not in text, "legacy root-attribute JUnit parser must not return"
    assert "ebh7_junit_evidence.py" in text
    for token in ("summary --junit", "skip-disposition --junit", "meta-lines", "skip-disposition.json"):
        assert token in text, token
    assert (
        "grep -c 'test_ebh_5_dual_slug_rpl_e_operational_gates.py' build/qualification/ebh7-l3/pytest.log" not in text
    )


def test_workflow_final_status_fails_on_junit_or_skip_disposition() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "junit-summary.exit" in text and "skip-disposition.exit" in text
