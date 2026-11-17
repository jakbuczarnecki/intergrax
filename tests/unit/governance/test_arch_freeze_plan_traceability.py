# © Artur Czarnecki. All rights reserved.

"""ARCH-FREEZE traceability validator amendment (T-AF2-VALIDATOR-AMEND, ticket T-IGX-050-B)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from scripts.governance import verify_arch_freeze_plan_traceability as v

# ruff: noqa: E501

pytestmark = pytest.mark.unit

_REAL_QUAL = v._QUAL
_QUAL_FILES = (
    v._CHECKLIST.name,
    v._MATRIX.name,
    v._WBS.name,
    v._GAPS.name,
    v._QUAL_PLAN.name,
    v._IIAC_PROTOCOL.name,
    "ARCH_FREEZE_PRE_ADMISSION_READINESS_REPORT.md",
    "IIAC_POLICY_ACTIVATION_RECORD.json",
)


class Sandbox:
    """A copy of the qualification files that the validator is pointed at."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, real: Path) -> Path:
        return self.root / real.name

    def read(self, real: Path) -> str:
        return self.path(real).read_text(encoding="utf-8")

    def write(self, real: Path, text: str) -> None:
        self.path(real).write_text(text, encoding="utf-8", newline="\n")

    def set_deps(self, task_id: str, deps: str) -> None:
        lines = self.read(v._WBS).split("\n")
        for i, line in enumerate(lines):
            if line.startswith(f"| {task_id} |"):
                parts = line.split("|")
                parts[7] = f" {deps} "
                lines[i] = "|".join(parts)
                self.write(v._WBS, "\n".join(lines))
                return
        raise AssertionError(f"{task_id} not in WBS")

    def add_row(self, task_id: str, deps: str = "[]", qualification: str = "CURRENTLY_OPEN") -> None:
        row = (
            f"| {task_id} | W0 |  |  | implementer | scripts/ | {deps} | action | tests | evidence | accept "
            f"| LOW | no | PLANNED | {qualification} |"
        )
        text = self.read(v._WBS)
        lines = text.split("\n")
        last = max(i for i, line in enumerate(lines) if line.startswith("| T-"))
        lines.insert(last + 1, row)
        self.write(v._WBS, "\n".join(lines))

    def drop_row(self, task_id: str) -> None:
        lines = [x for x in self.read(v._WBS).split("\n") if not x.startswith(f"| {task_id} |")]
        self.write(v._WBS, "\n".join(lines))


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sandbox:
    for name in _QUAL_FILES:
        src = _REAL_QUAL / name
        if src.exists():
            shutil.copy(src, tmp_path / name)
    monkeypatch.setattr(v, "_QUAL", tmp_path)
    for attr in ("_CHECKLIST", "_MATRIX", "_WBS", "_GAPS", "_QUAL_PLAN", "_IIAC_PROTOCOL", "_IIAC_ACTIVATION"):
        monkeypatch.setattr(v, attr, tmp_path / getattr(v, attr).name)
    return Sandbox(tmp_path)


def _errors() -> list[str]:
    return v.collect()[0]


# --------------------------------------------------------------------------------------
# Positive: current repository state and amended behaviour
# --------------------------------------------------------------------------------------


def test_current_repo_state_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert v.main([]) == 0
    out = capsys.readouterr().out
    assert "[OK]" in out
    assert "dag_self_test=OK" in out
    assert "FIND-PROD-REACH-DEFERRED" in out  # reported as INFO


def test_counters_are_computed_from_files() -> None:
    errors, stats = v.collect()
    assert errors == []
    statuses = v._parse_checklist()
    assert stats["frz"] == len(statuses)
    assert stats["open"] == sum(1 for s in statuses.values() if s == "OPEN")
    assert stats["passed"] == sum(1 for s in statuses.values() if s.startswith("PASS"))
    assert stats["passed"] + stats["open"] == stats["frz"]
    assert stats["wbs"] == stats["wbs_base"] + stats["wbs_extension"]
    assert stats["wbs_base"] == v.BASE_WBS_TASKS


def test_self_test_passes_and_covers_all_coord_prerequisites() -> None:
    assert v.dag_self_test() == []
    assert len(v.FRZ_FRZ02_PREREQ_TASKS) == 20
    labels = [label for label, _, _ in v._self_test_cases()]
    for prereq in v.FRZ_FRZ02_PREREQ_TASKS:
        assert f"COORD without {prereq}" in labels


def test_reference_graph_is_valid_and_acyclic() -> None:
    assert v._dag_errors(v._reference_graph()) == []


def test_task_id_regex_uses_word_boundary() -> None:
    assert v._extract_task_ids("[PT-Q11,T-AF2-VALIDATOR-AMEND,T-B1-INV]") == [
        "PT-Q11",
        "T-AF2-VALIDATOR-AMEND",
        "T-B1-INV",
    ]
    assert v._extract_task_ids("PT-Q11") == ["PT-Q11"]  # never T-Q11
    assert v._extract_task_ids("ENT-B6, ENT-AUDIT-X, OPT-1") == []  # no T- inside a longer word


def test_extension_rows_with_pt_and_t_af2_ids_are_accepted(sandbox: Sandbox) -> None:
    sandbox.add_row("T-AF2-NEW-NODE", "[T-B0-001]")
    sandbox.add_row("PT-Q11", "[T-AF2-NEW-NODE]")
    sandbox.set_deps("T-B1-INV", "[T-B0-SSOT-REFRESH,PT-Q11]")
    errors, stats = v.collect()
    assert errors == []
    assert stats["wbs"] == 56
    assert stats["wbs_base"] == 54
    assert stats["wbs_extension"] == 2


def test_pt_dependency_is_not_read_as_t_dependency(sandbox: Sandbox) -> None:
    sandbox.add_row("PT-Q11")
    sandbox.set_deps("T-B1-INV", "[T-B0-SSOT-REFRESH,PT-Q11]")
    assert v.collect()[0] == []
    # A PT- dependency without a PT- row is reported as PT-Q99, not as T-Q99.
    sandbox.set_deps("T-B1-INV", "[T-B0-SSOT-REFRESH,PT-Q99]")
    assert any("missing dependency PT-Q99" in e for e in _errors())


def test_superseded_with_authority_ref_is_satisfied() -> None:
    rows = [{"TASK_ID": "T-X", "V21_QUALIFICATION": "SUPERSEDED_BY_CURRENT_AUTHORITY - authority_ref: be77b3a5f"}]
    errors: list[str] = []
    assert v._check_qualification_states(rows, errors) == {"T-X"}
    assert errors == []


def test_qual_runs_are_computed(sandbox: Sandbox) -> None:
    text = sandbox.read(v._QUAL_PLAN).rstrip("\n")
    text += "\n| L1-EXTRA-RUN | L1 | trigger | cmd | env | scope | evidence | sha | reuse | failure | gate | owner |\n"
    sandbox.write(v._QUAL_PLAN, text)
    errors, stats = v.collect()
    assert errors == []
    assert stats["qual_runs"] == 17


def test_checklist_status_change_updates_counters_without_error(sandbox: Sandbox) -> None:
    before = v.collect()[1]
    text = sandbox.read(v._CHECKLIST)
    new = re.sub(r"(\| FRZ-BND-01 \|(?:[^|]*\|){3}) OPEN \|", r"\1 PASS |", text, count=1)
    assert new != text
    sandbox.write(v._CHECKLIST, new)
    errors, after = v.collect()
    assert errors == []
    assert after["open"] == before["open"] - 1  # type: ignore[operator]
    assert after["passed"] == before["passed"] + 1  # type: ignore[operator]


def test_info_finding_is_not_a_blocker() -> None:
    assert "FIND-PROD-REACH-DEFERRED" not in v.BLOCKER_FINDINGS
    assert "FIND-PROD-REACH-DEFERRED" in v.INFO_FINDINGS
    assert v._parse_findings()["FIND-PROD-REACH-DEFERRED"].upper() == "INFO"


def _align_matrix_with_checklist(sandbox: Sandbox) -> None:
    statuses = v._parse_checklist()
    out: list[str] = []
    for line in sandbox.read(v._MATRIX).split("\n"):
        if line.startswith("| FRZ-"):
            parts = line.split("|")
            parts[2] = f" {v._status_class(statuses[parts[1].strip()])} "
            line = "|".join(parts)
        out.append(line)
    text = "\n".join(out)
    families = len({f.split("-")[1] for f in statuses})
    text = re.sub(
        r"\*\*Rows:\*\* \d+ \| \*\*Families:\*\* \d+",
        f"**Rows:** {len(statuses)} | **Families:** {families}",
        text,
        count=1,
    )
    sandbox.write(v._MATRIX, text)


def test_crosscheck_is_opt_in_via_matrix_header_flag(sandbox: Sandbox) -> None:
    # Stale matrix status, no flag: the cross-check does not run (W0 state).
    text = sandbox.read(v._MATRIX).replace("| FRZ-BND-01 | OPEN |", "| FRZ-BND-01 | PASS |", 1)
    sandbox.write(v._MATRIX, text)
    errors, stats = v.collect()
    assert not any("matrix cross-check" in e for e in errors)
    assert stats["matrix_crosscheck"] == "OFF"


def test_crosscheck_enabled_and_consistent_passes(sandbox: Sandbox) -> None:
    _align_matrix_with_checklist(sandbox)
    flagged = sandbox.read(v._MATRIX).replace("\n\n", "\n\n**MATRIX_CHECKLIST_CROSSCHECK:** ENABLED\n\n", 1)
    sandbox.write(v._MATRIX, flagged)
    errors, stats = v.collect()
    assert stats["matrix_crosscheck"] == "ENABLED"
    assert not any("matrix cross-check" in e for e in errors)


# --------------------------------------------------------------------------------------
# Negative
# --------------------------------------------------------------------------------------


def test_crosscheck_enabled_detects_status_drift(sandbox: Sandbox) -> None:
    _align_matrix_with_checklist(sandbox)
    text = sandbox.read(v._MATRIX).replace("\n\n", "\n\n**MATRIX_CHECKLIST_CROSSCHECK:** ENABLED\n\n", 1)
    text = text.replace("| FRZ-BND-01 | OPEN |", "| FRZ-BND-01 | PASS |", 1)
    sandbox.write(v._MATRIX, text)
    assert any("matrix cross-check: FRZ-BND-01 status differs" in e for e in _errors())


def test_crosscheck_enabled_detects_missing_matrix_row(sandbox: Sandbox) -> None:
    _align_matrix_with_checklist(sandbox)
    lines = [x for x in sandbox.read(v._MATRIX).split("\n") if not x.startswith("| FRZ-BND-01 |")]
    text = "\n".join(lines).replace("\n\n", "\n\n**MATRIX_CHECKLIST_CROSSCHECK:** ENABLED\n\n", 1)
    sandbox.write(v._MATRIX, text)
    errs = _errors()
    assert any("FRZ-BND-01 is in the checklist but not in the matrix" in e for e in errs)


def test_crosscheck_enabled_detects_header_count_drift(sandbox: Sandbox) -> None:
    _align_matrix_with_checklist(sandbox)
    text = sandbox.read(v._MATRIX).replace("\n\n", "\n\n**MATRIX_CHECKLIST_CROSSCHECK:** ENABLED\n\n", 1)
    text = re.sub(r"\*\*Rows:\*\* \d+", "**Rows:** 224", text, count=1)
    sandbox.write(v._MATRIX, text)
    assert any("header declares 224 rows" in e for e in _errors())


def test_superseded_without_authority_ref_is_rejected(sandbox: Sandbox) -> None:
    text = sandbox.read(v._WBS)
    new = text.replace(" - authority_ref: be77b3a5f", "")
    assert new != text
    sandbox.write(v._WBS, new)
    assert any(
        "T-B5-EVOL-FORMAL-CLOSURE: SUPERSEDED_BY_CURRENT_AUTHORITY requires a non-empty authority_ref" in e
        for e in _errors()
    )


@pytest.mark.parametrize("ref", ["TBD", "pending", "Pending.", "`PENDING`", "?", "none", "N/A", "todo", "—", "unknown"])
def test_superseded_with_placeholder_authority_ref_is_rejected(ref: str) -> None:
    rows = [{"TASK_ID": "T-X", "V21_QUALIFICATION": f"SUPERSEDED_BY_CURRENT_AUTHORITY - authority_ref: {ref}"}]
    errors: list[str] = []
    assert v._check_qualification_states(rows, errors) == set()
    assert errors


def test_deleting_a_frz_row_fails(sandbox: Sandbox) -> None:
    lines = [x for x in sandbox.read(v._CHECKLIST).split(chr(10)) if not x.startswith("| FRZ-EXE-01 |")]
    sandbox.write(v._CHECKLIST, chr(10).join(lines))
    assert any("FRZ count expected 225 got 224" in e for e in _errors())


def test_deleting_a_qualification_run_fails(sandbox: Sandbox) -> None:
    lines = [x for x in sandbox.read(v._QUAL_PLAN).split(chr(10)) if not x.startswith("| L1-GOV |")]
    sandbox.write(v._QUAL_PLAN, chr(10).join(lines))
    assert any("qual RUN_ID count must be at least 16" in e for e in _errors())


def test_unknown_checklist_status_is_rejected(sandbox: Sandbox) -> None:
    text = sandbox.read(v._CHECKLIST)
    new = re.sub(r"(\| FRZ-BND-01 \|(?:[^|]*\|){3}) OPEN \|", r"\1 DONE-ISH |", text, count=1)
    assert new != text
    sandbox.write(v._CHECKLIST, new)
    assert any("unknown FRZ status 'DONE-ISH'" in e for e in _errors())


def test_historical_task_count_is_enforced(sandbox: Sandbox) -> None:
    sandbox.drop_row("T-B8-ECAP-16")
    assert any("WBS historical task count expected 54 got 53" in e for e in _errors())


def test_edge_evol_frz_after_ecap_2c_is_rejected_in_wbs(sandbox: Sandbox) -> None:
    sandbox.set_deps("T-B5-EVOL-FRZ", "[T-B5-EVOL-B6,T-B7-ECAP-2C]")
    errs = _errors()
    assert any("PD2-01: T-B5-EVOL-FRZ is a pre-lock ancestor and must not follow T-B7-ECAP-2C" in e for e in errs)
    assert any("dependency cycle" in e for e in errs)


def test_edge_evol_frz_after_ecap_2c_is_rejected_in_graph_fixture() -> None:
    graph = v._with_edge(v._reference_graph(), "T-B5-EVOL-FRZ", "T-B7-ECAP-2C")
    errs = v._dag_errors(graph)
    assert any("PD2-01: T-B5-EVOL-FRZ" in e for e in errs)


def test_dropping_a_coord_prerequisite_in_wbs_is_rejected(sandbox: Sandbox) -> None:
    lines = sandbox.read(v._WBS).split("\n")
    row = next(x for x in lines if x.startswith("| T-B3-DOC |"))
    assert row  # T-B3-DOC is a COORD prerequisite; isolate it from every other row
    out: list[str] = []
    for line in lines:
        if line.startswith("| T-") and not line.startswith("| T-B3-DOC |"):
            parts = line.split("|")
            parts[7] = parts[7].replace("T-B3-DOC,", "").replace(",T-B3-DOC", "").replace("T-B3-DOC", "")
            line = "|".join(parts)
        out.append(line)
    sandbox.write(v._WBS, "\n".join(out))
    assert any("T-B7-COORD-ARCH-FRZ must follow T-B3-DOC" in e for e in _errors())


def test_gap_register_info_severity_is_enforced(sandbox: Sandbox) -> None:
    text = sandbox.read(v._GAPS)
    new = text.replace("| FIND-PROD-REACH-DEFERRED | INFO |", "| FIND-PROD-REACH-DEFERRED | HIGH |", 1)
    assert new != text
    sandbox.write(v._GAPS, new)
    assert any("FIND-PROD-REACH-DEFERRED must have severity INFO" in e for e in _errors())


def test_blocker_findings_still_require_a_mapped_task() -> None:
    assert {"FIND-VBN-PARTIAL", "FIND-FORMAL-B6", "FIND-ACCEPTED-BASELINE", "FIND-PROD-REACH-CLASSIFICATION"} == set(
        v.BLOCKER_FINDINGS
    )


@pytest.mark.parametrize(
    ("label", "graph", "expected"),
    v._self_test_cases(),
    ids=[c[0] for c in v._self_test_cases()],
)
def test_every_self_test_mutation_is_rejected(label: str, graph: dict[str, list[str]], expected: str) -> None:
    assert any(expected in e for e in v._dag_errors(graph)), label


@pytest.mark.parametrize("flag", ["--sync-wbs", "--sync-matrix"])
def test_sync_modes_are_forbidden_and_do_not_touch_files(flag: str, capsys: pytest.CaptureFixture[str]) -> None:
    before = {p: p.read_bytes() for p in (v._WBS, v._MATRIX)}
    rc = v.main([flag])
    err = capsys.readouterr().err
    assert rc != 0
    assert "forbidden" in err
    assert "section C" in err
    assert before == {p: p.read_bytes() for p in (v._WBS, v._MATRIX)}


def test_both_sync_flags_together_are_forbidden(capsys: pytest.CaptureFixture[str]) -> None:
    assert v.main(["--sync-wbs", "--sync-matrix"]) != 0
    assert "forbidden" in capsys.readouterr().err


def test_self_test_cli_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert v.main(["--self-test"]) == 0
    assert "DAG self-test passed" in capsys.readouterr().out


def test_sync_functions_are_gone() -> None:
    assert not hasattr(v, "sync_wbs")
    assert not hasattr(v, "sync_matrix")
