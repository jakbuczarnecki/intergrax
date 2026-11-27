# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_BUILD = _REPO / "scripts/governance/build_rw_plan.py"
_PIN = "c" * 40


def _import_build():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_rw_plan", _BUILD)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_PAD = "| x" * 7


def _fixture_row(frz: str, meaning: str, neg: str, fix: str, cert: str) -> str:
    return f"| `{frz}` | {meaning} {_PAD} | ENT-A | {neg} | {fix} | {cert} | x |"


APPENDIX_FIXTURE = (
    "## Appendix A. fixture\n"
    + _fixture_row(
        "FRZ-BND-04",
        "private implementation leakage = 0",
        "NT-BND-04",
        "CODE_FIX: Appendix C, T-IGX-043-G",
        "RW ticket T-IGX-043-G",
    )
    + "\n"
    + _fixture_row(
        "FRZ-BND-05",
        "cross-layer construction",
        "NEG-0",
        "none (requalification only)",
        "PT-WA",
    )
    + "\n"
    + _fixture_row(
        "FRZ-OWN-01",
        "duplicate one",
        "NT-OWN-01",
        "CODE_FIX: Appendix C, T-IGX-050-D",
        "RW ticket T-IGX-050-D",
    )
    + "\n"
    + _fixture_row(
        "FRZ-OWN-02",
        "duplicate two",
        "NT-OWN-02",
        "CODE_FIX: Appendix C, T-IGX-050-D",
        "RW ticket T-IGX-051-A",
    )
    + "\n"
)


def _find_af2_entries() -> list[dict[str, object]]:
    return [{"id": f"FIND-AF2-{n:02d}", "mapping": "node", "in_train": False} for n in range(1, 24)]


def _minimal_bindings(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema": "rw_plan_bindings.v1",
        "read_pin": _PIN,
        "red_first_record": None,
        "md_notes": [],
        "tickets": {},
        "lanes": [],
        "serialization_edges": [],
        "code_fix_rows": {},
        "record_corrections": {},
        "coverage_overrides": {},
        "find_af2": _find_af2_entries(),
        "path_b_invariants": ["invariant"],
        "g2_entry_conditions": ["condition"],
        "dependencies": {"W2": ["x"], "W3": ["SCOPE"]},
        "implementation_disposition": {"published_on_development": ["x"]},
    }
    data.update(overrides)
    return data


def _prepare_build(
    mod,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: list,
    prior_rows: list[dict[str, object]] | None = None,
    bindings: dict[str, object] | None = None,
) -> None:
    rw_json = tmp_path / "rw_plan.v1.json"
    if prior_rows is not None:
        rw_json.write_text(json.dumps({"code_fix_inventory": prior_rows}), encoding="utf-8")
    bindings_path = tmp_path / "bindings.json"
    bindings_path.write_text(json.dumps(bindings or _minimal_bindings()), encoding="utf-8")
    monkeypatch.setattr(mod, "_RW_PLAN_JSON", rw_json)
    monkeypatch.setattr(mod, "_BINDINGS_JSON", bindings_path)
    monkeypatch.setattr(mod, "_pin_tree", lambda pin: frozenset({"tests/unit/test_present.py"}))
    monkeypatch.setattr(mod, "parse_appendix_a_code_fix_inventory", lambda: list(rows))


def _row(mod, frz: str, *, negative_test: str | None = None, owner: str | None = "T-IGX-043-G"):
    return mod.CodeFixRow(
        frz_id=frz,
        summary="s",
        negative_test=negative_test or f"NT-{frz.removeprefix('FRZ-')}",
        remediation_track="CODE_FIX:",
        owner_ticket=owner,
        regression_scope="ENT-A",
    )


def test_code_fix_extraction_and_summary():
    mod = _import_build()
    rows = mod.parse_appendix_a_code_fix_inventory(APPENDIX_FIXTURE)
    by_id = {r.frz_id: r for r in rows}
    assert set(by_id) == {"FRZ-BND-04", "FRZ-OWN-01", "FRZ-OWN-02"}
    assert by_id["FRZ-BND-04"].summary == "private implementation leakage = 0"
    assert "FRZ-BND-05" not in by_id


def test_non_code_fix_rows_excluded():
    mod = _import_build()
    rows = mod.parse_appendix_a_code_fix_inventory(APPENDIX_FIXTURE)
    assert all(r.remediation_track.startswith("CODE_FIX") for r in rows)


def test_duplicate_code_fix_rejected():
    mod = _import_build()
    conflict = APPENDIX_FIXTURE.replace(
        "FRZ-OWN-02` | duplicate two",
        "FRZ-OWN-01` | duplicate two",
    )
    with pytest.raises(SystemExit):
        mod.parse_appendix_a_code_fix_inventory(conflict)


def test_explicit_t050d_owner_valid():
    mod = _import_build()
    rows = mod.parse_appendix_a_code_fix_inventory(APPENDIX_FIXTURE)
    assert any(r.owner_ticket == "T-IGX-050-D" for r in rows)


def test_no_implicit_t050d_fallback():
    mod = _import_build()
    section = """\
## Appendix A. x
| `FRZ-X-01` | meaning | x | x | x | x | x | x | x | ENT-A | NT-X-01 | CODE_FIX: Appendix C | PT-WA only | x |
"""
    rows = mod.parse_appendix_a_code_fix_inventory(section)
    assert len(rows) == 1
    assert rows[0].owner_ticket is None


def test_delta_findings_come_from_pass_condition_column():
    mod = _import_build()
    cells = ["x"] * 18
    cells[0] = "`FRZ-X-01`"
    cells[10] = "NT-X-01"
    cells[11] = "CODE_FIX: Appendix C"
    cells[13] = "operator column A9-999"
    cells[14] = "delta A4-012 and A5-001"
    section = "## Appendix A. x\n| " + " | ".join(cells) + " |\n"
    rows = mod.parse_appendix_a_code_fix_inventory(section)
    assert rows[0].delta_findings == ["A4-012", "A5-001"]


def test_build_red_first_null_not_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    _prepare_build(
        mod,
        tmp_path,
        monkeypatch,
        rows=[_row(mod, "FRZ-BND-04")],
        prior_rows=[{"frz_id": "FRZ-BND-04", "red_first_status": "EVIDENCE_MISSING", "red_first_prefix_sha": head}],
    )
    inv = mod.build()["code_fix_inventory"][0]
    assert inv["red_first_prefix_sha"] is None
    assert inv["red_first_status"] == "EVIDENCE_MISSING"


def test_preserve_red_at_prefix_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    _prepare_build(
        mod,
        tmp_path,
        monkeypatch,
        rows=[_row(mod, "FRZ-BND-04")],
        prior_rows=[
            {
                "frz_id": "FRZ-BND-04",
                "red_first_status": "RED_AT_PREFIX",
                "red_first_prefix_sha": "a" * 40,
                "red_first_evidence_ref": "docs/x.json#FRZ-BND-04",
            }
        ],
    )
    inv = mod.build()["code_fix_inventory"][0]
    assert inv["red_first_status"] == "RED_AT_PREFIX"
    assert inv["red_first_prefix_sha"] == "a" * 40


def test_preserve_red_first_unreproducible_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    _prepare_build(
        mod,
        tmp_path,
        monkeypatch,
        rows=[_row(mod, "FRZ-CTR-01")],
        prior_rows=[
            {
                "frz_id": "FRZ-CTR-01",
                "red_first_status": "RED_FIRST_UNREPRODUCIBLE",
                "red_first_prefix_sha": "b" * 40,
                "red_first_evidence_ref": "docs/x.json#FRZ-CTR-01",
                "red_first_reason": "strict exit 2",
            }
        ],
    )
    inv = mod.build()["code_fix_inventory"][0]
    assert inv["red_first_status"] == "RED_FIRST_UNREPRODUCIBLE"
    assert inv["red_first_reason"] == "strict exit 2"


@pytest.mark.parametrize("status", ["RED_FIRST_OUT", "RED_FIRST_LATER_WAVE"])
def test_regeneration_preserves_operator_dispositions(status: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    prior = {
        "frz_id": "FRZ-PLG-02",
        "red_first_status": status,
        "red_first_prefix_sha": "d" * 40,
        "red_first_evidence_ref": "docs/x.json#FRZ-PLG-02",
        "red_first_reason": "operator disposition",
        "red_first_historical_status": "RED_FIRST_UNREPRODUCIBLE",
        "operator_citation": "e" * 40,
        "operator_packet_ref": "docs/packet.md",
    }
    _prepare_build(mod, tmp_path, monkeypatch, rows=[_row(mod, "FRZ-PLG-02")], prior_rows=[prior])
    inv = mod.build()["code_fix_inventory"][0]
    for key, value in prior.items():
        assert inv[key] == value, key


def _write_record(tmp_path: Path, rows: list[dict[str, object]]) -> str:
    path = tmp_path / "RED_FIRST_RECORD.json"
    path.write_text(json.dumps({"schema": "af2.red_first_ga2r.v1", "rows": rows}), encoding="utf-8")
    return path.as_posix()


def test_red_first_record_supersedes_prior_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    record = _write_record(
        tmp_path,
        [{"frz_id": "FRZ-BND-04", "status": "EVIDENCE_MISSING", "reason": "not behavioural", "due_ticket": "T-X"}],
    )
    _prepare_build(
        mod,
        tmp_path,
        monkeypatch,
        rows=[_row(mod, "FRZ-BND-04")],
        prior_rows=[
            {
                "frz_id": "FRZ-BND-04",
                "red_first_status": "RED_AT_PREFIX",
                "red_first_prefix_sha": "a" * 40,
                "red_first_evidence_ref": "docs/old.json#FRZ-BND-04",
            }
        ],
        bindings=_minimal_bindings(red_first_record=record),
    )
    inv = mod.build()["code_fix_inventory"][0]
    assert inv["red_first_status"] == "EVIDENCE_MISSING"
    assert inv["red_first_prefix_sha"] is None
    assert inv["red_first_reason"] == "not behavioural"
    assert inv["red_first_due_ticket"] == "T-X"
    assert inv["red_first_superseded_ref"] == "docs/old.json#FRZ-BND-04"
    assert inv["red_first_evidence_ref"].endswith("#FRZ-BND-04")


def test_red_first_record_cannot_change_operator_disposition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    record = _write_record(
        tmp_path,
        [{"frz_id": "FRZ-PLG-02", "status": "RED_AT_PREFIX", "red_first_prefix_sha": "a" * 40, "reason": "r"}],
    )
    _prepare_build(
        mod,
        tmp_path,
        monkeypatch,
        rows=[_row(mod, "FRZ-PLG-02")],
        prior_rows=[
            {
                "frz_id": "FRZ-PLG-02",
                "red_first_status": "RED_FIRST_OUT",
                "red_first_prefix_sha": "d" * 40,
                "red_first_evidence_ref": "docs/x.json#FRZ-PLG-02",
            }
        ],
        bindings=_minimal_bindings(red_first_record=record),
    )
    with pytest.raises(SystemExit):
        mod.build()


def test_build_records_fresh_identity_and_read_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    _prepare_build(mod, tmp_path, monkeypatch, rows=[_row(mod, "FRZ-BND-04")])
    payload = mod.build()
    assert payload["execution_sha"] == head
    assert payload["read_pin"] == _PIN
    generated = datetime.fromisoformat(str(payload["generated_at"]))
    assert datetime.now(UTC) - generated < timedelta(minutes=5)


def test_regression_scope_source_names_real_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    bindings = _minimal_bindings(
        code_fix_rows={"FRZ-BND-04": {"regression_tests": ["tests/unit/test_present.py"]}},
    )
    _prepare_build(mod, tmp_path, monkeypatch, rows=[_row(mod, "FRZ-BND-04")], bindings=bindings)
    inv = mod.build()["code_fix_inventory"][0]
    assert inv["regression_scope"] == "ENT-A"
    assert inv["regression_scope_source"][0] == "matrix_appendix_a_tests"
    assert "rw_plan_bindings" in inv["regression_scope_source"]
    assert inv["regression_command"] == "uv run pytest tests/unit/test_present.py -q"


def test_regression_test_missing_at_pin_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    bindings = _minimal_bindings(
        code_fix_rows={"FRZ-BND-04": {"regression_tests": ["tests/unit/test_absent.py"]}},
    )
    _prepare_build(mod, tmp_path, monkeypatch, rows=[_row(mod, "FRZ-BND-04")], bindings=bindings)
    with pytest.raises(SystemExit):
        mod.build()


@pytest.mark.parametrize("marker", ["NEG-0", "NEG-P"])
def test_class_marker_requires_named_negative_test_binding(
    marker: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mod = _import_build()
    _prepare_build(mod, tmp_path, monkeypatch, rows=[_row(mod, "FRZ-BND-04", negative_test=marker)])
    with pytest.raises(SystemExit):
        mod.build()


def test_named_negative_test_replaces_class_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    bindings = _minimal_bindings(
        code_fix_rows={
            "FRZ-BND-04": {
                "negative_tests": [
                    {
                        "nt_id": "NT-Z-01",
                        "node": "tests/unit/test_present.py::test_a",
                        "mechanism": "m",
                        "owner_ticket": "T-IGX-043-G",
                    },
                    {
                        "nt_id": "NT-Z-02",
                        "node": "tests/unit/test_absent.py",
                        "mechanism": "m",
                        "owner_ticket": "T-IGX-043-G",
                    },
                ]
            }
        }
    )
    _prepare_build(mod, tmp_path, monkeypatch, rows=[_row(mod, "FRZ-BND-04", negative_test="NEG-0")], bindings=bindings)
    inv = mod.build()["code_fix_inventory"][0]
    assert inv["negative_test_id"] == "NT-Z-01"
    assert inv["matrix_negative_test_marker"] == "NEG-0"
    assert [t["file_state_at_pin"] for t in inv["negative_tests"]] == ["exists", "absent"]


def test_find_af2_mapping_requires_all_23(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    bindings = _minimal_bindings(find_af2=_find_af2_entries()[:-1])
    _prepare_build(mod, tmp_path, monkeypatch, rows=[_row(mod, "FRZ-BND-04")], bindings=bindings)
    with pytest.raises(SystemExit):
        mod.build()


def _matrix_row(frz: str, status: str, fix: str, cert: str, owner: str = "EBH / accountable: A") -> str:
    cells = ["x"] * 18
    cells[0] = f"`{frz}`"
    cells[2] = owner
    cells[6] = status
    cells[11] = fix
    cells[12] = cert
    return "| " + " | ".join(cells) + " |"


def test_coverage_derives_every_open_row_outside_inventory():
    mod = _import_build()
    section = "\n".join(
        [
            "## Appendix A. x",
            _matrix_row("FRZ-A-01", "OPEN", "DOC_PATCH", "PT-D01, T-B7-A; row nodes: PT-D02 (+ audit)"),
            _matrix_row("FRZ-A-02", "OPEN", "CODE_FIX: x", "PT-D01"),
            _matrix_row("FRZ-A-03", "PASS", "none", "PT-D01"),
            _matrix_row("FRZ-A-04", "OPEN", "none", "PT-D03", owner="EBH only"),
        ]
    )
    coverage = mod.build_coverage(mod.parse_appendix_a_all_rows(section), {"FRZ-A-02"}, {})
    assert [c["frz_id"] for c in coverage] == ["FRZ-A-01", "FRZ-A-04"]
    assert coverage[0]["reason"] == "DOC_PATCH"
    assert coverage[0]["owner_nodes"] == ["PT-D01", "T-B7-A", "PT-D02"]
    assert coverage[1]["accountable"] is None


def test_coverage_fails_without_owning_node():
    mod = _import_build()
    section = "## Appendix A. x\n" + _matrix_row("FRZ-A-01", "OPEN", "DOC_PATCH", "")
    with pytest.raises(SystemExit):
        mod.build_coverage(mod.parse_appendix_a_all_rows(section), set(), {})


_PLAN_FIXTURE = """\
# Plan

## 4. Exact write set

Counts: see lists. Cites `scripts/qualification/cited.py@H:20-37` and `docs/x.md:12`.

```text
{fenced}
pyproject.toml
.github/workflows/unit-tests.yml
intergrax/llm_adapters/providers/*   (glob note mentioning intergrax/not/included.py)
```

| Path | Kind |
|---|---|
| `tests/unit/test_table.py` | edited, see `scripts/other/mentioned.py` |
| Area | `governance/tickets/T-X.md` |

- `tests/unit/test_bullet_a.py`, `tests/unit/test_bullet_b.py` (retargeted; see `scripts/other/cited_in_note.py`)

Docs are described in prose: `docs/project/architecture/SKILLS.md`.

**Forbidden:** every other path, in particular `tests/unit/test_forbidden.py`.

### 3c. Out of write set

- `tests/unit/test_out.py`

### D - documents

```text
docs/excluded.md
```

## 5. Next

`tests/unit/test_next_section.py`
"""


def test_write_set_extraction_is_section_based_and_untruncated():
    mod = _import_build()
    fenced = "\n".join(f"tests/unit/test_sweep_{n}.py" for n in range(120))
    spec = {"sections": [{"heading": "## 4. Exact write set"}], "exclude_headings": ["### D - documents"]}
    result = mod.extract_write_set(_PLAN_FIXTURE.format(fenced=fenced), spec)
    paths = set(result["paths"])
    assert len([p for p in paths if "test_sweep_" in p]) == 120
    assert paths - {f"tests/unit/test_sweep_{n}.py" for n in range(120)} == {
        "pyproject.toml",
        ".github/workflows/unit-tests.yml",
        "intergrax/llm_adapters/providers/*",
        "tests/unit/test_table.py",
        "governance/tickets/T-X.md",
        "tests/unit/test_bullet_a.py",
        "tests/unit/test_bullet_b.py",
    }
    assert result["prose_mentions"] == ["docs/project/architecture/SKILLS.md"]


def test_write_set_extra_paths_and_expected_count():
    mod = _import_build()
    spec = {
        "sections": [{"heading": "## 5. Next"}],
        "extra_paths": [{"path": "tests/unit/test_next_section.py", "reason": "listed in prose"}],
        "expected_count": 1,
    }
    assert mod.extract_write_set(_PLAN_FIXTURE.format(fenced=""), spec)["paths"] == ["tests/unit/test_next_section.py"]
    spec["expected_count"] = 2
    with pytest.raises(SystemExit):
        mod.extract_write_set(_PLAN_FIXTURE.format(fenced=""), spec)


@pytest.mark.parametrize(
    ("plan", "heading"),
    [
        ("## A\n\n```\ntests/a.py\n```\n", "## Missing"),
        ("## A\n\nonly prose\n", "## A"),
        ("## A\n\n```\ntests/a.py\n```\n\n## A\n\n```\ntests/b.py\n```\n", "## A"),
    ],
)
def test_write_set_fails_closed(plan: str, heading: str):
    mod = _import_build()
    with pytest.raises(SystemExit):
        mod.extract_write_set(plan, {"sections": [{"heading": heading}]})


def test_write_set_expands_brace_shorthand():
    mod = _import_build()
    plan = (
        "## A\n\n```text\n"
        "intergrax/scaffold/{new_application, new_product}.py   (generated hosts)\n"
        "applications/{alpha,beta}_application/host/factory.py\n"
        "```\n"
    )
    result = mod.extract_write_set(plan, {"sections": [{"heading": "## A"}]})
    assert result["paths"] == [
        "applications/alpha_application/host/factory.py",
        "applications/beta_application/host/factory.py",
        "intergrax/scaffold/new_application.py",
        "intergrax/scaffold/new_product.py",
    ]


def test_write_set_occurrence_selects_duplicate_heading():
    mod = _import_build()
    plan = "## A\n\n```\ntests/a.py\n```\n\n## A\n\n```\ntests/b.py\n```\n"
    result = mod.extract_write_set(plan, {"sections": [{"heading": "## A", "occurrence": 2}]})
    assert result["paths"] == ["tests/b.py"]


def test_label_path_states():
    mod = _import_build()
    tree = frozenset({"tests/a.py", "intergrax/x/y.py"})
    assert mod.label_path("tests/a.py", tree) == "exists"
    assert mod.label_path("tests/b.py", tree) == "new"
    assert mod.label_path("intergrax/x/*", tree) == "glob"
    assert mod.label_path("intergrax/x/", tree) == "exists"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("tests/a.py", "tests/a.py", True),
        ("intergrax/p/*", "intergrax/p/sub/z.py", True),
        ("intergrax/p/", "intergrax/p/z.py", True),
        ("intergrax/p/*", "intergrax/p/sub/*", True),
        ("tests/a.py", "tests/ab.py", False),
        ("intergrax/p/*", "intergrax/q/z.py", False),
    ],
)
def test_paths_overlap(a: str, b: str, expected: bool):
    mod = _import_build()
    assert mod.paths_overlap(a, b) is expected
    assert mod.paths_overlap(b, a) is expected


_WS = {"T-A": ["tests/shared.py"], "T-B": ["tests/shared.py"], "T-C": ["tests/c.py"]}


def test_overlap_matrix_lists_pairs_with_paths():
    mod = _import_build()
    assert mod.overlap_matrix(_WS) == [{"a": "T-A", "b": "T-B", "paths": ["tests/shared.py"]}]


def test_unordered_overlap_pair_fails():
    mod = _import_build()
    with pytest.raises(SystemExit):
        mod.check_merge_order(set(_WS), set(), [("T-A", "T-C")], mod.overlap_matrix(_WS))


def test_transitive_order_satisfies_overlap_pair():
    mod = _import_build()
    order = mod.check_merge_order(set(_WS), set(), [("T-A", "T-C"), ("T-C", "T-B")], mod.overlap_matrix(_WS))
    assert order.index("T-A") < order.index("T-C") < order.index("T-B")


def test_landed_ticket_pair_needs_no_edge():
    mod = _import_build()
    mod.check_merge_order(set(_WS), {"T-A"}, [], mod.overlap_matrix(_WS))


def test_dependency_cycle_fails():
    mod = _import_build()
    with pytest.raises(SystemExit):
        mod.check_merge_order(set(_WS), set(), [("T-A", "T-B"), ("T-B", "T-A")], mod.overlap_matrix(_WS))


def test_lanes_accept_disjoint_groups():
    mod = _import_build()
    lanes = [
        {"lane": "lane_1", "label": "x", "tickets": ["T-A", "T-C"]},
        {"lane": "lane_2", "label": "y", "tickets": ["T-B"]},
    ]
    mod.check_lanes(lanes, _WS, [("T-A", "T-B")], set(_WS), set())


@pytest.mark.parametrize(
    "lanes",
    [
        [["T-A", "T-B", "T-C"]],
        [["T-A", "T-C"], ["T-B", "T-C"]],
        [["T-A"], ["T-B"]],
    ],
    ids=["overlap_inside_lane", "ticket_twice", "open_ticket_missing"],
)
def test_lanes_reject_invalid_groups(lanes: list[list[str]]):
    mod = _import_build()
    payload = [{"lane": f"lane_{i}", "label": "x", "tickets": t} for i, t in enumerate(lanes, 1)]
    with pytest.raises(SystemExit):
        mod.check_lanes(payload, _WS, [("T-A", "T-B")], set(_WS), set())


def test_lane_cap_is_five():
    mod = _import_build()
    tickets = [f"T-{n}" for n in range(6)]
    write_sets = {t: [f"tests/{t}.py"] for t in tickets}
    with pytest.raises(SystemExit):
        mod.check_lanes([{"lane": "lane_1", "label": "x", "tickets": tickets}], write_sets, [], set(tickets), set())


def test_lane_members_must_not_depend_on_each_other():
    mod = _import_build()
    with pytest.raises(SystemExit):
        mod.check_lanes(
            [
                {"lane": "lane_1", "label": "x", "tickets": ["T-A", "T-C"]},
                {"lane": "lane_2", "label": "y", "tickets": ["T-B"]},
            ],
            _WS,
            [("T-A", "T-C"), ("T-A", "T-B")],
            set(_WS),
            set(),
        )


def test_open_w3_ticket_needs_pt_wa_predecessor():
    mod = _import_build()
    write_sets = {mod.PT_WA_TICKET: ["tests/a.py"], "T-B": ["tests/b.py"]}
    lanes = [
        {"lane": "lane_1", "label": "x", "tickets": [mod.PT_WA_TICKET]},
        {"lane": "lane_2", "label": "y", "tickets": ["T-B"]},
    ]
    with pytest.raises(SystemExit):
        mod.check_lanes(lanes, write_sets, [], set(write_sets), {"T-B"})
    mod.check_lanes(lanes, write_sets, [(mod.PT_WA_TICKET, "T-B")], set(write_sets), {"T-B"})


def _blank_row(frz: str):
    mod = _import_build()
    return mod.CodeFixRow(
        frz_id=frz,
        summary="s",
        negative_test="NEG-0",
        remediation_track="CODE_FIX:",
        owner_ticket=None,
        regression_scope="ENT-A",
    )


@pytest.mark.parametrize(
    ("frz", "ticket", "node"),
    [
        ("FRZ-TYP-01", "T-IGX-078-C", "T-AF2-RW-TYP"),
        ("FRZ-TYP-03", "T-IGX-078-C", "T-AF2-RW-TYP"),
        ("FRZ-RPL-04", "T-IGX-078-C", "T-AF2-RW-TYP"),
        ("FRZ-TEN-02", "T-IGX-078-D", "PT-T01"),
        ("FRZ-TEN-07", "T-IGX-078-D", "PT-T01"),
    ],
)
def test_ga2r_reownership(frz: str, ticket: str, node: str):
    mod = _import_build()
    applied = mod.apply_explicit_ownership(frz, _blank_row(frz))
    assert applied["owner_ticket"] == ticket
    assert applied["owner_node"] == node
    assert applied["owner_provenance"] == "explicit_ga2r"


def test_a4_006_audit_conflict_owned_by_pt_t01_ticket():
    mod = _import_build()
    a4 = next(c for c in mod.AUDIT_FLAG_CONFLICTS if c["finding_id"] == "A4-006")
    assert a4["owner_ticket"] == "T-IGX-078-D"
    assert a4["owner_node"] == "PT-T01"


def test_scope_hard_predecessors_complete():
    mod = _import_build()
    assert set(mod.SCOPE_HARD_PREDECESSORS) == {
        "M1",
        "DS-1",
        "DS-2",
        "DS-3",
        "DS-4",
        "DS-EXC",
        "PT-V04",
        "PT-B01",
        "PT-B02",
        "REV-MASTER",
        "T-AF2-RW-PLAN",
        "T-AF2-RW-PLAN-AUDIT",
    }


def test_frz_frz_10_canonical_node_ownership():
    mod = _import_build()
    applied = mod.apply_explicit_ownership("FRZ-FRZ-10", _blank_row("FRZ-FRZ-10"))
    assert applied["owner_ticket"] is None
    assert applied["owner_node"] == "T-B7-PFP-ACTIVATE"
    assert applied["verification_node"] == "T-B7-PFP-VERIFY"
    assert applied["owner_provenance"] == "explicit_mep_wp12"
    assert applied.get("pre_g2_execution_ticket") == "T-IGX-052-I"


def test_unrelated_frz_without_explicit_has_no_provenance():
    mod = _import_build()
    row = mod.CodeFixRow(
        frz_id="FRZ-BND-04",
        summary="s",
        negative_test="NT",
        remediation_track="CODE_FIX:",
        owner_ticket="T-IGX-043-G",
        regression_scope="ENT-A",
    )
    applied = mod.apply_explicit_ownership("FRZ-BND-04", row)
    assert applied == {"owner_ticket": "T-IGX-043-G"}


def test_live_bindings_put_pt_wa_before_pt_wb():
    mod = _import_build()
    bindings = mod.load_bindings()
    labels = [lane["label"] for lane in bindings["lanes"]]
    assert labels.index("PT-WA") < labels.index("PT-WB")
    assert len({lane["lane"] for lane in bindings["lanes"]}) == len(labels)


def test_live_wave_dependencies_have_no_wave_self_dependency():
    mod = _import_build()
    bindings = mod.load_bindings()
    assert "PT-WA" not in bindings["dependencies"]["W3"]


def test_live_audit_blocker_rows_name_a_pytest_node():
    mod = _import_build()
    rows = mod.load_bindings()["code_fix_rows"]
    for frz in ("FRZ-SEC-06", "FRZ-TEN-02", "FRZ-TEN-07", "FRZ-TYP-01", "FRZ-TYP-03", "FRZ-RPL-04"):
        first = rows[frz]["negative_tests"][0]
        path, separator, name = first["node"].partition("::")
        assert separator and path.endswith(".py") and name.startswith("test_"), (frz, first["node"])


_REGRESSION_TREE = frozenset(
    {"tests/unit/a/test_row.py", "tests/unit/a/test_ws.py", "tests/unit/a/helper.py", "tests/unit/g/test_glob.py"}
)


def test_ticket_regression_command_unions_row_write_set_and_bound_tests():
    mod = _import_build()
    write_set = ["tests/unit/a/test_ws.py", "tests/unit/a/helper.py", "tests/unit/a/test_new.py", "tests/unit/g/*"]
    out = mod.ticket_regression(
        "T-A",
        {"regression_tests": ["tests/unit/a/test_row.py::test_x"], "regression_extra_commands": ["uv run python s.py"]},
        [{"owner_ticket": "T-A", "regression_tests": ["tests/unit/a/test_row.py"]}],
        write_set,
        _REGRESSION_TREE,
    )
    assert out["regression_tests"] == [
        "tests/unit/a/test_row.py",
        "tests/unit/a/test_ws.py",
        "tests/unit/g/test_glob.py",
    ]
    assert out["regression_commands"] == [
        "uv run pytest tests/unit/a/test_row.py tests/unit/a/test_ws.py tests/unit/g/test_glob.py -q",
        "uv run python s.py",
    ]


def test_ticket_without_regression_scope_fails_closed():
    mod = _import_build()
    with pytest.raises(SystemExit):
        mod.ticket_regression("T-A", {}, [], ["docs/x.md"], _REGRESSION_TREE)
    out = mod.ticket_regression("T-A", {"regression_absent_reason": "r"}, [], [], _REGRESSION_TREE)
    assert out["regression_commands"] == [] and out["regression_absent_reason"] == "r"


def test_ticket_bound_regression_test_must_exist_at_pin():
    mod = _import_build()
    with pytest.raises(SystemExit):
        mod.ticket_regression("T-A", {"regression_tests": ["tests/unit/a/test_gone.py"]}, [], [], _REGRESSION_TREE)


def test_live_bnd_06_regression_enumerates_adapter_tests():
    mod = _import_build()
    tests = mod.load_bindings()["code_fix_rows"]["FRZ-BND-06"]["regression_tests"]
    for prefix in (
        "tests/unit/hosting/",
        "tests/unit/queueing/",
        "tests/unit/applications/test_mcp_",
        "tests/unit/applications/test_http_",
    ):
        assert any(t.startswith(prefix) and t.endswith(".py") for t in tests), prefix
    assert "tests/unit/runtime/architecture/test_harness_w5_typed_observability_transport_gate.py" in tests


def test_live_debt_02_negative_case_is_the_seeded_gate_node():
    mod = _import_build()
    first = mod.load_bindings()["code_fix_rows"]["FRZ-DEBT-02"]["negative_tests"][0]
    assert first["node"].endswith("::test_rw_dep_01c_gate_reports_seeded_undeclared_core_requirement")
    assert "seeded" in first["mechanism"].lower()


def test_live_code_fix_inventory_is_39_rows():
    mod = _import_build()
    payload = mod.build()
    assert len(payload["code_fix_inventory"]) == 39


def test_live_evidence_missing_rows_carry_od_af_close_b2():
    mod = _import_build()
    inv = {r["frz_id"]: r for r in mod.build()["code_fix_inventory"]}
    missing = [frz for frz, row in inv.items() if row["red_first_status"] == "EVIDENCE_MISSING"]
    assert len(missing) == 12
    for frz in missing:
        row = inv[frz]
        assert row.get("operator_decision_id") == "OD-AF-CLOSE-B2"
        assert row.get("operator_decision_ref")
        assert row.get("evidence_negative_test")
        assert row.get("evidence_mechanism")
        assert row.get("evidence_certification_obligation")
    for frz in ("FRZ-DEBT-02", "FRZ-DEBT-03"):
        assert inv[frz].get("evidence_seeded_red_required") is True


def test_live_frz_frz_10_pre_g2_schedule():
    mod = _import_build()
    payload = mod.build()
    schedule = payload["post_freeze_policy_schedule"]
    assert schedule["pre_g2_execution_ticket"] == "T-IGX-052-I"
    assert schedule["functional_owner_wave"] == "W5"
    assert schedule["verification_wave"] == "W8"
    frz = next(r for r in payload["code_fix_inventory"] if r["frz_id"] == "FRZ-FRZ-10")
    assert frz["pre_g2_execution_ticket"] == "T-IGX-052-I"
    ticket = next(t for t in payload["tickets"] if t["ticket"] == "T-IGX-052-I")
    assert ticket["wave"] == "pre-G2"
    assert ticket["post_freeze_policy"]["functional_owner_node"] == "T-B7-PFP-ACTIVATE"


# Append to test_build_rw_plan.py


def test_op_b_metadata_applied_when_red_at_prefix_in_prior_plan():
    """Bindings must stamp OP-B fields even after a row records RED_AT_PREFIX."""
    mod = _import_build()
    bindings = mod.load_bindings()
    prior_frz = next(iter(bindings["evidence_missing_obligations"]["rows"]))
    # build() reads prior from rw_plan.v1.json; verify helper applies OP-B metadata at RED_AT_PREFIX.
    entry: dict = {
        "red_first_status": "RED_AT_PREFIX",
        "red_first_prefix_sha": "a" * 40,
        "owner_ticket": bindings["evidence_missing_obligations"]["rows"][prior_frz]["owner_ticket"],
        "negative_tests": [{"node": "tests/unit/x.py::test_a", "mechanism": "m"}],
    }
    mod._apply_evidence_missing_obligation(prior_frz, entry, bindings.get("evidence_missing_obligations"))
    assert entry.get("operator_decision_id") == "OD-AF-CLOSE-B2"
    assert entry.get("evidence_negative_test")
