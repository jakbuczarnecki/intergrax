# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_BUILD = _REPO / "scripts/governance/build_rw_plan.py"


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


def test_build_red_first_null_not_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    payload = {
        "code_fix_inventory": [
            {
                "frz_id": "FRZ-BND-04",
                "red_first_status": "EVIDENCE_MISSING",
                "red_first_prefix_sha": head,
            }
        ]
    }
    rw_json = tmp_path / "rw_plan.v1.json"
    rw_json.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(mod, "_RW_PLAN_JSON", rw_json)
    monkeypatch.setattr(mod, "_load_prior_red_first", lambda: {})
    monkeypatch.setattr(
        mod,
        "parse_appendix_a_code_fix_inventory",
        lambda: [
            mod.CodeFixRow(
                frz_id="FRZ-BND-04",
                summary="s",
                negative_test="NT-BND-04",
                remediation_track="CODE_FIX:",
                owner_ticket="T-IGX-043-G",
                regression_scope="ENT-A",
            )
        ],
    )
    inv = mod.build()["code_fix_inventory"][0]
    assert inv["red_first_prefix_sha"] is None
    assert inv["red_first_status"] == "EVIDENCE_MISSING"


def test_preserve_red_at_prefix_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    prior = {
        "code_fix_inventory": [
            {
                "frz_id": "FRZ-BND-04",
                "red_first_status": "RED_AT_PREFIX",
                "red_first_prefix_sha": "a" * 40,
                "red_first_evidence_ref": ".tmp/proof/x",
            }
        ]
    }
    rw_json = tmp_path / "rw_plan.v1.json"
    rw_json.write_text(json.dumps(prior), encoding="utf-8")
    monkeypatch.setattr(mod, "_RW_PLAN_JSON", rw_json)
    monkeypatch.setattr(
        mod,
        "parse_appendix_a_code_fix_inventory",
        lambda: [
            mod.CodeFixRow(
                frz_id="FRZ-BND-04",
                summary="s",
                negative_test="NT-BND-04",
                remediation_track="CODE_FIX:",
                owner_ticket="T-IGX-043-G",
                regression_scope="ENT-A",
            )
        ],
    )
    inv = mod.build()["code_fix_inventory"][0]
    assert inv["red_first_status"] == "RED_AT_PREFIX"
    assert inv["red_first_prefix_sha"] == "a" * 40


def test_preserve_red_first_unreproducible_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _import_build()
    prior = {
        "code_fix_inventory": [
            {
                "frz_id": "FRZ-CTR-01",
                "red_first_status": "RED_FIRST_UNREPRODUCIBLE",
                "red_first_prefix_sha": "b" * 40,
                "red_first_evidence_ref": "docs/x.json#FRZ-CTR-01",
                "red_first_reason": "strict exit 2",
            }
        ]
    }
    rw_json = tmp_path / "rw_plan.v1.json"
    rw_json.write_text(json.dumps(prior), encoding="utf-8")
    monkeypatch.setattr(mod, "_RW_PLAN_JSON", rw_json)
    monkeypatch.setattr(
        mod,
        "parse_appendix_a_code_fix_inventory",
        lambda: [
            mod.CodeFixRow(
                frz_id="FRZ-CTR-01",
                summary="s",
                negative_test="NT-CTR-01",
                remediation_track="CODE_FIX:",
                owner_ticket="T-IGX-051-D",
                regression_scope="ENT-A",
            )
        ],
    )
    inv = mod.build()["code_fix_inventory"][0]
    assert inv["red_first_status"] == "RED_FIRST_UNREPRODUCIBLE"
    assert inv["red_first_reason"] == "strict exit 2"


def test_pt_wa_precedes_pt_wb():
    mod = _import_build()
    labels = [e[2] for e in mod.LANE_ORDER if isinstance(e, (list, tuple)) and len(e) >= 3 and isinstance(e[2], str)]
    assert labels.index("PT-WA") < labels.index("PT-WB")


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
    ("frz", "ticket"),
    [
        ("FRZ-TYP-01", "T-IGX-051-G"),
        ("FRZ-TYP-03", "T-IGX-051-G"),
        ("FRZ-RPL-04", "T-IGX-051-G"),
    ],
)
def test_explicit_frz_ownership_ticket(frz: str, ticket: str):
    mod = _import_build()
    applied = mod.apply_explicit_ownership(frz, _blank_row(frz))
    assert applied["owner_ticket"] == ticket
    assert applied["owner_provenance"] == "explicit_mep_wp12"


@pytest.mark.parametrize("frz", ["FRZ-TEN-02", "FRZ-TEN-07"])
def test_explicit_pt_t01_node_ownership_without_delivery_ticket(frz: str):
    mod = _import_build()
    applied = mod.apply_explicit_ownership(frz, _blank_row(frz))
    assert applied["owner_ticket"] is None
    assert applied["owner_node"] == "PT-T01"
    assert applied["owner_provenance"] == "explicit_mep_wp12"


def test_a4_006_audit_conflict_pt_t01_node():
    mod = _import_build()
    a4 = next(c for c in mod.AUDIT_FLAG_CONFLICTS if c["finding_id"] == "A4-006")
    assert a4["owner_ticket"] is None
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
