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


def test_pt_wa_precedes_pt_wb():
    mod = _import_build()
    labels = [e[2] for e in mod.LANE_ORDER if isinstance(e, (list, tuple)) and len(e) >= 3 and isinstance(e[2], str)]
    assert labels.index("PT-WA") < labels.index("PT-WB")
