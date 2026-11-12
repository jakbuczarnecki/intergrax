# © Artur Czarnecki. All rights reserved.

"""PDC-X P2 requirements matrix schema and integrity (T-IGX-036-C)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts/qualification/pdc_x_requirements.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pdc_x_requirements_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def req() -> ModuleType:
    return _load()


@pytest.fixture(scope="module")
def payload(req: ModuleType) -> dict[str, object]:
    return req.build()


def test_committed_outputs_are_current_and_valid(req: ModuleType, payload: dict[str, object]) -> None:
    assert req.check(payload) == []


def test_every_row_has_all_mandated_fields(req: ModuleType, payload: dict[str, object]) -> None:
    for row in payload["rows"]:
        assert tuple(row) == req.FIELDS


def test_ids_are_unique_and_cover_all_eight_domains(payload: dict[str, object]) -> None:
    ids = [r["REQUIREMENT_ID"] for r in payload["rows"]]
    assert len(ids) == len(set(ids))
    assert {r["DOMAIN_ID"] for r in payload["rows"]} == {f"PDC-D{i}" for i in range(1, 9)}


def test_no_row_is_certified_or_unknown(payload: dict[str, object]) -> None:
    for row in payload["rows"]:
        assert row["CERTIFICATION_STATUS"] == "NOT_CERTIFIED"
        assert row["AUDITOR_ID"] == "—" and row["AUDITED_SHA"] == "—"
        assert "UNKNOWN" not in row["GAP_CLASS"]


def test_non_reuse_rows_have_a_corrective_task(req: ModuleType, payload: dict[str, object]) -> None:
    for row in payload["rows"]:
        if row["GAP_CLASS"] not in req.NO_TASK_CLASSES:
            assert row["CORRECTIVE_TASK"] not in {"", "—"}, row["REQUIREMENT_ID"]


def test_reuse_rows_cite_an_accepted_sha_and_delta(payload: dict[str, object]) -> None:
    for row in payload["rows"]:
        if row["GAP_CLASS"] == "ALREADY_CERTIFIED_REUSABLE":
            assert len(row["EXISTING_ACCEPTANCE_SHA"]) == 40, row["REQUIREMENT_ID"]
            assert row["CURRENT_HEAD_DELTA"].endswith(row["CURRENT_HEAD_DELTA"].rsplit("@", 1)[-1])


def test_every_frz_id_exists_in_the_checklist(req: ModuleType, payload: dict[str, object]) -> None:
    known = req.checklist_ids()
    assert len(known) == req.EXPECTED_FRZ
    for row in payload["rows"]:
        for frz in row["FRZ_IDS"].split(","):
            assert frz in known, (row["REQUIREMENT_ID"], frz)


def test_validate_rejects_premature_certification(req: ModuleType, payload: dict[str, object]) -> None:
    rows = [dict(r) for r in payload["rows"]]
    rows[0]["CERTIFICATION_STATUS"] = "PASS"
    errors = req.validate({**payload, "rows": rows})
    assert any(e.startswith("PREMATURE_CERTIFICATION:") for e in errors)


def test_validate_rejects_unknown_frz_and_missing_task(req: ModuleType, payload: dict[str, object]) -> None:
    rows = [dict(r) for r in payload["rows"]]
    gap = next(r for r in rows if r["GAP_CLASS"] == "IMPLEMENTATION_COMPLETION")
    gap["FRZ_IDS"] = "FRZ-NOPE-01"
    gap["CORRECTIVE_TASK"] = "—"
    errors = req.validate({**payload, "rows": rows})
    assert any(e.startswith("UNKNOWN_FRZ:") for e in errors)
    assert any(e.startswith("NO_CORRECTIVE_TASK:") for e in errors)
