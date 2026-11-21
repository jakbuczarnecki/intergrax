# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-I Part C: FRZ family attribution and the stage-gate-only declaration (A6-005, A6-004). C01..C09."""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit import digests
from intergrax.platform.enterprise_audit import runner as runner_module
from intergrax.platform.enterprise_audit.baseline import (
    EnterpriseAuditBaseline,
    assert_baseline_manifest_digest_current,
)
from intergrax.platform.enterprise_audit.contracts import AggregateAuditStatus, AuditMode, EnterpriseAuditExitCode
from intergrax.platform.enterprise_audit.digests import _frz_for_check, frz_coverage_artifact, manifest_digest
from intergrax.platform.enterprise_audit.manifest import load_default_manifest
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner
from intergrax.platform.enterprise_audit.validators import validate_frz_coverage

_ROOT = Path(__file__).resolve().parents[4]
_CHECKLIST = _ROOT / "docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md"
_MANIFEST = load_default_manifest()
_NINE_FAMILIES = ("CMP", "EXE", "GOV", "OBS", "PLG", "REC", "REL", "SEC", "STA")
_DECLARED_FAMILIES = ("AUD", "CTL", "DEBT", "FRZ", "HRN", "TRC")


def _checklist_criteria() -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for line in _CHECKLIST.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\| (FRZ-([A-Z]+)-(\d+)) \|", line)
        if match:
            families.setdefault(match.group(2), []).append(match.group(1))
    return families


def _renamed_manifest() -> tuple:
    renamed = dataclasses.replace(_MANIFEST[0], check_id="ENT-A-ZZZ-01")
    return (renamed, *_MANIFEST[1:])


def test_wd_c01_no_check_uses_the_default_fallthrough() -> None:
    assert digests.frz_default_fallback_check_ids(_MANIFEST) == ()
    for family in _NINE_FAMILIES:
        checks = [c for c in _MANIFEST if c.check_id.startswith(f"ENT-A-{family}-")]
        assert len(checks) == 3, family
        for check in checks:
            assert _frz_for_check(check) == (f"FRZ-{family}-*",), check.check_id


def test_wd_c02_strict_artifact_refuses_unattributed_check() -> None:
    checks = _renamed_manifest()
    with pytest.raises(ValueError, match="ENT-A-ZZZ-01"):
        frz_coverage_artifact(checks, strict=True)
    lenient = frz_coverage_artifact(checks, strict=False)
    row = next(r for r in lenient["rows"] if r["check_id"] == "ENT-A-ZZZ-01")
    assert row["frz_mapping"] == ["FRZ-AUD-03"]


def test_wd_c03_certifying_run_is_blocked_on_fallthrough(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def refuse_execution(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no check may execute in this test")

    monkeypatch.setattr(runner_module, "execute_manifest_check", refuse_execution)
    checks = _renamed_manifest()
    report, _manifest, code = EnterpriseAuditRunner(_ROOT, manifest_checks=checks).run(
        AuditMode.FAST, output_dir=tmp_path / "certifying", certifying_run=True
    )
    assert report.status is AggregateAuditStatus.BLOCKED
    assert code is EnterpriseAuditExitCode.VIOLATION
    assert [f.finding_id for f in report.findings] == ["FRZ:COVERAGE_DEFAULT_FALLBACK"]
    assert "ENT-A-ZZZ-01" in report.findings[0].message
    dry_report, _m, _c = EnterpriseAuditRunner(_ROOT, manifest_checks=checks, dry_run=True).run(
        AuditMode.FAST, output_dir=tmp_path / "dry", certifying_run=True
    )
    assert all(f.finding_id != "FRZ:COVERAGE_DEFAULT_FALLBACK" for f in dry_report.findings)


def test_wd_c04_artifact_declares_only_what_the_new_checks_do_not_exercise() -> None:
    criteria = _checklist_criteria()
    artifact = frz_coverage_artifact(_MANIFEST)
    block = artifact["stage_gate_only"]
    assert [entry["family"] for entry in block] == list(_DECLARED_FAMILIES)
    for entry in block:
        family = entry["family"]
        assert entry["owner_stage"] and entry["basis"]
        declared = set(entry["criteria_not_exercised"])
        assert declared, family
        assert declared <= set(criteria[family]), family
        assert "FRZ-AUD-03" not in declared
        mapped = entry.get("mapped_check_id")
        exercised = set(criteria[family]) - declared - {"FRZ-AUD-03"}
        if mapped is None:
            assert exercised == set(), family
    assert block == [
        {
            "family": e.family,
            "owner_stage": e.owner_stage,
            "basis": e.basis,
            "criteria_not_exercised": list(e.criteria_not_exercised),
        }
        for e in digests.STAGE_GATE_ONLY_FRZ_FAMILIES
    ]


def test_wd_c05_checklist_families_are_attributed_or_declared() -> None:
    criteria = _checklist_criteria()
    assert criteria
    attributed = {
        mapping[len("FRZ-") : -len("-*")]
        for check in _MANIFEST
        for mapping in _frz_for_check(check)
        if mapping.endswith("-*")
    }
    declared = {entry.family for entry in digests.STAGE_GATE_ONLY_FRZ_FAMILIES}
    undecided = sorted(family for family in criteria if family not in attributed and family not in declared)
    assert undecided == []


def test_wd_c06_schema_accepts_block_rejects_malformed_and_accepts_legacy(tmp_path: Path) -> None:
    artifact = frz_coverage_artifact(_MANIFEST)

    def check(payload: dict) -> tuple[str, ...]:
        path = tmp_path / "frz_coverage.v1.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return validate_frz_coverage(path)

    assert check(artifact) == ()
    missing_owner = copy.deepcopy(artifact)
    del missing_owner["stage_gate_only"][0]["owner_stage"]
    assert check(missing_owner) != ()
    extra_key = copy.deepcopy(artifact)
    extra_key["stage_gate_only"][0]["unexpected"] = 1
    assert check(extra_key) != ()
    legacy = {k: v for k, v in artifact.items() if k != "stage_gate_only"}
    assert check(legacy) == ()


def test_wd_c07_rows_and_row_count_unchanged() -> None:
    artifact = frz_coverage_artifact(_MANIFEST)
    assert artifact["row_count"] == len(_MANIFEST) == len(artifact["rows"])
    assert all(set(row) == {"check_id", "pg_id", "frz_aud", "frz_mapping", "executable"} for row in artifact["rows"])
    assert [row["check_id"] for row in artifact["rows"]] == [c.check_id for c in _MANIFEST]


def test_wd_c08_manifest_digest_drift_is_refused() -> None:
    current = EnterpriseAuditBaseline(
        accepted_baseline_sha="a" * 40, findings={}, check_manifest_digest=manifest_digest()
    )
    assert_baseline_manifest_digest_current(current, manifest_digest())
    stale = dataclasses.replace(current, check_manifest_digest="0" * 64)
    with pytest.raises(ValueError, match="check_manifest_digest"):
        assert_baseline_manifest_digest_current(stale, manifest_digest())


def test_wd_c09_dup_checks_are_explicit_aggregator_only() -> None:
    dup = [c for c in _MANIFEST if c.check_id.startswith("ENT-A-DUP-")]
    assert len(dup) == 3
    for check in dup:
        assert _frz_for_check(check) == ("FRZ-AUD-03",)
    assert not any(c in digests.frz_default_fallback_check_ids(_MANIFEST) for c in (check.check_id for check in dup))
