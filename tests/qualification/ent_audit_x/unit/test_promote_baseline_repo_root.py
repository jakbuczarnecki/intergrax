"""F-02 (IIAC audit ENT-AUDIT-X): ``promote-baseline`` must judge the OD-ENT-01 gate against its own repository.

The CLI used to fall through to ``Path.cwd()`` inside ``promote_accepted_baseline``, so a forged world in the current
directory could drive the write of the real repository's accepted baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import scripts.enterprise_audit.run_enterprise_audit as cli

BASELINE_NAME = "ENTERPRISE_AUDIT_BASELINE.json"


def test_promote_passes_the_script_repository_not_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake(baseline_dir: Path, **kwargs: Any) -> Path:
        seen.update(kwargs)
        return baseline_dir / BASELINE_NAME

    monkeypatch.setattr(cli, "promote_accepted_baseline", fake)
    monkeypatch.chdir(tmp_path)
    record = tmp_path / "OD_ENT_01.json"
    record.write_text("{}", encoding="utf-8")

    assert cli.main(["promote-baseline", "--od-record", str(record), "--evidence-dir", str(tmp_path)]) == 0
    assert seen["repo_root"] == cli._repo_root()
    assert seen["repo_root"] != tmp_path


def test_forged_world_in_the_working_directory_cannot_promote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    forged = tmp_path / "forged"
    (forged / "operator_records").mkdir(parents=True)
    record = forged / "operator_records" / "OD_ENT_01.json"
    record.write_text(json.dumps({"decision_id": "OD-ENT-01", "operator_origin": True}), encoding="utf-8")
    monkeypatch.chdir(forged)

    real_baseline = cli._repo_root() / "tests/qualification/ent_audit_x" / BASELINE_NAME
    existed = real_baseline.exists()

    assert cli.main(["promote-baseline", "--od-record", str(record), "--evidence-dir", str(tmp_path)]) != 0
    assert real_baseline.exists() == existed, "a rejected promotion must not write a baseline"
    assert capsys.readouterr().err.strip(), "the rejection reason must be reported"
