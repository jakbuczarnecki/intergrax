# © Artur Czarnecki. All rights reserved.

"""UE-11G-C1 matrix anchor and proof-report gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.system.unified_execution.proof_runner.contracts import ProofReport

pytestmark = [
    pytest.mark.unit,
    pytest.mark.external_proof,
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPORT_PATH = _REPO_ROOT / ".tmp" / "session" / "ue-11g-c1" / "docker-run" / "proof-report.json"


def test_ue_11g_c1_proof_report_schema_when_present() -> None:
    if not _REPORT_PATH.is_file():
        pytest.skip("UE-11G-C1 docker proof report not present")
    payload = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    report = ProofReport.model_validate(payload)
    assert report.proof_id == "UE-11G-C1"
    if report.verdict == "PASS":
        assert report.evidence is not None
        assert report.evidence.functional_oracle_pass is True
        assert report.evidence.agent_id == "local_search"
        assert report.evidence.capability == "local.workspace.search"
