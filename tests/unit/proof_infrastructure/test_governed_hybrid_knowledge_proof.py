# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from proof_infrastructure.governed_hybrid_knowledge_proof.models import SemanticDecisionV1
from proof_infrastructure.governed_hybrid_knowledge_proof.runner import run_flagship_proof

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_flagship_proof_all_scenarios_pass() -> None:
    result = run_flagship_proof(emit_terminal=False)

    assert result.passed_count == 4
    assert result.all_passed is True
    assert result.overall_status == "PASS"
    assert result.scenario_1.observed == SemanticDecisionV1.NO
    assert result.scenario_1.http_read_count == 1
    assert result.scenario_2.observed == SemanticDecisionV1.YES
    assert result.scenario_2.http_read_count == 1
    assert result.scenario_3.http_read_count == 0
    assert result.scenario_3.llm_call_count == 0
    assert result.scenario_3.observed == SemanticDecisionV1.CANNOT_DETERMINE
    assert result.scenario_4.observed == SemanticDecisionV1.NO
    assert result.scenario_1_run_id is not None


def test_flagship_proof_repeatable_twice() -> None:
    first = run_flagship_proof(emit_terminal=False)
    second = run_flagship_proof(emit_terminal=False)
    assert first.all_passed is True
    assert second.all_passed is True


def test_flagship_proof_cli_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "proof_infrastructure.governed_hybrid_knowledge_proof",
            "--json",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-800:]
    json_start = completed.stdout.find("{")
    assert json_start >= 0
    payload = json.loads(completed.stdout[json_start:])
    assert payload["overall_status"] == "PASS"
    assert payload["scenario_1"]["passed"] is True
    assert payload["scenario_4"]["passed"] is True
