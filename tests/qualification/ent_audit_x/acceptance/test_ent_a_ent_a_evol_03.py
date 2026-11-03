from __future__ import annotations

import os
from pathlib import Path

import pytest
from tests.qualification.ent_audit_x.acceptance._gate_oracle import assert_semantic_gate_executes


@pytest.mark.ent_semantic_gate
def test_acceptance_ent_a_evol_03_semantic_gate(repo_root: Path) -> None:
    if os.environ.get("ENT_AUDIT_X_SEMANTIC") != "1":
        pytest.skip("semantic gate matrix requires ENT_AUDIT_X_SEMANTIC=1")
    assert_semantic_gate_executes(repo_root, "ENT-A-EVOL-03")
