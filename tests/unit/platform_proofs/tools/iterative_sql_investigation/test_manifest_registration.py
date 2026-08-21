# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

from scripts.proof.intergrax_proof_manifest import load_manifest


def test_tools_iterative_sql_investigation_registered() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    manifest = load_manifest(repo_root=repo_root)
    entry = next(
        item for item in manifest.entries if item.proof_id == "TOOLS-ITERATIVE-SQL-INVESTIGATION"
    )
    assert entry.domain == "tools"
    assert "run_proof.py" in entry.command.argv[-1]
    assert any(req.name == "INTERGRAX_LLM_PROVIDER" for req in entry.environment_requirements)
