# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from platform_proofs.scenarios.ai_incident_investigation.evidence_builder import (
    EVIDENCE_RESOLVED_FILENAME,
    EVIDENCE_UNRESOLVED_FILENAME,
    REPORT_RESOLVED_FILENAME,
    REPORT_UNRESOLVED_FILENAME,
    build_platform_proof_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.evaluator import evaluate_scenario_run
from platform_proofs.scenarios.ai_incident_investigation.fixtures import ScenarioVariant
from platform_proofs.scenarios.ai_incident_investigation.reproduction import (
    PROOF_ID,
    canonical_reproduction_shell_command,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario import (
    build_runtime_bundle,
    execute_resolved_skeleton,
)
from scripts.proof.intergrax_platform_proof_evidence import PlatformProofEvidence
from scripts.proof.intergrax_proof_runner import read_git_metadata

pytestmark = pytest.mark.unit


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def test_readme_canonical_command_matches_reproduction_module() -> None:
    readme = (
        _repo_root()
        / "platform_proofs/scenarios/ai_incident_investigation/README.md"
    ).read_text(encoding="utf-8")
    command = canonical_reproduction_shell_command()
    assert command in readme


def test_evidence_reproduction_command_matches_canonical_module() -> None:
    async def _build() -> None:
        bundle = build_runtime_bundle(variant=ScenarioVariant.RESOLVED)
        result = await execute_resolved_skeleton(bundle)
        evaluation = evaluate_scenario_run(result, bundle.fixture)
        evidence = build_platform_proof_evidence(
            result,
            variant=ScenarioVariant.RESOLVED,
            evaluation=evaluation,
            source_revision="testsha",
        )
        assert evidence.reproduction.command == canonical_reproduction_shell_command()

    import asyncio

    asyncio.run(_build())


@pytest.mark.integration
def test_canonical_cli_reproduces_scenario_proof() -> None:
    repo_root = _repo_root()
    command = canonical_reproduction_shell_command()
    argv = command.split()
    completed = subprocess.run(
        argv,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "PASS" in completed.stdout
    assert PROOF_ID in completed.stdout
    assert f"proofs/{PROOF_ID}" in completed.stdout.replace("\\", "/")

    artifact_root = repo_root / ".artifacts" / "proof"
    suite_dirs = sorted(artifact_root.iterdir(), key=lambda path: path.stat().st_mtime)
    proof_dir = next(
        path
        for path in reversed(suite_dirs)
        if (path / "proofs" / PROOF_ID).is_dir()
    ).resolve() / "proofs" / PROOF_ID

    for filename in (
        EVIDENCE_RESOLVED_FILENAME,
        EVIDENCE_UNRESOLVED_FILENAME,
        REPORT_RESOLVED_FILENAME,
        REPORT_UNRESOLVED_FILENAME,
        "domain_result.json",
    ):
        assert (proof_dir / filename).is_file(), filename

    resolved = PlatformProofEvidence.model_validate_json(
        (proof_dir / EVIDENCE_RESOLVED_FILENAME).read_text(encoding="utf-8")
    )
    unresolved = PlatformProofEvidence.model_validate_json(
        (proof_dir / EVIDENCE_UNRESOLVED_FILENAME).read_text(encoding="utf-8")
    )
    assert resolved.reproduction.command == command
    assert unresolved.reproduction.command == command
    assert "pytest" not in resolved.reproduction.command
    assert "pytest" not in unresolved.reproduction.command

    git_sha = read_git_metadata(repo_root).commit_sha
    assert resolved.reproduction.source_revision == git_sha
    assert unresolved.reproduction.source_revision == git_sha

    resolved_html = (proof_dir / REPORT_RESOLVED_FILENAME).read_text(encoding="utf-8")
    unresolved_html = (proof_dir / REPORT_UNRESOLVED_FILENAME).read_text(encoding="utf-8")
    assert command in resolved_html
    assert command in unresolved_html
    assert "pytest" not in _reproduction_section(resolved_html)
    assert "pytest" not in _reproduction_section(unresolved_html)


def test_unknown_proof_id_cli_fails_clearly() -> None:
    repo_root = _repo_root()
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/proof/run-intergrax-proof-suite.py",
            "--profile",
            "quick",
            "--proof-id",
            "DOES-NOT-EXIST",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "unknown proof_id" in (completed.stderr or completed.stdout)


def _reproduction_section(html: str) -> str:
    start = html.find('<section id="reproduction"')
    end = html.find("</section>", start)
    assert start != -1 and end != -1
    return html[start:end]
