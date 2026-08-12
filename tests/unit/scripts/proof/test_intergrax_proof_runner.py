# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.proof.intergrax_proof_contracts import (
    EnvRequirement,
    EnvRequirementKind,
    ProofArgvCommand,
    ProofManifestEntry,
    ProofProfile,
    ProofRunResult,
    ProofSafetyClass,
    ProofStatus,
    SuiteOverallStatus,
    SuiteReceipt,
)
from scripts.proof.intergrax_proof_runner import (
    RunnerConfig,
    aggregate_overall_status,
    evaluate_environment,
    execute_proof,
    render_console_summary,
    run_suite,
    suite_exit_code,
)


def _entry(
    proof_id: str,
    *,
    safety: ProofSafetyClass = ProofSafetyClass.LOCAL_READ_ONLY,
    env: tuple[EnvRequirement, ...] = (),
    timeout: int = 30,
    argv: tuple[str, ...] = ("-c", "import sys; sys.exit(0)"),
) -> ProofManifestEntry:
    return ProofManifestEntry(
        proof_id=proof_id,
        title=proof_id,
        domain="test",
        profiles=frozenset({ProofProfile.QUICK}),
        proof_kind="test",
        command=ProofArgvCommand(executable="python", argv=argv),
        environment_requirements=env,
        safety_class=safety,
        timeout_seconds=timeout,
    )


def test_env_absent_blocks_without_value_in_result(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _entry(
        "ENV-TEST",
        env=(EnvRequirement(kind=EnvRequirementKind.ENV_PRESENT, name="SECRET_TEST_VAR"),),
    )
    monkeypatch.delenv("SECRET_TEST_VAR", raising=False)
    results, ok = evaluate_environment(entry)
    assert ok is False
    assert results[0].satisfied is False
    serialized = json.dumps([item.model_dump() for item in results])
    assert "SECRET_TEST_VAR" in serialized
    assert "xox" not in serialized


def test_env_present_does_not_expose_value(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _entry(
        "ENV-TEST",
        env=(EnvRequirement(kind=EnvRequirementKind.ENV_PRESENT, name="SECRET_TEST_VAR"),),
    )
    monkeypatch.setenv("SECRET_TEST_VAR", "super-secret-token-value")
    results, ok = evaluate_environment(entry)
    assert ok is True
    serialized = json.dumps([item.model_dump() for item in results])
    assert "super-secret-token-value" not in serialized


def test_child_pass_and_fail(tmp_path: Path) -> None:
    pass_entry = _entry("PASS-TEST", argv=("-c", "print('ok')"))
    fail_entry = _entry("FAIL-TEST", argv=("-c", "import sys; sys.exit(3)"))

    pass_result = execute_proof(pass_entry, repo_root=tmp_path)
    fail_result = execute_proof(fail_entry, repo_root=tmp_path)

    assert pass_result.status == ProofStatus.PASS
    assert pass_result.exit_code == 0
    assert fail_result.status == ProofStatus.FAIL
    assert fail_result.exit_code == 3


def test_timeout(tmp_path: Path) -> None:
    entry = _entry(
        "TIMEOUT-TEST",
        timeout=1,
        argv=("-c", "import time; time.sleep(5)"),
    )
    result = execute_proof(entry, repo_root=tmp_path)
    assert result.status == ProofStatus.FAIL
    assert result.diagnostic_summary == "timeout"


def test_missing_executable(tmp_path: Path) -> None:
    entry = ProofManifestEntry(
        proof_id="MISSING",
        title="missing",
        domain="test",
        profiles=frozenset({ProofProfile.QUICK}),
        proof_kind="test",
        command=ProofArgvCommand(
            executable="definitely-missing-binary",
            argv=("noop",),
        ),
        safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
    )
    result = execute_proof(entry, repo_root=tmp_path)
    assert result.status == ProofStatus.FAIL
    assert result.diagnostic_summary == "missing_executable"


def test_bounded_stdout_stderr(tmp_path: Path) -> None:
    entry = _entry(
        "STDIO-TEST",
        argv=("-c", "print('x' * 10000)"),
    )

    def _runner(command, **kwargs):
        assert kwargs["shell"] is False
        return subprocess.run(command, **kwargs)

    result = execute_proof(entry, repo_root=tmp_path, subprocess_runner=_runner)
    assert result.status == ProofStatus.PASS
    assert len(result.stdout_tail) <= 4096 + 32


def test_one_fail_makes_suite_fail() -> None:
    results = (
        ProofRunResult(proof_id="A", status=ProofStatus.PASS, duration_seconds=1.0),
        ProofRunResult(proof_id="B", status=ProofStatus.FAIL, duration_seconds=1.0),
    )
    assert aggregate_overall_status(ProofProfile.QUICK, results) == SuiteOverallStatus.FAIL


def test_live_blocked_environment_is_pass_with_blocked() -> None:
    results = (
        ProofRunResult(proof_id="A", status=ProofStatus.PASS, duration_seconds=1.0),
        ProofRunResult(
            proof_id="SLACK-CONVERSATION-LIVE",
            status=ProofStatus.BLOCKED_ENVIRONMENT,
            duration_seconds=0.0,
        ),
    )
    assert aggregate_overall_status(ProofProfile.LIVE, results) == (
        SuiteOverallStatus.PASS_WITH_BLOCKED
    )


def test_quick_blocked_environment_fails_suite() -> None:
    results = (
        ProofRunResult(
            proof_id="A",
            status=ProofStatus.BLOCKED_ENVIRONMENT,
            duration_seconds=0.0,
        ),
    )
    assert aggregate_overall_status(ProofProfile.QUICK, results) == SuiteOverallStatus.FAIL


def test_skipped_platform_semantics() -> None:
    results = (
        ProofRunResult(
            proof_id="PLATFORM-MACOS",
            status=ProofStatus.SKIPPED_PLATFORM,
            duration_seconds=0.0,
        ),
        ProofRunResult(proof_id="A", status=ProofStatus.PASS, duration_seconds=1.0),
    )
    assert aggregate_overall_status(ProofProfile.FULL, results) == SuiteOverallStatus.PASS


def test_receipt_schema_and_ordering(tmp_path: Path) -> None:
    receipt = SuiteReceipt(
        suite_run_id="run-1",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        completed_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        git_commit_sha="abc123def456",
        git_dirty=True,
        profile=ProofProfile.QUICK,
        platform="windows",
        python_version="3.12.0",
        overall_status=SuiteOverallStatus.PASS,
        results=(
            ProofRunResult(proof_id="B", status=ProofStatus.PASS, duration_seconds=1.0),
            ProofRunResult(proof_id="A", status=ProofStatus.PASS, duration_seconds=1.0),
        ),
        passed_count=2,
        failed_count=0,
        blocked_count=0,
        skipped_count=0,
    )
    payload = receipt.model_dump(mode="json")
    assert payload["schema_version"] == "intergrax.proof_suite_receipt.v1"
    assert payload["git_dirty"] is True
    assert "super-secret" not in json.dumps(payload)


def test_external_mutating_requires_opt_in() -> None:
    entry = ProofManifestEntry(
        proof_id="EXT-MUT",
        title="ext",
        domain="slack",
        profiles=frozenset({ProofProfile.LIVE}),
        proof_kind="test",
        command=ProofArgvCommand(executable="python", argv=("-c", "print(1)")),
        safety_class=ProofSafetyClass.EXTERNAL_MUTATING,
    )
    config = RunnerConfig(
        profile=ProofProfile.LIVE,
        repo_root=Path("."),
        allow_external_mutating=False,
    )
    from scripts.proof.intergrax_proof_runner import _should_skip_profile

    assert _should_skip_profile(entry, config) == "external_mutating_opt_in_required"


def test_suite_exit_code_semantics() -> None:
    pass_receipt = SuiteReceipt(
        suite_run_id="x",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        git_commit_sha="sha",
        git_dirty=False,
        profile=ProofProfile.QUICK,
        platform="windows",
        python_version="3.12",
        overall_status=SuiteOverallStatus.PASS,
        results=(),
        passed_count=0,
        failed_count=0,
        blocked_count=0,
        skipped_count=0,
    )
    blocked_receipt = pass_receipt.model_copy(
        update={"overall_status": SuiteOverallStatus.PASS_WITH_BLOCKED}
    )
    fail_receipt = pass_receipt.model_copy(update={"overall_status": SuiteOverallStatus.FAIL})
    assert suite_exit_code(pass_receipt) == 0
    assert suite_exit_code(blocked_receipt) == 0
    assert suite_exit_code(fail_receipt) == 1


def test_render_console_summary_counts() -> None:
    receipt = SuiteReceipt(
        suite_run_id="x",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        git_commit_sha="abc123",
        git_dirty=False,
        profile=ProofProfile.QUICK,
        platform="windows",
        python_version="3.12",
        overall_status=SuiteOverallStatus.PASS,
        results=(
            ProofRunResult(
                proof_id="RUNTIME-TOKEN-OPTIMIZATION-OFFLINE",
                status=ProofStatus.PASS,
                duration_seconds=1.0,
            ),
        ),
        passed_count=1,
        failed_count=0,
        blocked_count=0,
        skipped_count=0,
    )
    text = render_console_summary(receipt)
    assert "INTERGRAX PROOF SUITE" in text
    assert "passed: 1" in text


def test_dry_run_skips_execution() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    config = RunnerConfig(
        profile=ProofProfile.QUICK,
        repo_root=repo_root,
        dry_run=True,
    )
    receipt, _ = run_suite(config)
    assert all(
        result.status == ProofStatus.SKIPPED_PROFILE for result in receipt.results
    )
