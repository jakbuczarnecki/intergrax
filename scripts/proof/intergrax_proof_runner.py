# © Artur Czarnecki. All rights reserved.

"""Master Intergrax proof suite runner (PUBLIC-PROOF-GATE-1)."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from scripts.proof.intergrax_proof_contracts import (
    EnvRequirementKind,
    EnvRequirementResult,
    IntergraxProofManifest,
    ProofManifestEntry,
    ProofProfile,
    ProofRunResult,
    ProofSafetyClass,
    ProofStatus,
    SuiteOverallStatus,
    SuiteReceipt,
)
from scripts.proof.intergrax_proof_manifest import (
    ManifestLoadError,
    expanded_profiles,
    load_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunnerConfig:
    profile: ProofProfile
    repo_root: Path
    verbose: bool = False
    allow_external_mutating: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class GitMetadata:
    commit_sha: str
    dirty: bool


SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


def detect_platform_family() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    if system == "darwin":
        return "macos"
    return "linux"


def read_git_metadata(repo_root: Path) -> GitMetadata:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError):
        return GitMetadata(commit_sha="unknown", dirty=False)
    return GitMetadata(
        commit_sha=commit.stdout.strip(),
        dirty=bool(status.stdout.strip()),
    )


def select_proofs(
    manifest: IntergraxProofManifest,
    *,
    profile: ProofProfile,
    platform_family: str,
) -> list[ProofManifestEntry]:
    active_profiles = expanded_profiles(profile)
    selected: list[ProofManifestEntry] = []
    for entry in manifest.entries:
        if not entry.profiles.intersection(active_profiles):
            continue
        selected.append(entry)
    selected.sort(key=lambda item: item.proof_id)
    return selected


def _env_present(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _command_available(name: str) -> bool:
    return shutil.which(name) is not None


def evaluate_environment(
    entry: ProofManifestEntry,
) -> tuple[tuple[EnvRequirementResult, ...], bool]:
    results: list[EnvRequirementResult] = []
    all_satisfied = True
    for requirement in entry.environment_requirements:
        if requirement.kind == EnvRequirementKind.ENV_PRESENT:
            satisfied = _env_present(requirement.name)
        elif requirement.kind == EnvRequirementKind.COMMAND_AVAILABLE:
            satisfied = _command_available(requirement.name)
        elif requirement.kind == EnvRequirementKind.DOCKER_AVAILABLE:
            satisfied = _docker_available()
        else:
            satisfied = False
        results.append(
            EnvRequirementResult(
                kind=requirement.kind,
                name=requirement.name,
                satisfied=satisfied,
            )
        )
        if not satisfied:
            all_satisfied = False
    return tuple(results), all_satisfied


def _platform_allowed(entry: ProofManifestEntry, platform_family: str) -> bool:
    if not entry.platform_requirements:
        return True
    return platform_family in entry.platform_requirements


def _should_skip_profile(
    entry: ProofManifestEntry,
    config: RunnerConfig,
) -> str | None:
    if config.dry_run:
        return "dry_run"
    if entry.safety_class == ProofSafetyClass.EXTERNAL_MUTATING:
        if not config.allow_external_mutating:
            return "external_mutating_opt_in_required"
    return None


def execute_proof(
    entry: ProofManifestEntry,
    *,
    repo_root: Path,
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> ProofRunResult:
    command = [entry.command.executable, *entry.command.argv]
    started = datetime.now(UTC)
    try:
        completed = subprocess_runner(
            command,
            cwd=repo_root,
            capture_output=True,
            text=False,
            timeout=entry.timeout_seconds,
            shell=False,
            env=os.environ.copy(),
        )
        duration = (datetime.now(UTC) - started).total_seconds()
        exit_code = completed.returncode
        status = ProofStatus.PASS if exit_code == 0 else ProofStatus.FAIL
        diagnostic = "child_exit_nonzero" if status == ProofStatus.FAIL else "child_exit_zero"
        return ProofRunResult(
            proof_id=entry.proof_id,
            status=status,
            duration_seconds=duration,
            exit_code=exit_code,
            diagnostic_summary=diagnostic,
        )
    except subprocess.TimeoutExpired:
        duration = (datetime.now(UTC) - started).total_seconds()
        return ProofRunResult(
            proof_id=entry.proof_id,
            status=ProofStatus.FAIL,
            duration_seconds=duration,
            exit_code=None,
            diagnostic_summary="timeout",
        )
    except FileNotFoundError:
        duration = (datetime.now(UTC) - started).total_seconds()
        return ProofRunResult(
            proof_id=entry.proof_id,
            status=ProofStatus.FAIL,
            duration_seconds=duration,
            exit_code=None,
            diagnostic_summary="missing_executable",
        )


def aggregate_overall_status(
    profile: ProofProfile,
    results: tuple[ProofRunResult, ...],
) -> SuiteOverallStatus:
    if any(result.status == ProofStatus.FAIL for result in results):
        return SuiteOverallStatus.FAIL

    if any(
        result.status == ProofStatus.BLOCKED_CONFIGURATION for result in results
    ):
        return SuiteOverallStatus.FAIL

    blocked_env = [
        result
        for result in results
        if result.status == ProofStatus.BLOCKED_ENVIRONMENT
    ]
    if blocked_env:
        if profile == ProofProfile.LIVE:
            return SuiteOverallStatus.PASS_WITH_BLOCKED
        return SuiteOverallStatus.FAIL

    return SuiteOverallStatus.PASS


def _count_results(results: tuple[ProofRunResult, ...]) -> tuple[int, int, int, int]:
    passed = 0
    failed = 0
    blocked = 0
    skipped = 0
    for result in results:
        if result.status == ProofStatus.PASS:
            passed += 1
        elif result.status == ProofStatus.FAIL:
            failed += 1
        elif result.status in {
            ProofStatus.BLOCKED_ENVIRONMENT,
            ProofStatus.BLOCKED_CONFIGURATION,
        }:
            blocked += 1
        elif result.status in {
            ProofStatus.SKIPPED_PLATFORM,
            ProofStatus.SKIPPED_PROFILE,
        }:
            skipped += 1
    return passed, failed, blocked, skipped


def render_console_summary(
    receipt: SuiteReceipt,
    *,
    verbose: bool = False,
) -> str:
    lines = [
        "INTERGRAX PROOF SUITE",
        f"profile: {receipt.profile.value}",
        f"commit: {receipt.git_commit_sha[:12]}",
        f"platform: {receipt.platform}",
        "",
    ]
    for result in receipt.results:
        lines.append(f"{result.status.value:<22} {result.proof_id}")
        if verbose and result.diagnostic_summary:
            lines.append(f"  diagnostic: {result.diagnostic_summary}")
    passed, failed, blocked, skipped = (
        receipt.passed_count,
        receipt.failed_count,
        receipt.blocked_count,
        receipt.skipped_count,
    )
    lines.extend(
        [
            "",
            f"overall: {receipt.overall_status.value}",
            f"passed: {passed}",
            f"failed: {failed}",
            f"blocked: {blocked}",
            f"skipped: {skipped}",
        ]
    )
    return "\n".join(lines)


def write_receipt(receipt: SuiteReceipt, *, repo_root: Path) -> Path:
    output_dir = repo_root / ".artifacts" / "proof"
    output_dir.mkdir(parents=True, exist_ok=True)
    short_sha = receipt.git_commit_sha[:12]
    timestamp = receipt.started_at.strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{timestamp}-{receipt.profile.value}-{short_sha}.json"
    path.write_text(
        json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def run_suite(
    config: RunnerConfig,
    *,
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> tuple[SuiteReceipt, Path | None]:
    started_at = datetime.now(UTC)
    suite_run_id = str(uuid.uuid4())
    platform_family = detect_platform_family()
    git = read_git_metadata(config.repo_root)

    try:
        manifest = load_manifest(repo_root=config.repo_root)
    except ManifestLoadError:
        receipt = SuiteReceipt(
            suite_run_id=suite_run_id,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            git_commit_sha=git.commit_sha,
            git_dirty=git.dirty,
            profile=config.profile,
            platform=platform_family,
            python_version=platform.python_version(),
            overall_status=SuiteOverallStatus.FAIL_MANIFEST,
            results=(),
            passed_count=0,
            failed_count=0,
            blocked_count=0,
            skipped_count=0,
        )
        return receipt, None

    selected = select_proofs(
        manifest,
        profile=config.profile,
        platform_family=platform_family,
    )
    results: list[ProofRunResult] = []

    for entry in selected:
        skip_reason = _should_skip_profile(entry, config)
        if skip_reason is not None:
            results.append(
                ProofRunResult(
                    proof_id=entry.proof_id,
                    status=ProofStatus.SKIPPED_PROFILE,
                    duration_seconds=0.0,
                    diagnostic_summary=skip_reason,
                )
            )
            continue

        if not _platform_allowed(entry, platform_family):
            results.append(
                ProofRunResult(
                    proof_id=entry.proof_id,
                    status=ProofStatus.SKIPPED_PLATFORM,
                    duration_seconds=0.0,
                    diagnostic_summary=f"requires_platform={sorted(entry.platform_requirements)}",
                )
            )
            continue

        env_results, env_ok = evaluate_environment(entry)
        if not env_ok:
            results.append(
                ProofRunResult(
                    proof_id=entry.proof_id,
                    status=ProofStatus.BLOCKED_ENVIRONMENT,
                    duration_seconds=0.0,
                    diagnostic_summary="environment_requirement_unsatisfied",
                    environment_requirements=env_results,
                )
            )
            continue

        child_result = execute_proof(
            entry,
            repo_root=config.repo_root,
            subprocess_runner=subprocess_runner,
        )
        child_result = child_result.model_copy(
            update={"environment_requirements": env_results}
        )
        results.append(child_result)

    result_tuple = tuple(results)
    overall = aggregate_overall_status(config.profile, result_tuple)
    if config.dry_run:
        overall = SuiteOverallStatus.DRY_RUN
    passed, failed, blocked, skipped = _count_results(result_tuple)
    completed_at = datetime.now(UTC)
    receipt = SuiteReceipt(
        suite_run_id=suite_run_id,
        started_at=started_at,
        completed_at=completed_at,
        git_commit_sha=git.commit_sha,
        git_dirty=git.dirty,
        profile=config.profile,
        platform=platform_family,
        python_version=platform.python_version(),
        overall_status=overall,
        results=result_tuple,
        passed_count=passed,
        failed_count=failed,
        blocked_count=blocked,
        skipped_count=skipped,
    )
    receipt_path = write_receipt(receipt, repo_root=config.repo_root)
    return receipt, receipt_path


def suite_exit_code(receipt: SuiteReceipt) -> int:
    if receipt.overall_status in {
        SuiteOverallStatus.PASS,
        SuiteOverallStatus.PASS_WITH_BLOCKED,
        SuiteOverallStatus.DRY_RUN,
    }:
        return 0
    return 1
