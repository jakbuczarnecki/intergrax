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
from scripts.proof.intergrax_platform_proof_artifact_verifier import (
    apply_artifact_verification,
    verify_platform_proof_artifacts,
)
from scripts.proof.intergrax_platform_proof_evidence_verifier import (
    EvidenceVerificationResult,
    apply_evidence_verification,
    resolve_expected_evidence_path,
    verify_all_platform_proof_evidence,
    verify_platform_proof_evidence,
)
from scripts.proof.intergrax_platform_proof_publication import (
    apply_canonical_publication,
)
from scripts.proof.intergrax_platform_proof_artifact_verifier import (
    ProofArtifactVerificationSummary,
)
from scripts.proof.intergrax_platform_proof_execution import (
    INTERGRAX_PROOF_ARTIFACT_DIR_ENV,
    ProofExecutionSpec,
    load_manifest_bundle,
    proof_run_artifact_directory,
    suite_run_artifact_directory,
)
from scripts.proof.intergrax_proof_manifest import (
    ManifestLoadError,
    expanded_profiles,
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
    execution_spec: ProofExecutionSpec | None = None,
    proof_artifact_directory: Path | None = None,
    git_commit_sha: str = "unknown",
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> ProofRunResult:
    command = [entry.command.executable, *entry.command.argv]
    started = datetime.now(UTC)
    env = os.environ.copy()
    if proof_artifact_directory is not None:
        proof_artifact_directory.mkdir(parents=True, exist_ok=True)
        env[INTERGRAX_PROOF_ARTIFACT_DIR_ENV] = str(proof_artifact_directory.resolve())
    try:
        completed = subprocess_runner(
            command,
            cwd=repo_root,
            capture_output=True,
            text=False,
            timeout=entry.timeout_seconds,
            shell=False,
            env=env,
        )
        duration = (datetime.now(UTC) - started).total_seconds()
        exit_code = completed.returncode
        status = ProofStatus.PASS if exit_code == 0 else ProofStatus.FAIL
        diagnostic = "child_exit_nonzero" if status == ProofStatus.FAIL else "child_exit_zero"
        transport_result = ProofRunResult(
            proof_id=entry.proof_id,
            status=status,
            duration_seconds=duration,
            exit_code=exit_code,
            diagnostic_summary=diagnostic,
        )
        return _verify_post_execution(
            transport_result,
            execution_spec=execution_spec,
            proof_artifact_directory=proof_artifact_directory,
            git_commit_sha=git_commit_sha,
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
    run_directory = suite_run_artifact_directory(repo_root, receipt.suite_run_id)
    run_directory.mkdir(parents=True, exist_ok=True)
    path = run_directory / "suite-receipt.json"
    path.write_text(
        json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _verify_artifacts_if_declared(
    transport_result: ProofRunResult,
    *,
    execution_spec: ProofExecutionSpec,
    proof_artifact_directory: Path | None,
) -> ProofRunResult:
    if not execution_spec.expected_artifacts:
        return transport_result
    if proof_artifact_directory is None:
        return transport_result.model_copy(
            update={
                "status": ProofStatus.FAIL,
                "diagnostic_summary": "missing_proof_artifact_directory",
            }
        )
    summary = verify_platform_proof_artifacts(
        proof_artifact_directory=proof_artifact_directory,
        spec=execution_spec,
    )
    return apply_artifact_verification(transport_result, summary)


def _verify_evidence_if_required(
    transport_result: ProofRunResult,
    *,
    execution_spec: ProofExecutionSpec | None,
    proof_artifact_directory: Path | None,
    git_commit_sha: str,
) -> ProofRunResult:
    if execution_spec is None or not execution_spec.evidence_required:
        return transport_result
    if proof_artifact_directory is None:
        return transport_result.model_copy(
            update={
                "status": ProofStatus.FAIL,
                "diagnostic_summary": "missing_proof_artifact_directory",
            }
        )
    evidence_path = resolve_expected_evidence_path(
        proof_artifact_directory,
        execution_spec,
    )
    verification = verify_all_platform_proof_evidence(
        proof_artifact_directory=proof_artifact_directory,
        spec=execution_spec,
        subprocess_result=transport_result,
        expected_source_revision=git_commit_sha,
    )
    return apply_evidence_verification(transport_result, verification)


def _verify_post_execution(
    transport_result: ProofRunResult,
    *,
    execution_spec: ProofExecutionSpec | None,
    proof_artifact_directory: Path | None,
    git_commit_sha: str,
) -> ProofRunResult:
    """Artifact verification, evidence verification, then canonical publication."""
    result = transport_result
    artifact_summary: ProofArtifactVerificationSummary | None = None
    if execution_spec is not None and execution_spec.expected_artifacts:
        if proof_artifact_directory is None:
            result = transport_result.model_copy(
                update={
                    "status": ProofStatus.FAIL,
                    "diagnostic_summary": "missing_proof_artifact_directory",
                }
            )
        else:
            artifact_summary = verify_platform_proof_artifacts(
                proof_artifact_directory=proof_artifact_directory,
                spec=execution_spec,
            )
            result = apply_artifact_verification(result, artifact_summary)

    evidence_verification: EvidenceVerificationResult | None = None
    if execution_spec is not None and execution_spec.evidence_required:
        if proof_artifact_directory is None:
            result = result.model_copy(
                update={
                    "status": ProofStatus.FAIL,
                    "diagnostic_summary": "missing_proof_artifact_directory",
                }
            )
        else:
            evidence_verification = verify_all_platform_proof_evidence(
                proof_artifact_directory=proof_artifact_directory,
                spec=execution_spec,
                subprocess_result=result,
                expected_source_revision=git_commit_sha,
            )
            result = apply_evidence_verification(result, evidence_verification)

    return apply_canonical_publication(
        result,
        transport_result=transport_result,
        execution_spec=execution_spec,
        candidate_directory=proof_artifact_directory,
        artifact_summary=artifact_summary,
        evidence_verification=evidence_verification,
    )


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
        bundle = load_manifest_bundle(repo_root=config.repo_root)
        manifest = bundle.manifest
        execution_specs = bundle.execution_specs
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

        proof_artifact_dir = proof_run_artifact_directory(
            config.repo_root,
            suite_run_id,
            entry.proof_id,
        )
        child_result = execute_proof(
            entry,
            repo_root=config.repo_root,
            execution_spec=execution_specs.get(entry.proof_id),
            proof_artifact_directory=proof_artifact_dir,
            git_commit_sha=git.commit_sha,
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
