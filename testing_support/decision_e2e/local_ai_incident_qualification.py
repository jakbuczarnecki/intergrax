# © Artur Czarnecki. All rights reserved.

"""Live local AI Incident qualification orchestration (DS-E2E-15J-QI2)."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Protocol

from testing_support.decision_e2e.ai_incident_qualification_run import (
    CANONICAL_SCENARIO_INPUT_IDENTITY,
    AiIncidentQualificationRunOutcome,
    QUALIFICATION_OBSERVATION_ID_PREFIX,
    execute_ai_incident_qualification_run,
)
from testing_support.decision_e2e.local_qualification_session.artifact_contract import (
    DEFAULT_REQUIRED_ARTIFACTS,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    CanonicalRunRecord,
    QualificationExperimentIdentity,
    QualificationRuntimeIdentity,
    QualificationSessionState,
    QualificationSpec,
    SourceFingerprintSnapshot,
    VersionMatchPolicy,
)
from testing_support.decision_e2e.local_qualification_session.ollama_probe import (
    OllamaProbeConfig,
    probe_ollama_runtime_identity,
)
from testing_support.decision_e2e.local_qualification_session.checkpoint import (
    IllegalSessionTransitionError,
)
from testing_support.decision_e2e.local_qualification_session.session import (
    LocalQualificationSession,
)
from testing_support.decision_e2e.local_qualification_session.source_fingerprint import (
    capture_semantic_source_fingerprint,
)
from testing_support.decision_e2e.local_qualification_session.versioning import (
    ProviderRuntimeVersion,
)
from testing_support.decision_e2e.scenario_qualification import AI_INCIDENT_SCENARIO_ID

R4R1_PROFILE_ID = "DS-E2E-15J-L1.R4.R1"
R4R1_TASK_ID = "DS-E2E-15J-L1.R4.R1"
R4R5_PROFILE_ID = "DS-E2E-15J-L1.R4.R5"
R4R5_TASK_ID = "DS-E2E-15J-L1.R4.R5"
R4R1_PROVIDER = "ollama"
R4R1_RUNTIME_VERSION = ProviderRuntimeVersion(0, 34, 0)
R4R1_MODEL_NAME = "qwen2.5:14b"
R4R1_QUANTIZATION = "Q4_K_M"
R4R1_TEMPERATURE = 0.0
R4R1_RUN_COUNT = 20
R4R1_EVALUATOR_MAX_ITERATIONS = 2
R4R1_MAX_DECISION_REVISIONS = 0
R4R1_SEMANTIC_VERIFICATION = False
R4R1_MODEL_DIGEST_PLACEHOLDER = "sha256:7cdf5a0187d5c8e0b4a1f2e3d4c5b6a7f8e9d0c1b2a3f4e5d6c7b8a9b0c1d2e"


class QualificationCliExit(IntEnum):
    SUCCESS = 0
    BLOCKED_PRECONDITION = 2
    PARTIAL_OR_INVALID_SESSION = 3
    FAILED_FINALIZATION = 4
    CRITICAL_SAFETY_FAILURE = 5


def semantic_source_groups_for_r4r1() -> dict[str, tuple[str, ...]]:
    return {
        "15I": (
            "platform_proofs/scenarios/ai_incident_investigation/application/completion_revision_context.py",
        ),
        "L0": (
            "testing_support/decision_e2e/scenario_qualification.py",
        ),
        "T1": (
            "intergrax/decision_system/qualification/taxonomy.py",
        ),
        "C1": (
            "intergrax/decision_system/qualification/classifier.py",
        ),
        "15K-B": (
            "platform_proofs/scenarios/ai_incident_investigation/application/completion_alignment.py",
            "platform_proofs/scenarios/ai_incident_investigation/application/completion_alignment_correction.py",
        ),
        "QI1": (
            "testing_support/decision_e2e/local_qualification_session/session.py",
            "testing_support/decision_e2e/local_qualification_session/contracts.py",
            "testing_support/decision_e2e/local_qualification_session/run_registry.py",
            "testing_support/decision_e2e/local_qualification_session/finalization.py",
        ),
        "QI2": (
            "testing_support/decision_e2e/local_ai_incident_qualification.py",
        ),
        "ai_incident_executor": (
            "testing_support/decision_e2e/ai_incident_qualification_run.py",
        ),
        "evaluator_revision": (
            "platform_proofs/scenarios/ai_incident_investigation/application/investigator_agent.py",
            "platform_proofs/scenarios/ai_incident_investigation/application/observability.py",
        ),
        "O1": (
            "intergrax/runtime/nexus/tracing/execution/evaluator_model_attempt.py",
            "intergrax/runtime/nexus/tracing/execution/reconciliation_phase.py",
            "intergrax/runtime/observability/qualification_runtime_trace.py",
            "intergrax/runtime/nexus/execution/graph_executor.py",
            "intergrax/runtime/nexus/orchestration/graph_trace_callbacks.py",
            "intergrax/runtime/nexus/orchestration/graph_runner.py",
            "platform_proofs/scenarios/ai_incident_investigation/application/scenario.py",
        ),
        "O2": (
            "intergrax/runtime/diagnostics/completion_alignment_diag.py",
            "intergrax/runtime/observability/qualification_runtime_trace.py",
            "platform_proofs/scenarios/ai_incident_investigation/application/completion_alignment_telemetry.py",
            "platform_proofs/scenarios/ai_incident_investigation/application/investigator_agent.py",
            "platform_proofs/scenarios/ai_incident_investigation/application/scenario.py",
            "testing_support/decision_e2e/completion_alignment_producer_reachability.py",
            "testing_support/decision_e2e/local_qualification_session/trace_readback.py",
            "testing_support/decision_e2e/local_qualification_session/behavioral_coverage_evidence.py",
        ),
    }


def flattened_source_blob_paths(groups: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for paths in groups.values():
        for path in paths:
            if path not in seen:
                seen.add(path)
                ordered.append(path)
    return tuple(ordered)


def resolve_repository_head_sha(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip()


def build_generation_config_fingerprint(
    *,
    temperature: float,
    semantic_verification: bool,
    evaluator_max_iterations: int,
    max_decision_revisions: int,
) -> str:
    payload = (
        f"temperature={temperature}|"
        f"semantic_verification={semantic_verification}|"
        f"evaluator_max_iterations={evaluator_max_iterations}|"
        f"max_decision_revisions={max_decision_revisions}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_qualification_config_fingerprint(
    *,
    provider: str,
    model: str,
    generation_config_fingerprint: str,
    scenario_id: str,
    input_id: str,
) -> str:
    payload = (
        f"provider={provider}|model={model}|"
        f"generation={generation_config_fingerprint}|"
        f"scenario={scenario_id}|input={input_id}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_fingerprint_matches(
    expected: str,
    *,
    provider: str,
    model: str,
    generation_config_fingerprint: str,
    scenario_id: str,
    input_id: str,
) -> bool:
    return expected == build_qualification_config_fingerprint(
        provider=provider,
        model=model,
        generation_config_fingerprint=generation_config_fingerprint,
        scenario_id=scenario_id,
        input_id=input_id,
    )


@dataclass(frozen=True, slots=True)
class R4R1ProfileParams:
    model_digest: str
    run_count: int = R4R1_RUN_COUNT
    temperature: float = R4R1_TEMPERATURE
    semantic_verification: bool = R4R1_SEMANTIC_VERIFICATION
    evaluator_max_iterations: int = R4R1_EVALUATOR_MAX_ITERATIONS
    max_decision_revisions: int = R4R1_MAX_DECISION_REVISIONS


def build_r4r1_qualification_spec(
    repo_root: Path,
    *,
    params: R4R1ProfileParams,
    repository_head_sha: str,
) -> tuple[QualificationSpec, str, SourceFingerprintSnapshot]:
    groups = semantic_source_groups_for_r4r1()
    blob_paths = flattened_source_blob_paths(groups)
    frozen_source = capture_semantic_source_fingerprint(
        repo_root,
        semantic_source_groups=groups,
        repository_head_sha=repository_head_sha,
    )
    generation_fp = build_generation_config_fingerprint(
        temperature=params.temperature,
        semantic_verification=params.semantic_verification,
        evaluator_max_iterations=params.evaluator_max_iterations,
        max_decision_revisions=params.max_decision_revisions,
    )
    config_fp = build_qualification_config_fingerprint(
        provider=R4R1_PROVIDER,
        model=R4R1_MODEL_NAME,
        generation_config_fingerprint=generation_fp,
        scenario_id=AI_INCIDENT_SCENARIO_ID,
        input_id=CANONICAL_SCENARIO_INPUT_IDENTITY,
    )
    identity = QualificationExperimentIdentity(
        provider_kind=R4R1_PROVIDER,
        provider_runtime_version=R4R1_RUNTIME_VERSION,
        provider_runtime_version_policy=VersionMatchPolicy.EXACT,
        model_name=R4R1_MODEL_NAME,
        model_digest=params.model_digest,
        model_digest_policy=VersionMatchPolicy.EXACT,
        quantization=R4R1_QUANTIZATION,
        generation_config_fingerprint=generation_fp,
        scenario_id=AI_INCIDENT_SCENARIO_ID,
        input_id=CANONICAL_SCENARIO_INPUT_IDENTITY,
        source_fingerprint=frozen_source.semantic_fingerprint(),
        config_fingerprint=config_fp,
    )
    planned = params.run_count
    checkpoints = tuple(
        index for index in (0, 10, 19) if 0 <= index < planned
    ) or (0,)
    spec = QualificationSpec(
        experiment_identity=identity,
        run_count=planned,
        required_artifacts=DEFAULT_REQUIRED_ARTIFACTS,
        source_blob_paths=blob_paths,
        semantic_source_groups=groups,
        max_evaluator_attempt_index=max(0, params.evaluator_max_iterations - 1),
        source_checkpoint_run_indices=checkpoints,
    )
    return spec, config_fp, frozen_source


def build_r4r5_qualification_spec(
    repo_root: Path,
    *,
    params: R4R1ProfileParams,
    repository_head_sha: str,
) -> tuple[QualificationSpec, str, SourceFingerprintSnapshot]:
    return build_r4r1_qualification_spec(
        repo_root,
        params=params,
        repository_head_sha=repository_head_sha,
    )


class ProviderIdentityProbe(Protocol):
    def probe(self, *, model_name: str) -> QualificationRuntimeIdentity | None:
        """Return observed provider identity or None when unavailable."""


class QualificationRunExecutor(Protocol):
    async def execute(self, *, run_index: int) -> AiIncidentQualificationRunOutcome:
        """Execute one canonical qualification run."""
        ...


@dataclass(frozen=True, slots=True)
class OllamaProviderIdentityProbe:
    config: OllamaProbeConfig

    def probe(self, *, model_name: str) -> QualificationRuntimeIdentity | None:
        return probe_ollama_runtime_identity(self.config, model_name=model_name)


@dataclass(frozen=True, slots=True)
class AiIncidentSingleRunExecutor:
    async def execute(self, *, run_index: int) -> AiIncidentQualificationRunOutcome:
        return await execute_ai_incident_qualification_run(run_index=run_index)


def canonical_run_id_from_outcome(outcome: AiIncidentQualificationRunOutcome) -> str | None:
    if outcome.environment_event:
        return None
    if outcome.runtime_execution_run_id is not None:
        return outcome.runtime_execution_run_id
    if outcome.qualification_observation_run_id is not None:
        return outcome.qualification_observation_run_id
    return None


def trace_events_from_outcome(
    outcome: AiIncidentQualificationRunOutcome,
) -> tuple[dict[str, object], ...]:
    if outcome.trace_evidence is None:
        return ()
    return outcome.trace_evidence.trace_events


@dataclass(frozen=True, slots=True)
class LocalQualificationOrchestrationResult:
    exit_code: QualificationCliExit
    session_state: QualificationSessionState
    executor_invocations: tuple[int, ...]


async def run_local_ai_incident_qualification(
    *,
    repo_root: Path,
    session_dir: Path,
    spec: QualificationSpec,
    config_fingerprint: str,
    source_fingerprint: str,
    provider_probe: ProviderIdentityProbe,
    run_executor: QualificationRunExecutor,
    resume: bool = False,
    finalize_only: bool = False,
    repository_head_sha: str | None = None,
    task_id: str = R4R1_TASK_ID,
    temperature: float = R4R1_TEMPERATURE,
) -> LocalQualificationOrchestrationResult:
    head_sha = repository_head_sha or resolve_repository_head_sha(repo_root)
    groups = dict(spec.semantic_source_groups)
    frozen_source = capture_semantic_source_fingerprint(
        repo_root,
        semantic_source_groups=groups,
        repository_head_sha=head_sha,
    )

    session = LocalQualificationSession(
        session_dir=session_dir,
        spec=spec,
        frozen_source=frozen_source,
        task_id=task_id,
    )
    invocations: list[int] = []

    if finalize_only:
        try:
            _prepare_finalize_only_session(session)
        except IllegalSessionTransitionError:
            return LocalQualificationOrchestrationResult(
                exit_code=QualificationCliExit.PARTIAL_OR_INVALID_SESSION,
                session_state=session.state,
                executor_invocations=(),
            )
        observed = provider_probe.probe(model_name=spec.experiment_identity.model_name)
        integrity = session.integrity_report(
            observed,
            config_matches=True,
            source_matches=True,
        )
        finalized = session.finalize(
            integrity=integrity,
            observed=observed,
            temperature=temperature,
            regenerate_derived=True,
        )
        state = finalized.finalization_status
        exit_code = (
            QualificationCliExit.SUCCESS
            if state is QualificationSessionState.FINALIZED
            else QualificationCliExit.FAILED_FINALIZATION
        )
        return LocalQualificationOrchestrationResult(
            exit_code=exit_code,
            session_state=state,
            executor_invocations=(),
        )

    observed = provider_probe.probe(model_name=spec.experiment_identity.model_name)

    if session.state is QualificationSessionState.CREATED:
        start = session.start(
            observed,
            config_fingerprint=config_fingerprint,
            source_fingerprint=source_fingerprint,
        )
        if start.state is not QualificationSessionState.RUNNING:
            return LocalQualificationOrchestrationResult(
                exit_code=QualificationCliExit.BLOCKED_PRECONDITION,
                session_state=start.state,
                executor_invocations=(),
            )
    elif resume:
        precondition = session.assert_resume_identity(
            observed,
            config_fingerprint=config_fingerprint,
            source_fingerprint=source_fingerprint,
        )
        if not precondition.eligible or session.state is QualificationSessionState.BLOCKED:
            return LocalQualificationOrchestrationResult(
                exit_code=QualificationCliExit.BLOCKED_PRECONDITION,
                session_state=session.state,
                executor_invocations=(),
            )
    else:
        return LocalQualificationOrchestrationResult(
            exit_code=QualificationCliExit.PARTIAL_OR_INVALID_SESSION,
            session_state=session.state,
            executor_invocations=(),
        )

    pending = session.pending_run_indices()
    for run_index in pending:
        current_source = capture_semantic_source_fingerprint(
            repo_root,
            semantic_source_groups=groups,
            repository_head_sha=resolve_repository_head_sha(repo_root),
        )
        drift = session.checkpoint_source_drift(current_source)
        if drift.qualification_semantic_source_drift:
            return LocalQualificationOrchestrationResult(
                exit_code=QualificationCliExit.PARTIAL_OR_INVALID_SESSION,
                session_state=session.state,
                executor_invocations=tuple(invocations),
            )
        outcome = await run_executor.execute(run_index=run_index)
        invocations.append(run_index)
        canonical_id = canonical_run_id_from_outcome(outcome)
        if canonical_id is None:
            continue
        session.persist_canonical_run(
            CanonicalRunRecord(
                run_index=run_index,
                run_id=canonical_id,
                trace_events=trace_events_from_outcome(outcome),
            )
        )
        _append_qualification_run_log(
            session_dir,
            run_index=run_index,
            run_id=canonical_id,
        )

    if session.pending_run_indices():
        return LocalQualificationOrchestrationResult(
            exit_code=QualificationCliExit.PARTIAL_OR_INVALID_SESSION,
            session_state=session.state,
            executor_invocations=tuple(invocations),
        )

    observed_after = provider_probe.probe(model_name=spec.experiment_identity.model_name)
    source_matches = (
        session.frozen_source.semantic_fingerprint()
        == spec.experiment_identity.source_fingerprint
    )
    integrity = session.integrity_report(
        observed_after,
        config_matches=config_fingerprint_matches(
            spec.experiment_identity.config_fingerprint,
            provider=spec.experiment_identity.provider_kind,
            model=spec.experiment_identity.model_name,
            generation_config_fingerprint=spec.experiment_identity.generation_config_fingerprint,
            scenario_id=spec.experiment_identity.scenario_id,
            input_id=spec.experiment_identity.input_id,
        ),
        source_matches=source_matches,
    )
    finalized = session.finalize(
        integrity=integrity,
        observed=observed_after,
        temperature=temperature,
    )
    state = finalized.finalization_status
    if state is QualificationSessionState.FINALIZED:
        exit_code = QualificationCliExit.SUCCESS
    elif state is QualificationSessionState.FAILED_FINALIZATION:
        exit_code = QualificationCliExit.FAILED_FINALIZATION
    elif state in {QualificationSessionState.BLOCKED, QualificationSessionState.INVALID}:
        exit_code = QualificationCliExit.BLOCKED_PRECONDITION
    else:
        exit_code = QualificationCliExit.PARTIAL_OR_INVALID_SESSION
    return LocalQualificationOrchestrationResult(
        exit_code=exit_code,
        session_state=state,
        executor_invocations=tuple(invocations),
    )


def qualification_observation_id_is_synthetic(run_id: str) -> bool:
    return run_id.startswith(QUALIFICATION_OBSERVATION_ID_PREFIX)


def _append_qualification_run_log(session_dir: Path, *, run_index: int, run_id: str) -> None:
    path = session_dir / "run.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"run_index={run_index} run_id={run_id}\n")


def _prepare_finalize_only_session(session: LocalQualificationSession) -> None:
    session.prepare_finalize_only()
