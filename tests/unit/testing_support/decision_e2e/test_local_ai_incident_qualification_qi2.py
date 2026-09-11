# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-QI2 live local qualification entrypoint integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from testing_support.decision_e2e.ai_incident_qualification_run import (
    AiIncidentQualificationRunOutcome,
    AiIncidentQualificationTraceEvidence,
)
from testing_support.decision_e2e.local_ai_incident_qualification import (
    QualificationCliExit,
    R4R1ProfileParams,
    build_r4r1_qualification_spec,
    qualification_observation_id_is_synthetic,
    run_local_ai_incident_qualification,
)
from testing_support.decision_e2e.local_qualification_session.artifact_contract import (
    DEFAULT_REQUIRED_ARTIFACTS,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationRuntimeIdentity,
    QualificationSessionState,
    QualificationSpec,
    TraceReadbackStatus,
)
from testing_support.decision_e2e.local_qualification_session.trace_readback import (
    read_typed_alignment_events,
)
from testing_support.decision_e2e.local_qualification_session.versioning import (
    ProviderRuntimeVersion,
)


@dataclass(frozen=True, slots=True)
class _RecordingExecutor:
    invocations: list[int]

    async def execute(self, *, run_index: int) -> AiIncidentQualificationRunOutcome:
        self.invocations.append(run_index)
        return AiIncidentQualificationRunOutcome(
            run_index=run_index,
            valid_model_trial=True,
            environment_event=False,
            run_id=None,
            runtime_execution_run_id=f"run_{run_index:032x}",
            qualification_observation_run_id=None,
            signals=None,
            run_result=None,
            block_reason=None,
            trace_evidence=AiIncidentQualificationTraceEvidence(
                runtime_execution_run_id=f"run_{run_index:032x}",
                trace_events=(),
                alignment_readback=read_typed_alignment_events((), trace_available=True),
                trace_correlation=TraceReadbackStatus.PASS,
            ),
        )


@dataclass(frozen=True, slots=True)
class _FixedProbe:
    observed: QualificationRuntimeIdentity | None

    def probe(self, *, model_name: str) -> QualificationRuntimeIdentity | None:
        return self.observed


def _observed(
    *,
    version: ProviderRuntimeVersion | None = ProviderRuntimeVersion(0, 33, 3),
    digest: str = "digest-a",
) -> QualificationRuntimeIdentity:
    return QualificationRuntimeIdentity(
        provider_kind="ollama",
        runtime_version=version,
        endpoint_host="http://127.0.0.1:11434",
        model_name="qwen2.5:14b",
        model_digest=digest,
        quantization="Q4_K_M",
    )


def _spec_bundle(
    repo_root: Path,
    *,
    run_count: int = 1,
    digest: str = "digest-a",
) -> tuple[QualificationSpec, str, str]:
    spec, config_fp, frozen = build_r4r1_qualification_spec(
        repo_root,
        params=R4R1ProfileParams(model_digest=digest, run_count=run_count),
        repository_head_sha="test-head",
    )
    return spec, config_fp, frozen.semantic_fingerprint()


@pytest.mark.asyncio
async def test_wrong_ollama_version_blocks_before_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    spec, config_fp, source_fp = _spec_bundle(repo_root)
    executor = _RecordingExecutor(invocations=[])
    result = await run_local_ai_incident_qualification(
        repo_root=repo_root,
        session_dir=tmp_path,
        spec=spec,
        config_fingerprint=config_fp,
        source_fingerprint=source_fp,
        provider_probe=_FixedProbe(_observed(version=ProviderRuntimeVersion(0, 34, 0))),
        run_executor=executor,
    )
    assert result.exit_code is QualificationCliExit.BLOCKED_PRECONDITION
    assert executor.invocations == []


@pytest.mark.asyncio
async def test_provider_unavailable_blocks_zero_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    spec, config_fp, source_fp = _spec_bundle(repo_root)
    executor = _RecordingExecutor(invocations=[])
    result = await run_local_ai_incident_qualification(
        repo_root=repo_root,
        session_dir=tmp_path,
        spec=spec,
        config_fingerprint=config_fp,
        source_fingerprint=source_fp,
        provider_probe=_FixedProbe(None),
        run_executor=executor,
    )
    assert result.exit_code is QualificationCliExit.BLOCKED_PRECONDITION
    assert executor.invocations == []


@pytest.mark.asyncio
async def test_model_digest_mismatch_blocks(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    spec, config_fp, source_fp = _spec_bundle(repo_root, digest="digest-a")
    executor = _RecordingExecutor(invocations=[])
    result = await run_local_ai_incident_qualification(
        repo_root=repo_root,
        session_dir=tmp_path,
        spec=spec,
        config_fingerprint=config_fp,
        source_fingerprint=source_fp,
        provider_probe=_FixedProbe(_observed(digest="digest-b")),
        run_executor=executor,
    )
    assert result.exit_code is QualificationCliExit.BLOCKED_PRECONDITION
    assert executor.invocations == []


@pytest.mark.asyncio
async def test_config_mismatch_blocks(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    spec, config_fp, source_fp = _spec_bundle(repo_root)
    executor = _RecordingExecutor(invocations=[])
    result = await run_local_ai_incident_qualification(
        repo_root=repo_root,
        session_dir=tmp_path,
        spec=spec,
        config_fingerprint="wrong-config",
        source_fingerprint=source_fp,
        provider_probe=_FixedProbe(_observed()),
        run_executor=executor,
    )
    assert result.exit_code is QualificationCliExit.BLOCKED_PRECONDITION
    assert executor.invocations == []


@pytest.mark.asyncio
async def test_one_canonical_run_updates_registry_and_checkpoint(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    spec, config_fp, source_fp = _spec_bundle(repo_root, run_count=1)
    executor = _RecordingExecutor(invocations=[])
    result = await run_local_ai_incident_qualification(
        repo_root=repo_root,
        session_dir=tmp_path,
        spec=spec,
        config_fingerprint=config_fp,
        source_fingerprint=source_fp,
        provider_probe=_FixedProbe(_observed(digest=spec.experiment_identity.model_digest or "")),
        run_executor=executor,
    )
    assert result.exit_code is QualificationCliExit.SUCCESS
    assert executor.invocations == [0]
    assert (tmp_path / "runs.json").is_file()
    assert (tmp_path / "session-checkpoint.json").is_file()
    for name in DEFAULT_REQUIRED_ARTIFACTS:
        assert (tmp_path / name).is_file()


@pytest.mark.asyncio
async def test_resume_skips_completed_runs(tmp_path: Path) -> None:
    from testing_support.decision_e2e.local_qualification_session.contracts import (
        CanonicalRunRecord,
    )
    from testing_support.decision_e2e.local_qualification_session.session import (
        LocalQualificationSession,
    )
    from testing_support.decision_e2e.local_qualification_session.source_fingerprint import (
        capture_semantic_source_fingerprint,
    )

    repo_root = Path(__file__).resolve().parents[4]
    spec, config_fp, source_fp = _spec_bundle(repo_root, run_count=2)
    probe = _FixedProbe(_observed(digest=spec.experiment_identity.model_digest or ""))
    frozen = capture_semantic_source_fingerprint(
        repo_root,
        semantic_source_groups=dict(spec.semantic_source_groups),
        repository_head_sha="test-head",
    )
    session = LocalQualificationSession(
        session_dir=tmp_path,
        spec=spec,
        frozen_source=frozen,
    )
    start = session.start(
        probe.probe(model_name=spec.experiment_identity.model_name),
        config_fingerprint=config_fp,
        source_fingerprint=source_fp,
    )
    assert start.state is QualificationSessionState.RUNNING
    session.persist_canonical_run(
        CanonicalRunRecord(run_index=0, run_id="run_00000000000000000000000000000000", trace_events=())
    )
    executor = _RecordingExecutor(invocations=[])
    resumed = await run_local_ai_incident_qualification(
        repo_root=repo_root,
        session_dir=tmp_path,
        spec=spec,
        config_fingerprint=config_fp,
        source_fingerprint=source_fp,
        provider_probe=probe,
        run_executor=executor,
        resume=True,
    )
    assert executor.invocations == [1]
    assert resumed.exit_code is QualificationCliExit.SUCCESS


@pytest.mark.asyncio
async def test_finalize_only_makes_no_executor_calls(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    spec, config_fp, source_fp = _spec_bundle(repo_root, run_count=1)
    executor = _RecordingExecutor(invocations=[])
    probe = _FixedProbe(_observed(digest=spec.experiment_identity.model_digest or ""))
    await run_local_ai_incident_qualification(
        repo_root=repo_root,
        session_dir=tmp_path,
        spec=spec,
        config_fingerprint=config_fp,
        source_fingerprint=source_fp,
        provider_probe=probe,
        run_executor=executor,
    )
    executor.invocations.clear()
    result = await run_local_ai_incident_qualification(
        repo_root=repo_root,
        session_dir=tmp_path,
        spec=spec,
        config_fingerprint=config_fp,
        source_fingerprint=source_fp,
        provider_probe=_FixedProbe(None),
        run_executor=executor,
        finalize_only=True,
    )
    assert executor.invocations == []
    assert result.exit_code in {
        QualificationCliExit.SUCCESS,
        QualificationCliExit.FAILED_FINALIZATION,
    }


def test_trace_pass_zero_events_is_pass_not_failed() -> None:
    readback = read_typed_alignment_events((), trace_available=True)
    assert readback.status is TraceReadbackStatus.PASS


def test_qualification_observation_id_is_synthetic() -> None:
    assert qualification_observation_id_is_synthetic("qual-obs-abc")
    assert not qualification_observation_id_is_synthetic("run_0123456789abcdef0123456789abcdef")
