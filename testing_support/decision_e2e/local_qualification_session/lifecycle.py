# © Artur Czarnecki. All rights reserved.

"""Non-behavioral dry-run qualification lifecycle (DS-E2E-15J-QI1 / R4.R1 readiness)."""

from __future__ import annotations

from pathlib import Path

from testing_support.decision_e2e.local_qualification_session.artifact_contract import (
    DEFAULT_REQUIRED_ARTIFACTS,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    CanonicalRunRecord,
    QualificationExperimentIdentity,
    QualificationRuntimeIdentity,
    QualificationSessionState,
    QualificationSpec,
    VersionMatchPolicy,
)
from testing_support.decision_e2e.local_qualification_session.session import (
    LocalQualificationSession,
)
from testing_support.decision_e2e.local_qualification_session.source_fingerprint import (
    capture_source_fingerprint,
)
from testing_support.decision_e2e.local_qualification_session.versioning import (
    ProviderRuntimeVersion,
)


def run_qualification_lifecycle_dry_run(
    *,
    repo_root: Path,
    session_dir: Path,
    source_blob_paths: tuple[str, ...],
    repository_head_sha: str,
) -> QualificationSessionState:
    frozen_source = capture_source_fingerprint(
        repo_root,
        blob_paths=source_blob_paths,
        semantic_group="qi1-dry-run",
        repository_head_sha=repository_head_sha,
    )
    identity = QualificationExperimentIdentity(
        provider_kind="ollama",
        provider_runtime_version=ProviderRuntimeVersion(0, 33, 3),
        provider_runtime_version_policy=VersionMatchPolicy.EXACT,
        model_name="qwen2.5:14b",
        model_digest="sha256:dry-run",
        model_digest_policy=VersionMatchPolicy.EXACT,
        quantization=None,
        generation_config_fingerprint="dry-run",
        scenario_id="ai_incident_investigation",
        input_id="ai_incident_investigation:resolved:canonical",
        source_fingerprint=frozen_source.semantic_fingerprint(),
        config_fingerprint="dry-run-config",
    )
    spec = QualificationSpec(
        experiment_identity=identity,
        run_count=1,
        required_artifacts=DEFAULT_REQUIRED_ARTIFACTS,
        source_blob_paths=source_blob_paths,
        semantic_source_groups={"qi1-dry-run": source_blob_paths},
        max_evaluator_attempt_index=1,
        source_checkpoint_run_indices=(0,),
    )
    observed = QualificationRuntimeIdentity(
        provider_kind="ollama",
        runtime_version=ProviderRuntimeVersion(0, 33, 3),
        endpoint_host="http://127.0.0.1:11434",
        model_name="qwen2.5:14b",
        model_digest="sha256:dry-run",
        quantization=None,
    )
    session = LocalQualificationSession(
        session_dir=session_dir,
        spec=spec,
        frozen_source=frozen_source,
    )
    start = session.start(
        observed,
        config_fingerprint="dry-run-config",
        source_fingerprint=frozen_source.semantic_fingerprint(),
    )
    if start.state is not QualificationSessionState.RUNNING:
        return start.state
    session.persist_canonical_run(
        CanonicalRunRecord(run_index=0, run_id="run-dry-0", trace_events=())
    )
    integrity = session.integrity_report(
        observed,
        config_matches=True,
        source_matches=True,
    )
    finalized_integrity = session.finalize(integrity=integrity)
    return finalized_integrity.finalization_status
