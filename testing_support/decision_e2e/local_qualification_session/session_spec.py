# © Artur Czarnecki. All rights reserved.

"""Serialization for QualificationSpec and experiment identity."""

from __future__ import annotations

from testing_support.decision_e2e.local_qualification_session.artifact_contract import (
    DEFAULT_REQUIRED_ARTIFACTS,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationExperimentIdentity,
    QualificationSpec,
    VersionMatchPolicy,
)
from testing_support.decision_e2e.local_qualification_session.versioning import (
    parse_provider_runtime_version,
)


def spec_to_dict(spec: QualificationSpec) -> dict[str, object]:
    identity = spec.experiment_identity
    runtime_version = (
        identity.provider_runtime_version.normalized()
        if identity.provider_runtime_version is not None
        else None
    )
    return {
        "experiment_identity": {
            "provider_kind": identity.provider_kind,
            "provider_runtime_version": runtime_version,
            "provider_runtime_version_policy": identity.provider_runtime_version_policy.value,
            "model_name": identity.model_name,
            "model_digest": identity.model_digest,
            "model_digest_policy": identity.model_digest_policy.value,
            "quantization": identity.quantization,
            "generation_config_fingerprint": identity.generation_config_fingerprint,
            "scenario_id": identity.scenario_id,
            "input_id": identity.input_id,
            "source_fingerprint": identity.source_fingerprint,
            "config_fingerprint": identity.config_fingerprint,
        },
        "run_count": spec.run_count,
        "required_artifacts": list(spec.required_artifacts),
        "source_blob_paths": list(spec.source_blob_paths),
        "semantic_source_groups": {
            key: list(value) for key, value in spec.semantic_source_groups.items()
        },
        "max_evaluator_attempt_index": spec.max_evaluator_attempt_index,
        "source_checkpoint_run_indices": list(spec.source_checkpoint_run_indices),
    }


def spec_from_dict(raw: dict[str, object]) -> QualificationSpec:
    identity_raw = raw.get("experiment_identity")
    if not isinstance(identity_raw, dict):
        raise ValueError("spec.experiment_identity required")
    runtime_raw = identity_raw.get("provider_runtime_version")
    runtime = (
        parse_provider_runtime_version(str(runtime_raw))
        if isinstance(runtime_raw, str)
        else None
    )
    identity = QualificationExperimentIdentity(
        provider_kind=str(identity_raw.get("provider_kind", "")),
        provider_runtime_version=runtime,
        provider_runtime_version_policy=VersionMatchPolicy(
            str(identity_raw.get("provider_runtime_version_policy", VersionMatchPolicy.EXACT.value))
        ),
        model_name=str(identity_raw.get("model_name", "")),
        model_digest=(
            str(identity_raw.get("model_digest"))
            if identity_raw.get("model_digest") is not None
            else None
        ),
        model_digest_policy=VersionMatchPolicy(
            str(identity_raw.get("model_digest_policy", VersionMatchPolicy.EXACT.value))
        ),
        quantization=(
            str(identity_raw.get("quantization"))
            if identity_raw.get("quantization") is not None
            else None
        ),
        generation_config_fingerprint=str(identity_raw.get("generation_config_fingerprint", "")),
        scenario_id=str(identity_raw.get("scenario_id", "")),
        input_id=str(identity_raw.get("input_id", "")),
        source_fingerprint=str(identity_raw.get("source_fingerprint", "")),
        config_fingerprint=str(identity_raw.get("config_fingerprint", "")),
    )
    required = raw.get("required_artifacts")
    blob_paths = raw.get("source_blob_paths")
    groups = raw.get("semantic_source_groups")
    checkpoint_indices = raw.get("source_checkpoint_run_indices")
    return QualificationSpec(
        experiment_identity=identity,
        run_count=int(str(raw.get("run_count", 0))),
        required_artifacts=(
            tuple(str(item) for item in required)
            if isinstance(required, list)
            else DEFAULT_REQUIRED_ARTIFACTS
        ),
        source_blob_paths=tuple(str(item) for item in blob_paths) if isinstance(blob_paths, list) else (),
        semantic_source_groups=(
            {str(k): tuple(str(v) for v in values) for k, values in groups.items()}
            if isinstance(groups, dict)
            else {}
        ),
        max_evaluator_attempt_index=int(str(raw.get("max_evaluator_attempt_index", 1))),
        source_checkpoint_run_indices=(
            tuple(int(item) for item in checkpoint_indices)
            if isinstance(checkpoint_indices, list)
            else (0, 10, 19)
        ),
    )
