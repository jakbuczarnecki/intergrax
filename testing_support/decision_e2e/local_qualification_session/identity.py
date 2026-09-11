# © Artur Czarnecki. All rights reserved.

"""Frozen experiment identity validation."""

from __future__ import annotations

from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationExperimentIdentity,
    QualificationIdentityStatus,
    QualificationPreconditionFailure,
    QualificationPreconditionFailureKind,
    QualificationPreconditionResult,
    QualificationRuntimeIdentity,
)
from testing_support.decision_e2e.local_qualification_session.versioning import (
    compare_digest_values,
    compare_runtime_versions,
)


def evaluate_runtime_identity_match(
    frozen: QualificationExperimentIdentity,
    observed: QualificationRuntimeIdentity | None,
) -> QualificationIdentityStatus:
    if observed is None:
        return QualificationIdentityStatus.UNVERIFIABLE
    if observed.provider_kind != frozen.provider_kind:
        return QualificationIdentityStatus.MISMATCH
    return compare_runtime_versions(
        frozen.provider_runtime_version,
        observed.runtime_version,
        policy=frozen.provider_runtime_version_policy,
    )


def evaluate_model_identity_match(
    frozen: QualificationExperimentIdentity,
    observed: QualificationRuntimeIdentity | None,
) -> QualificationIdentityStatus:
    if observed is None:
        return QualificationIdentityStatus.UNVERIFIABLE
    if observed.model_name != frozen.model_name:
        return QualificationIdentityStatus.MISMATCH
    return compare_digest_values(
        frozen.model_digest,
        observed.model_digest,
        policy=frozen.model_digest_policy,
    )


def evaluate_preconditions(
    frozen: QualificationExperimentIdentity,
    observed: QualificationRuntimeIdentity | None,
    *,
    config_fingerprint: str,
    source_fingerprint: str,
    strict_tool_capable: bool,
    structured_output_capable: bool,
) -> QualificationPreconditionResult:
    failures: list[QualificationPreconditionFailure] = []

    if observed is None:
        failures.append(
            QualificationPreconditionFailure(
                kind=QualificationPreconditionFailureKind.PROVIDER_UNAVAILABLE,
                detail="provider runtime identity probe failed",
            )
        )
    else:
        runtime_status = evaluate_runtime_identity_match(frozen, observed)
        if runtime_status is QualificationIdentityStatus.MISMATCH:
            failures.append(
                QualificationPreconditionFailure(
                    kind=QualificationPreconditionFailureKind.PROVIDER_RUNTIME_MISMATCH,
                    detail="BLOCKED_RUNTIME_IDENTITY_MISMATCH",
                )
            )
        elif runtime_status is QualificationIdentityStatus.UNVERIFIABLE:
            failures.append(
                QualificationPreconditionFailure(
                    kind=QualificationPreconditionFailureKind.PROVIDER_UNAVAILABLE,
                    detail="provider runtime version unverifiable",
                )
            )

        model_status = evaluate_model_identity_match(frozen, observed)
        if model_status is QualificationIdentityStatus.MISMATCH:
            failures.append(
                QualificationPreconditionFailure(
                    kind=QualificationPreconditionFailureKind.MODEL_IDENTITY_MISMATCH,
                    detail="frozen model digest or name mismatch",
                )
            )
        elif model_status is QualificationIdentityStatus.UNVERIFIABLE and frozen.model_digest is not None:
            failures.append(
                QualificationPreconditionFailure(
                    kind=QualificationPreconditionFailureKind.MODEL_IDENTITY_MISMATCH,
                    detail="model digest unverifiable",
                )
            )

    if config_fingerprint != frozen.config_fingerprint:
        failures.append(
            QualificationPreconditionFailure(
                kind=QualificationPreconditionFailureKind.CONFIG_MISMATCH,
                detail="qualification config fingerprint drift",
            )
        )

    if source_fingerprint != frozen.source_fingerprint:
        failures.append(
            QualificationPreconditionFailure(
                kind=QualificationPreconditionFailureKind.SOURCE_DRIFT,
                detail="qualification semantic source fingerprint drift",
            )
        )

    if not strict_tool_capable:
        failures.append(
            QualificationPreconditionFailure(
                kind=QualificationPreconditionFailureKind.STRICT_TOOL_PRECONDITION_FAILED,
                detail="strict tool conformance precondition failed",
            )
        )

    if not structured_output_capable:
        failures.append(
            QualificationPreconditionFailure(
                kind=QualificationPreconditionFailureKind.STRUCTURED_OUTPUT_PRECONDITION_FAILED,
                detail="structured output precondition failed",
            )
        )

    return QualificationPreconditionResult(
        eligible=len(failures) == 0,
        failures=tuple(failures),
    )
