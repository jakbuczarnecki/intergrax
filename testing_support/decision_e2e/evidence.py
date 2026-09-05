# © Artur Czarnecki. All rights reserved.

"""Qualification evidence helpers."""

from __future__ import annotations

from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage

from testing_support.decision_e2e.contracts import QualificationEvidenceRef
from testing_support.decision_e2e.bindings import ProviderBindingEvidence
from testing_support.decision_e2e.qualification_evidence import (
    DockerCrashEvidence,
    ScenarioExecutionEvidence,
)


def provider_evidence_ref(binding: ProviderBindingEvidence) -> QualificationEvidenceRef:
    model = binding.model or "default"
    detail = f"provider={binding.provider};model={model}"
    if binding.host:
        detail = f"{detail};host={binding.host}"
    return QualificationEvidenceRef(
        kind="provider_binding",
        ref=binding.profile_id,
        detail=detail,
    )


def decision_identity_evidence(identity: DecisionIdentity) -> QualificationEvidenceRef:
    return QualificationEvidenceRef(
        kind="decision_identity",
        ref=str(identity.decision_id),
        detail=(
            f"version={identity.version.value};tenant={identity.tenant_id};"
            f"scope={identity.scope.namespace}/{identity.scope.subject}"
        ),
    )


def lifecycle_stage_evidence(stage: DecisionLifecycleStage) -> QualificationEvidenceRef:
    return QualificationEvidenceRef(
        kind="lifecycle_stage",
        ref=stage.value,
    )


def invocation_count_evidence(count: int) -> QualificationEvidenceRef:
    return QualificationEvidenceRef(
        kind="provider_invocation_count",
        ref=str(count),
    )


def runtime_event_count_evidence(count: int) -> QualificationEvidenceRef:
    return QualificationEvidenceRef(
        kind="runtime_event_count",
        ref=str(count),
    )


def docker_crash_evidence_ref(evidence: DockerCrashEvidence) -> QualificationEvidenceRef:
    detail = (
        f"kill_method={evidence.kill_method};"
        f"killed_container_id={evidence.killed_container_id};"
        f"killed_exit_code={evidence.killed_exit_code};"
        f"resume_container_id={evidence.resume_container_id};"
        f"durable_store_path={evidence.durable_store_path};"
        f"window={evidence.window};"
        f"final_disposition={evidence.final_disposition}"
    )
    return QualificationEvidenceRef(
        kind="docker_crash",
        ref=evidence.window,
        detail=detail,
    )


def scenario_execution_evidence_ref(
    evidence: ScenarioExecutionEvidence,
) -> QualificationEvidenceRef:
    modules = ",".join(sorted(evidence.runtime_modules)) if evidence.runtime_modules else "none"
    detail = (
        f"invocation={evidence.invocation};"
        f"provider={evidence.provider};"
        f"model={evidence.model or 'default'};"
        f"executed={evidence.executed};"
        f"decision_path={evidence.decision_path_exercised};"
        f"mock_provider={evidence.used_mock_provider};"
        f"outcome={evidence.outcome or 'unknown'};"
        f"runtime_modules={modules}"
    )
    return QualificationEvidenceRef(
        kind="scenario_execution",
        ref=evidence.scenario_id,
        detail=detail,
    )
