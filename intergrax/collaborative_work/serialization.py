# © Artur Czarnecki. All rights reserved.

"""Canonical JSON serialization for durable Collaborative Work records."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from intergrax.collaborative_work.repository import PublishedWorkArtifactVersion
from intergrax.contracts.collaborative_decision_binding import CollaborativeDecisionBinding
from intergrax.contracts.collaborative_work import (
    Assignment,
    AuthorityDelegation,
    CollaborativeOperationPolicyProfile,
    CollaborativePolicyRule,
    PrincipalAuthorityGrant,
    WorkArtifact,
    WorkArtifactVersion,
    WorkArtifactVersionRef,
    WorkItem,
    WorkItemExecutionLink,
    WorkspaceMembership,
)
from intergrax.contracts.execution_identity import (
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.decision_proposal_ref_wire import (
    decision_proposal_ref_from_canonical_json,
    decision_proposal_ref_to_canonical_json,
)
from intergrax.contracts.decision_record import DecisionProposalRef
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef


def _encode_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _decode_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def workspace_membership_to_json(record: WorkspaceMembership) -> str:
    return record.model_dump_json()


def workspace_membership_from_json(payload: str) -> WorkspaceMembership:
    return WorkspaceMembership.model_validate_json(payload)


def authority_delegation_to_json(record: AuthorityDelegation) -> str:
    return record.model_dump_json()


def authority_delegation_from_json(payload: str) -> AuthorityDelegation:
    return AuthorityDelegation.model_validate_json(payload)


def principal_authority_grant_to_json(record: PrincipalAuthorityGrant) -> str:
    return record.model_dump_json()


def principal_authority_grant_from_json(payload: str) -> PrincipalAuthorityGrant:
    return PrincipalAuthorityGrant.model_validate_json(payload)


def collaborative_policy_rule_to_json(record: CollaborativePolicyRule) -> str:
    return record.model_dump_json()


def collaborative_policy_rule_from_json(payload: str) -> CollaborativePolicyRule:
    return CollaborativePolicyRule.model_validate_json(payload)


def operation_policy_profile_to_json(record: CollaborativeOperationPolicyProfile) -> str:
    return record.model_dump_json()


def operation_policy_profile_from_json(payload: str) -> CollaborativeOperationPolicyProfile:
    return CollaborativeOperationPolicyProfile.model_validate_json(payload)


def work_item_to_json(record: WorkItem) -> str:
    return record.model_dump_json()


def work_item_from_json(payload: str) -> WorkItem:
    return WorkItem.model_validate_json(payload)


def assignment_to_json(record: Assignment) -> str:
    return record.model_dump_json()


def assignment_from_json(payload: str) -> Assignment:
    return Assignment.model_validate_json(payload)


def work_item_execution_link_to_json(record: WorkItemExecutionLink) -> str:
    payload = record.model_dump(mode="json")
    payload["execution"] = {
        "task_id": str(record.execution.task_id),
        "run_id": str(record.execution.run_id),
        "attempt_id": str(record.execution.attempt_id),
        "execution_id": str(record.execution.execution_id),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def work_item_execution_link_from_json(payload: str) -> WorkItemExecutionLink:
    raw = json.loads(payload)
    execution_raw = raw["execution"]
    execution = ExecutionProvenanceRef(
        task_id=validate_task_id(execution_raw["task_id"]),
        run_id=validate_run_id(execution_raw["run_id"]),
        attempt_id=validate_attempt_id(execution_raw["attempt_id"]),
        execution_id=validate_execution_id(execution_raw["execution_id"]),
    )
    return WorkItemExecutionLink(
        schema_version=raw["schema_version"],
        execution_link_id=raw["execution_link_id"],
        tenant_id=raw["tenant_id"],
        workspace_id=raw["workspace_id"],
        work_item_id=raw["work_item_id"],
        execution=execution,
        linked_at=datetime.fromisoformat(raw["linked_at"]),
    )


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def work_artifact_to_json(record: WorkArtifact) -> str:
    return record.model_dump_json()


def work_artifact_from_json(payload: str) -> WorkArtifact:
    return WorkArtifact.model_validate_json(payload)


def work_artifact_version_to_json(record: WorkArtifactVersion) -> str:
    payload = record.model_dump(mode="json")
    if record.execution is None:
        payload["execution"] = None
    else:
        payload["execution"] = {
            "task_id": str(record.execution.task_id),
            "run_id": str(record.execution.run_id),
            "attempt_id": str(record.execution.attempt_id),
            "execution_id": str(record.execution.execution_id),
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def work_artifact_version_from_json(payload: str) -> WorkArtifactVersion:
    return WorkArtifactVersion.model_validate_json(payload)


def published_work_artifact_version_to_json(record: PublishedWorkArtifactVersion) -> str:
    payload = {
        "artifact": json.loads(work_artifact_to_json(record.artifact)),
        "version": json.loads(work_artifact_version_to_json(record.version)),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def published_work_artifact_version_from_json(payload: str) -> PublishedWorkArtifactVersion:
    return PublishedWorkArtifactVersion.model_validate_json(payload)


def collaborative_decision_binding_to_json(record: CollaborativeDecisionBinding) -> str:
    payload = record.model_dump(mode="json", exclude={"decision_proposal", "work_artifact_version"})
    payload["decision_proposal"] = json.loads(
        decision_proposal_ref_to_canonical_json(record.decision_proposal),
    )
    if record.work_artifact_version is None:
        payload["work_artifact_version"] = None
    else:
        payload["work_artifact_version"] = record.work_artifact_version.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def collaborative_decision_binding_from_json(payload: str) -> CollaborativeDecisionBinding:
    raw = json.loads(payload)
    proposal_wire = raw["decision_proposal"]
    artifact_raw = raw.get("work_artifact_version")
    work_artifact_version = None
    if artifact_raw is not None:
        work_artifact_version = WorkArtifactVersionRef.model_validate(artifact_raw)
    return CollaborativeDecisionBinding(
        schema_version=raw["schema_version"],
        binding_id=raw["binding_id"],
        tenant_id=raw["tenant_id"],
        workspace_id=raw["workspace_id"],
        work_item_id=raw["work_item_id"],
        work_artifact_version=work_artifact_version,
        decision_proposal=decision_proposal_ref_from_canonical_json(
            json.dumps(proposal_wire, sort_keys=True, separators=(",", ":")),
        ),
        created_by_principal_id=raw["created_by_principal_id"],
        created_at=datetime.fromisoformat(raw["created_at"]),
    )
