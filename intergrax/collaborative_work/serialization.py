# © Artur Czarnecki. All rights reserved.

"""Canonical JSON serialization for durable Collaborative Work records."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from intergrax.collaborative_work.repository import PublishedWorkArtifactVersion
from intergrax.contracts.collaborative_work import (
    Assignment,
    AuthorityDelegation,
    ArtifactContentRef,
    CollaborativeOperationPolicyProfile,
    CollaborativePolicyRule,
    PrincipalAuthorityGrant,
    WorkArtifact,
    WorkArtifactVersion,
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
    raw = json.loads(payload)
    execution_raw = raw.get("execution")
    execution = None
    if execution_raw is not None:
        execution = ExecutionProvenanceRef(
            task_id=validate_task_id(execution_raw["task_id"]),
            run_id=validate_run_id(execution_raw["run_id"]),
            attempt_id=validate_attempt_id(execution_raw["attempt_id"]),
            execution_id=validate_execution_id(execution_raw["execution_id"]),
        )
    content_raw = raw["content_ref"]
    content_ref = ArtifactContentRef.model_validate(content_raw)
    return WorkArtifactVersion(
        schema_version=raw["schema_version"],
        work_artifact_version_id=raw["work_artifact_version_id"],
        work_artifact_id=raw["work_artifact_id"],
        tenant_id=raw["tenant_id"],
        workspace_id=raw["workspace_id"],
        work_item_id=raw["work_item_id"],
        created_by_principal_id=raw["created_by_principal_id"],
        published_by_principal_id=raw["published_by_principal_id"],
        content_ref=content_ref,
        created_at=datetime.fromisoformat(raw["created_at"]),
        published_at=datetime.fromisoformat(raw["published_at"]),
        execution=execution,
    )


def published_work_artifact_version_to_json(record: PublishedWorkArtifactVersion) -> str:
    payload = {
        "artifact": json.loads(work_artifact_to_json(record.artifact)),
        "version": json.loads(work_artifact_version_to_json(record.version)),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def published_work_artifact_version_from_json(payload: str) -> PublishedWorkArtifactVersion:
    raw = json.loads(payload)
    artifact = work_artifact_from_json(json.dumps(raw["artifact"], sort_keys=True, separators=(",", ":")))
    version = work_artifact_version_from_json(
        json.dumps(raw["version"], sort_keys=True, separators=(",", ":")),
    )
    return PublishedWorkArtifactVersion(artifact=artifact, version=version)
