# © Artur Czarnecki. All rights reserved.

"""Canonical typed serialization helper for ``DecisionProposalRef`` (contract-only).

Ownership: platform canonical Decision contracts — encode/decode/validate wire fields only.
Does not define Decision lifecycle, identity rules, or outcome semantics (see
``decision_identity`` / ``decision_record`` validators).
"""

from __future__ import annotations

import json
from typing import TypeAlias

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    validate_decision_id,
    validate_decision_tenant_id,
)
from intergrax.contracts.decision_record import (
    DecisionLineageRef,
    DecisionProposalRef,
    decision_lineage_ref,
    validate_decision_branch_id,
)
from intergrax.contracts.execution_identity import (
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.structured_json_value import StructuredJsonObject

JsonObject: TypeAlias = StructuredJsonObject


def _require_mapping(value: object, label: str) -> JsonObject:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_str(value: object, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be str")
    return value


def _require_int(value: object, label: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise ValueError(f"{label} must be int")
    return value


def _decode_scope_wire(wire: JsonObject) -> DecisionScope:
    return DecisionScope(
        namespace=_require_str(wire.get("namespace"), "scope.namespace"),
        subject=_require_str(wire.get("subject"), "scope.subject"),
    )


def _encode_scope(scope: DecisionScope) -> JsonObject:
    return {
        "namespace": scope.namespace,
        "subject": scope.subject,
    }


def _decode_execution_wire(wire: JsonObject) -> DecisionExecutionLineage:
    execution_id_wire = wire.get("execution_id")
    execution_id = None
    if execution_id_wire is not None:
        execution_id = validate_execution_id(_require_str(execution_id_wire, "execution.execution_id"))
    return DecisionExecutionLineage(
        task_id=validate_task_id(_require_str(wire.get("task_id"), "execution.task_id")),
        run_id=validate_run_id(_require_str(wire.get("run_id"), "execution.run_id")),
        attempt_id=validate_attempt_id(_require_str(wire.get("attempt_id"), "execution.attempt_id")),
        execution_id=execution_id,
    )


def _encode_execution(execution: DecisionExecutionLineage) -> JsonObject:
    wire: JsonObject = {
        "task_id": str(execution.task_id),
        "run_id": str(execution.run_id),
        "attempt_id": str(execution.attempt_id),
    }
    if execution.execution_id is not None:
        wire["execution_id"] = str(execution.execution_id)
    return wire


def _decode_identity_wire(wire: JsonObject) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=validate_decision_id(_require_str(wire.get("decision_id"), "identity.decision_id")),
        version=DecisionVersion(_require_int(wire.get("decision_version"), "identity.decision_version")),
        scope=_decode_scope_wire(_require_mapping(wire.get("scope"), "identity.scope")),
        tenant_id=validate_decision_tenant_id(_require_str(wire.get("tenant_id"), "identity.tenant_id")),
        execution=_decode_execution_wire(_require_mapping(wire.get("execution"), "identity.execution")),
    )


def _encode_identity(identity: DecisionIdentity) -> JsonObject:
    return {
        "decision_id": str(identity.decision_id),
        "decision_version": identity.version.value,
        "tenant_id": identity.tenant_id,
        "scope": _encode_scope(identity.scope),
        "execution": _encode_execution(identity.execution),
    }


def _decode_lineage_ref_wire(wire: JsonObject) -> DecisionLineageRef:
    return decision_lineage_ref(
        DecisionVersion(_require_int(wire.get("version"), "lineage_ref.version")),
        validate_decision_branch_id(_require_str(wire.get("branch_id"), "lineage_ref.branch_id")),
    )


def _encode_lineage_ref(ref: DecisionLineageRef) -> JsonObject:
    return {
        "version": ref.version.value,
        "branch_id": str(ref.branch_id),
    }


def decode_decision_proposal_ref_wire(wire: JsonObject) -> DecisionProposalRef:
    return DecisionProposalRef(
        identity=_decode_identity_wire(_require_mapping(wire.get("identity"), "proposal_ref.identity")),
        lineage_ref=_decode_lineage_ref_wire(
            _require_mapping(wire.get("lineage_ref"), "proposal_ref.lineage_ref"),
        ),
    )


def encode_decision_proposal_ref_wire(proposal_ref: DecisionProposalRef) -> JsonObject:
    return {
        "identity": _encode_identity(proposal_ref.identity),
        "lineage_ref": _encode_lineage_ref(proposal_ref.lineage_ref),
    }


def decision_proposal_ref_to_canonical_json(proposal_ref: DecisionProposalRef) -> str:
    payload = encode_decision_proposal_ref_wire(proposal_ref)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def decision_proposal_ref_from_canonical_json(payload: str) -> DecisionProposalRef:
    raw = json.loads(payload)
    return decode_decision_proposal_ref_wire(_require_mapping(raw, "proposal_ref"))
