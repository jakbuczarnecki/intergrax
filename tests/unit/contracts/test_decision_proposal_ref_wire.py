# © Artur Czarnecki. All rights reserved.

"""MP-4R4 — canonical DecisionProposalRef wire encode/decode qualification."""

from __future__ import annotations

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_proposal_ref_wire import (
    decode_decision_proposal_ref_wire,
    decision_proposal_ref_from_canonical_json,
    decision_proposal_ref_to_canonical_json,
    encode_decision_proposal_ref_wire,
)
from intergrax.contracts.decision_record import (
    DecisionBranchId,
    DecisionProposalRef,
    decision_lineage_ref,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-wire"


def _proposal(
    *,
    version: DecisionVersion | None = None,
    branch_id: DecisionBranchId | None = None,
    execution_id_present: bool = True,
) -> DecisionProposalRef:
    resolved_version = version or initial_decision_version()
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=resolved_version,
        scope=DecisionScope(namespace="incident", subject="subject-1"),
        tenant_id=_TENANT,
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id() if execution_id_present else None,
        ),
    )
    lineage = (
        decision_lineage_ref(identity.version)
        if branch_id is None
        else decision_lineage_ref(identity.version, branch_id=branch_id)
    )
    return DecisionProposalRef(identity=identity, lineage_ref=lineage)


@pytest.mark.parametrize(
    ("factory_kwargs",),
    (
        ({},),
        ({"version": DecisionVersion(2)},),
        ({"branch_id": DecisionBranchId("feature-branch")},),
        ({"execution_id_present": True},),
        ({"execution_id_present": False},),
    ),
)
def test_decision_proposal_ref_wire_roundtrip(factory_kwargs: dict[str, object]) -> None:
    proposal = _proposal(**factory_kwargs)  # type: ignore[arg-type]
    wire = encode_decision_proposal_ref_wire(proposal)
    roundtrip = decode_decision_proposal_ref_wire(wire)
    assert roundtrip == proposal
    json_roundtrip = decision_proposal_ref_from_canonical_json(
        decision_proposal_ref_to_canonical_json(proposal),
    )
    assert json_roundtrip == proposal


@pytest.mark.parametrize(
    ("wire", "match"),
    (
        ({}, "proposal_ref"),
        ({"identity": {}}, "identity.decision_id"),
        (
            {
                "identity": {
                    "decision_id": "not-a-decision-id",
                    "decision_version": 1,
                    "tenant_id": _TENANT,
                    "scope": {"namespace": "n", "subject": "s"},
                    "execution": {
                        "task_id": mint_task_id(),
                        "run_id": mint_run_id(),
                        "attempt_id": mint_attempt_id(),
                    },
                },
                "lineage_ref": {"version": 1, "branch_id": "main"},
            },
                "decision_",
            ),
        (
            {
                "identity": {
                    "decision_id": mint_decision_id(),
                    "decision_version": True,
                    "tenant_id": _TENANT,
                    "scope": {"namespace": "n", "subject": "s"},
                    "execution": {
                        "task_id": mint_task_id(),
                        "run_id": mint_run_id(),
                        "attempt_id": mint_attempt_id(),
                    },
                },
                "lineage_ref": {"version": 1, "branch_id": "main"},
            },
            "identity.decision_version",
        ),
        (
            {
                "identity": {
                    "decision_id": mint_decision_id(),
                    "decision_version": 1,
                    "tenant_id": _TENANT,
                    "scope": {"namespace": "n", "subject": "s"},
                    "execution": {
                        "run_id": mint_run_id(),
                        "attempt_id": mint_attempt_id(),
                    },
                },
                "lineage_ref": {"version": 1, "branch_id": "main"},
            },
            "execution.task_id",
        ),
    ),
)
def test_malformed_wire_fail_closed(wire: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        decode_decision_proposal_ref_wire(wire)  # type: ignore[arg-type]


def test_lineage_identity_version_mismatch_fail_closed() -> None:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=next_decision_version(initial_decision_version()),
        scope=DecisionScope(namespace="n", subject="s"),
        tenant_id=_TENANT,
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=None,
        ),
    )
    aligned = DecisionProposalRef(
        identity=identity,
        lineage_ref=decision_lineage_ref(identity.version),
    )
    wire = encode_decision_proposal_ref_wire(aligned)
    wire["lineage_ref"] = {"version": initial_decision_version().value, "branch_id": "main"}
    with pytest.raises(ValueError, match="identity.version must match lineage_ref.version"):
        decode_decision_proposal_ref_wire(wire)
