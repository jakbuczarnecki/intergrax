# © Artur Czarnecki. All rights reserved.

"""DS-E2E-10 — two-tenant isolation."""

from __future__ import annotations

import pytest

from intergrax.contracts.decision_finalization import decision_finalization_key
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    decision_lineage_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
)
from intergrax.runtime.execution.decision_finalization_conformance import (
    IncidentDecisionPayload,
)

from testing_support.decision_e2e.composition import (
    build_sqlite_persistence,
    mint_qualification_identity,
)
from testing_support.decision_e2e.contracts import (
    DecisionE2EProofId,
    DecisionE2EQualificationResult,
    QualificationDisposition,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.qualification,
    pytest.mark.no_ci,
]


def test_ds_e2e_10_two_tenant_isolation(
    tmp_path,
    decision_e2e_report_collector,
) -> None:
    tenant_a = mint_qualification_identity(
        tenant_id="tenant-a",
        namespace="shared",
        subject="shared-subject",
    )
    tenant_b = mint_qualification_identity(
        tenant_id="tenant-b",
        namespace="shared",
        subject="shared-subject",
    )
    store_a = build_sqlite_persistence(tmp_path / "tenant-a").finalization
    store_b = build_sqlite_persistence(tmp_path / "tenant-b").finalization
    accepted_a = AuthoritativeAcceptedDecision(
        identity=tenant_a,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("incident_resolution"),
            content=IncidentDecisionPayload(recommendation="tenant-a"),
        ),
        lineage=decision_version_lineage(current=decision_lineage_ref(tenant_a.version)),
    )
    accepted_b = AuthoritativeAcceptedDecision(
        identity=tenant_b,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("incident_resolution"),
            content=IncidentDecisionPayload(recommendation="tenant-b"),
        ),
        lineage=decision_version_lineage(current=decision_lineage_ref(tenant_b.version)),
    )
    store_a.commit_authoritative_outcome(
        key=decision_finalization_key(tenant_a),
        requested_outcome=accepted_a,
    )
    store_b.commit_authoritative_outcome(
        key=decision_finalization_key(tenant_b),
        requested_outcome=accepted_b,
    )
    loaded_a = store_a.load_guard_state(key=decision_finalization_key(tenant_a))
    loaded_b = store_b.load_guard_state(key=decision_finalization_key(tenant_b))
    assert loaded_a is not None and loaded_a.authoritative_outcome is not None
    assert loaded_b is not None and loaded_b.authoritative_outcome is not None
    assert loaded_a.authoritative_outcome.identity.tenant_id == "tenant-a"
    assert loaded_b.authoritative_outcome.identity.tenant_id == "tenant-b"
    cross = store_a.load_guard_state(key=decision_finalization_key(tenant_b))
    assert cross is None or cross.authoritative_outcome is None

    decision_e2e_report_collector.record(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_10,
            disposition=QualificationDisposition.PASSED,
            evidence=(),
        ),
    )
