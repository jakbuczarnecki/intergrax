# © Artur Czarnecki. All rights reserved.

import itertools

import pytest
from testing_support.change_safety.assess.deployment_safety import (
    DeployFacts,
    SchemaSnapshot,
    SidecarSnapshotPort,
    StoreSchema,
    assess_deploy_safety,
    assess_rollback_eligibility,
    facts_from_snapshots,
    load_deploy_matrix,
    load_na_register,
)
from testing_support.change_safety.contracts.enums import (
    CompatibilityAtChangeResult as C,
)
from testing_support.change_safety.contracts.enums import (
    DeployRefusalReason as R,
)
from testing_support.change_safety.contracts.enums import (
    DeploySafetyVerdict as V,
)

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_matrix_covers_dm_01_to_07() -> None:
    rows = load_deploy_matrix()
    assert {r["dm_id"] for r in rows} == {f"DM-0{i}" for i in range(1, 8)}


def test_na_register_lists_na_01_to_04_out_of_scope() -> None:
    entries = load_na_register()
    assert [e["na_id"] for e in entries] == ["NA-01", "NA-02", "NA-03", "NA-04"]
    assert all(e["disposition"] == "OUT_OF_SCOPE" and e["evidence"] for e in entries)


@pytest.mark.parametrize(
    ("facts", "matrix_id", "verdict", "reason"),
    [
        (DeployFacts(operation="rollback"), "DM-01", V.ELIGIBLE, None),
        (
            DeployFacts(operation="rollback", persisted_schema_compatible=False),
            "DM-02",
            V.REFUSE,
            R.PERSISTED_STATE_INCOMPATIBLE,
        ),
        (
            DeployFacts(operation="rollback", inflight_pins=True, drain_possible=True),
            "DM-03",
            V.ELIGIBLE_WITH_DRAIN,
            None,
        ),
        (
            DeployFacts(operation="rollback", inflight_pins=True, drain_possible=False),
            "DM-03",
            V.REFUSE,
            R.INFLIGHT_PINS,
        ),
        (
            DeployFacts(operation="rollback", durable_forward_only_applied=True),
            "DM-04",
            V.REFUSE,
            R.DURABLE_STATE_FORWARD_ONLY,
        ),
        (
            DeployFacts(operation="roll_forward", contract_compat=C.INCOMPATIBLE),
            "DM-05",
            V.REFUSE,
            R.CONTRACT_INCOMPATIBLE,
        ),
        (
            DeployFacts(operation="rollback", contract_compat=C.MIGRATION_REQUIRED),
            "DM-05",
            V.REFUSE,
            R.CONTRACT_INCOMPATIBLE,
        ),
        (DeployFacts(operation="rollback", evidence_present=False), "DM-06", V.REFUSE, R.INSUFFICIENT_EVIDENCE),
        (DeployFacts(operation="rollback", contract_compat=C.UNKNOWN), "DM-06", V.REFUSE, R.INSUFFICIENT_EVIDENCE),
        (DeployFacts(operation="roll_forward", contract_compat=C.MIGRATION_REQUIRED), "DM-07", V.ELIGIBLE, None),
        (DeployFacts(operation="sideways"), "DM-07", V.REFUSE, R.INSUFFICIENT_EVIDENCE),
    ],
)
def test_matrix_rows(facts: DeployFacts, matrix_id: str, verdict: V, reason: R | None) -> None:
    out = assess_deploy_safety(facts)
    assert (out.matrix_id, out.verdict, out.reason) == (matrix_id, verdict, reason)
    assert out.dimensions


def test_forward_only_rollback_never_eligible_over_input_space() -> None:
    for op, ev, pc, pins, drain, cc in itertools.product(
        ("rollback",), (True, False), (True, False), (True, False), (True, False), tuple(C)
    ):
        out = assess_deploy_safety(
            DeployFacts(
                operation=op,
                evidence_present=ev,
                persisted_schema_compatible=pc,
                durable_forward_only_applied=True,
                inflight_pins=pins,
                drain_possible=drain,
                contract_compat=cc,
            )
        )
        assert out.verdict is V.REFUSE


def test_backward_compatible_rollback_entry_point() -> None:
    assert assess_rollback_eligibility(durable_state_forward_only=True) == (V.REFUSE, R.DURABLE_STATE_FORWARD_ONLY)
    assert assess_rollback_eligibility(durable_state_forward_only=False) == (V.ELIGIBLE, None)


def _snap(rev: str, **stores: StoreSchema) -> SchemaSnapshot:
    return SchemaSnapshot(rev, tuple(sorted(stores.items())))


def test_snapshot_derivation_forward_only_and_incompatible() -> None:
    old = _snap("r1", orders=StoreSchema(version=1, max_readable_version=1))
    new_fwd = _snap("r2", orders=StoreSchema(version=2, max_readable_version=2, forward_only=True))
    new_plain = _snap("r2", orders=StoreSchema(version=2, max_readable_version=2))
    # rollback r2 -> r1: forward-only migration blocks rollback (DM-04)
    facts = facts_from_snapshots(operation="rollback", current=new_fwd, target=old, contract_compat=C.COMPATIBLE)
    assert not facts.persisted_schema_compatible and facts.durable_forward_only_applied
    assert assess_deploy_safety(facts).matrix_id == "DM-04"
    # rollback without forward-only marker: persisted state unreadable by target (DM-02)
    facts = facts_from_snapshots(operation="rollback", current=new_plain, target=old, contract_compat=C.COMPATIBLE)
    assert assess_deploy_safety(facts).matrix_id == "DM-02"
    # roll forward r1 -> r2 is readable
    facts = facts_from_snapshots(operation="roll_forward", current=old, target=new_plain, contract_compat=C.COMPATIBLE)
    assert assess_deploy_safety(facts).verdict is V.ELIGIBLE
    # missing snapshot -> insufficient evidence (never silent ELIGIBLE)
    facts = facts_from_snapshots(operation="rollback", current=None, target=old, contract_compat=C.COMPATIBLE)
    assert assess_deploy_safety(facts).reason is R.INSUFFICIENT_EVIDENCE


def test_sidecar_manifest_roundtrip_and_schema_check() -> None:
    port = SidecarSnapshotPort.from_manifest(
        {
            "schema_version": "schema_snapshot_manifest.v1",
            "revisions": {"r1": {"stores": {"orders": {"version": 3, "forward_only": True}}}},
        }
    )
    snap = port.schema_snapshot("r1")
    assert snap is not None and dict(snap.stores)["orders"] == StoreSchema(3, 3, True)
    assert port.schema_snapshot("missing") is None
    with pytest.raises(ValueError, match="schema_version"):
        SidecarSnapshotPort.from_manifest({"schema_version": "x", "revisions": {}})
