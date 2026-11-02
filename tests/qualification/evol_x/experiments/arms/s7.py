# © Artur Czarnecki. All rights reserved.

"""EXP-S7: rollback safety (DM oracle; M-08). Assessor-only claim (OD-EVOL-05b Option 1)."""

from __future__ import annotations

from testing_support.change_safety.assess.deployment_safety import (
    DeployFacts,
    SidecarSnapshotPort,
    assess_deploy_safety,
    facts_from_snapshots,
)
from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult as C
from testing_support.change_safety.contracts.enums import DeploySafetyVerdict
from tests.qualification.evol_x.adapters.ad_harness import observe_ad_rollback
from tests.qualification.evol_x.experiments._harness import ArmResult, load_oracle

SCENARIO = "S7"


def _facts(raw: dict) -> DeployFacts:
    data = dict(raw)
    if "contract_compat" in data:
        data["contract_compat"] = C(data["contract_compat"])
    return DeployFacts(**data)


def run() -> tuple[ArmResult, ArmResult]:
    oracle = load_oracle("s7_rollback")
    outcomes = 0
    rows = oracle["rows"]
    for row in rows:
        verdict = assess_deploy_safety(_facts(row["facts"]))
        refusal = verdict.reason.value if verdict.reason else None
        if (
            verdict.matrix_id == row["expected_matrix_id"]
            and verdict.verdict.value == row["expected_verdict"]
            and refusal == row["expected_refusal"]
        ):
            outcomes += 1
    ad = observe_ad_rollback()
    clean = assess_deploy_safety(DeployFacts(operation="rollback"))
    pos_ok = outcomes == len(rows) and clean.verdict is DeploySafetyVerdict.ELIGIBLE and ad.succeeded
    # Negative arm: real sidecar snapshots with a forward-only migration => REFUSE (DM-04) although the AD layer
    # alone would let the rollback through (the documented FRZ-EVOL-07 boundary: assessor, not production port).
    port = SidecarSnapshotPort.from_manifest(oracle["snapshot_manifest"])
    facts = facts_from_snapshots(
        operation="rollback",
        current=port.schema_snapshot("rev-2"),
        target=port.schema_snapshot("rev-1"),
        contract_compat=C.COMPATIBLE,
    )
    dm04 = assess_deploy_safety(facts)
    false_safe = int(dm04.verdict is not DeploySafetyVerdict.REFUSE)
    neg_ok = dm04.matrix_id == "DM-04" and false_safe == 0
    return (
        ArmResult(SCENARIO, "positive", pos_ok, f"dm_outcomes={outcomes}/{len(rows)} ad_rollback_ok={ad.succeeded}"),
        ArmResult(
            SCENARIO,
            "negative",
            neg_ok,
            f"dm04={dm04.verdict.value} false_safe={false_safe} ad_layer_alone_succeeds={ad.succeeded}",
        ),
    )
