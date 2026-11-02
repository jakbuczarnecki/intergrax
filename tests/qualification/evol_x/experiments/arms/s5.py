# © Artur Czarnecki. All rights reserved.

"""EXP-S5: persisted state migration - typed MIGRATE/INCOMPATIBLE, never a silent break (COMPAT oracle; M-07)."""

from __future__ import annotations

from testing_support.change_safety.assess.contract_diff import assess_persisted_schema_change
from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult as C
from tests.qualification.evol_x.experiments._harness import ArmResult, load_oracle

SCENARIO = "S5"


def run() -> tuple[ArmResult, ArmResult]:
    oracle = load_oracle("s5_state_migration")
    before, breaking, additive = (
        oracle["schema_before"],
        oracle["schema_after_breaking"],
        oracle["schema_after_additive"],
    )
    with_migration = assess_persisted_schema_change(before, breaking, migration_registered=True)
    additive_res = assess_persisted_schema_change(before, additive, migration_registered=False)
    silent = assess_persisted_schema_change(before, breaking, migration_registered=False)
    unknown = assess_persisted_schema_change(before, None, migration_registered=True)
    pos_ok = (
        with_migration.result is C.MIGRATION_REQUIRED
        and additive_res.result is C.COMPATIBLE
        and unknown.result in {C.UNKNOWN, C.INCOMPATIBLE, C.MIGRATION_REQUIRED}
    )
    neg_ok = silent.result is C.INCOMPATIBLE and any("silent break" in r for r in silent.reasons)
    return (
        ArmResult(
            SCENARIO, "positive", pos_ok, f"migrate={with_migration.result.value} additive={additive_res.result.value}"
        ),
        ArmResult(SCENARIO, "negative", neg_ok, f"no_migration={silent.result.value} reasons={list(silent.reasons)}"),
    )
