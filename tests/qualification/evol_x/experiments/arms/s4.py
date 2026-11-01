# © Artur Czarnecki. All rights reserved.

"""EXP-S4: provider replacement - swap is composition-only; coupled consumer flags a rewrite (O-F)."""

from __future__ import annotations

from testing_support.change_safety.assess.extension_safety import (
    ExtensionVerdict,
    assess_extension_safety,
    swap_rewrite_set,
)
from tests.qualification.evol_x.experiments._harness import ArmResult, fixture_inventory, load_oracle

SCENARIO = "S4"


def run() -> tuple[ArmResult, ArmResult]:
    oracle = load_oracle("s4_provider_replace")
    composition = frozenset(oracle["composition_modules"])
    with fixture_inventory("s4_provider_replace") as inventory:
        clean = swap_rewrite_set(
            inventory, provider_module=oracle["swap_target_clean"]["provider"], composition_modules=composition
        )
        coupled = swap_rewrite_set(
            inventory, provider_module=oracle["swap_target_coupled"]["provider"], composition_modules=composition
        )
    ext = assess_extension_safety(changed_paths=frozenset(oracle["swap_change_paths"]))
    pos_ok = (
        list(clean) == oracle["swap_target_clean"]["expected_rewrite_set"]
        and ext.verdict is ExtensionVerdict.EXTENSION_ONLY
        and ext.effective_class == "A"
    )
    neg_ok = list(coupled) == oracle["swap_target_coupled"]["expected_rewrite_set"] and len(coupled) > 0
    return (
        ArmResult(SCENARIO, "positive", pos_ok, f"rewrite_set={list(clean)} extension={ext.verdict.value}"),
        ArmResult(SCENARIO, "negative", neg_ok, f"rewrite_required={list(coupled)}"),
    )
