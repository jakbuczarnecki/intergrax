# © Artur Czarnecki. All rights reserved.

"""EXP-S2: shared contract change - every consumer listed, hidden consumer not missed (O-D; M-07)."""

from __future__ import annotations

import re

from testing_support.change_safety.assess.consumers import (
    consumers_of_targets,
    critical_consumer_closure,
    hidden_consumers_of_targets,
)
from testing_support.change_safety.assess.contract_diff import diff_schema_record
from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult as C
from testing_support.change_safety.discovery.extractors import module_id_for_path
from tests.qualification.evol_x.experiments._harness import (
    ArmResult,
    fixture_inventory,
    fixture_sources,
    load_oracle,
)

SCENARIO = "S2"
_STATIC_IMPORT = re.compile(r"^\s*(?:from|import)\s+intergrax\.contracts(?:\.fx_orders|\s+import\s+fx_orders)", re.M)


def naive_static_import_consumers() -> tuple[str, ...]:
    """Static ``import`` statement scan only (misses dynamic/literal references)."""
    return tuple(
        sorted(
            module_id_for_path(path)
            for path, text in fixture_sources("s2_shared_contract").items()
            if _STATIC_IMPORT.search(text) and "fx_orders.py" not in path
        )
    )


def run() -> tuple[ArmResult, ArmResult]:
    oracle = load_oracle("s2_shared_contract")
    target = frozenset({oracle["contract_module"]})
    with fixture_inventory("s2_shared_contract") as inventory:
        consumers = sorted(consumers_of_targets(inventory, target))
        hidden = hidden_consumers_of_targets(inventory, target)
        closure = critical_consumer_closure(inventory)
    breaking = diff_schema_record(oracle["schema_before"], oracle["schema_after_breaking"]).result
    additive = diff_schema_record(oracle["schema_before"], oracle["schema_after_additive"]).result
    pos_ok = (
        consumers == sorted(oracle["declared_consumers"])
        and hidden == ()
        and closure.orphan_count == 0
        and breaking is C.INCOMPATIBLE
        and additive is C.COMPATIBLE
    )
    naive = list(naive_static_import_consumers())
    missed_by_naive = sorted(set(oracle["hidden_consumers"]) - set(naive))
    neg_ok = (
        set(oracle["hidden_consumers"]) <= set(consumers)  # tool found the hidden consumer
        and missed_by_naive == sorted(oracle["hidden_consumers"])  # baseline would have missed it
    )
    return (
        ArmResult(
            SCENARIO,
            "positive",
            pos_ok,
            f"consumers={consumers} orphans={closure.orphan_count} breaking={breaking.value}",
        ),
        ArmResult(SCENARIO, "negative", neg_ok, f"hidden={oracle['hidden_consumers']} naive_missed={missed_by_naive}"),
    )
