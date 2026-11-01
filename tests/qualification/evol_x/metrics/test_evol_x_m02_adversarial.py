# © Artur Czarnecki. All rights reserved.

"""B-M02-01: adversarial text vs graph — hidden consumers are not graph orphans when text oracle runs."""

import pytest
from testing_support.change_safety.assess.consumers import (
    consumers_of_targets,
    critical_consumer_closure,
)
from tests.qualification.evol_x.experiments._harness import fixture_inventory, load_oracle
from tests.qualification.evol_x.experiments.arms.s2 import naive_static_import_consumers
from tests.qualification.evol_x.metrics.measure import measure_m02

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_m02_graph_text_gap_hidden_consumer_not_orphan() -> None:
    oracle = load_oracle("s2_shared_contract")
    target = frozenset({oracle["contract_module"]})
    with fixture_inventory("s2_shared_contract") as inventory:
        consumers = consumers_of_targets(inventory, target)
        assert set(oracle["hidden_consumers"]) <= set(consumers)
        closure = critical_consumer_closure(inventory)
        assert closure.orphan_count == 0
        m02, m02c = measure_m02(inventory)
        assert m02.passed and m02c.passed
        naive = naive_static_import_consumers()
        missed = sorted(set(oracle["hidden_consumers"]) - set(naive))
        assert missed == sorted(oracle["hidden_consumers"])
