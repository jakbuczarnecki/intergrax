# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.assess.extension_safety import assess_frozen_core_touch
from testing_support.change_safety.assess.triggers import assess_evidence_freshness
from testing_support.change_safety.diagnostics.dossier import build_change_dossier
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.intake.git_change_set import rev_parse_head
from testing_support.change_safety.selection.policy import selection_policy_forbid_empty

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_extension_and_freshness_gates() -> None:
    assert selection_policy_forbid_empty()
    assert assess_frozen_core_touch(changed_paths=frozenset({"intergrax/contracts/migrations/registry.py"}))
    head = rev_parse_head()
    inventory = build_closed_world_inventory(source_commit_sha=head)
    fresh = assess_evidence_freshness(
        expected_graph_digest=inventory.inventory_digest,
        actual_graph_digest=inventory.inventory_digest,
        expected_inventory_digest=inventory.inventory_digest,
        actual_inventory_digest=inventory.inventory_digest,
    )
    assert fresh.allowed
    dossier = build_change_dossier(
        graph_digest=inventory.inventory_digest,
        inventory_digest=inventory.inventory_digest,
    )
    assert dossier.schema_version == "evol_change_dossier.v1"
