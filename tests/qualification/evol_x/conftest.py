# © Artur Czarnecki. All rights reserved.

"""Session fixtures for EVOL-X qualification (one inventory build per pytest session)."""

from __future__ import annotations

import pytest
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory, build_closed_world_inventory
from testing_support.change_safety.intake.git_change_set import rev_parse_head


@pytest.fixture(scope="session")
def evol_head() -> str:
    return rev_parse_head()


@pytest.fixture(scope="session")
def evol_inventory(evol_head: str) -> ClosedWorldInventory:
    return build_closed_world_inventory(source_commit_sha=evol_head)


@pytest.fixture(scope="session")
def evol_owner_rules() -> dict:
    return load_evol_x_data_document("owner_rules.v1.json")


@pytest.fixture(scope="session")
def evol_exclusions() -> dict:
    return load_evol_x_data_document("exclusion_register.v1.json")
