# © Artur Czarnecki. All rights reserved.

"""Shared fixture for the p6_ecapfix catalog-row tests (T-IGX-040-A)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from intergrax.integrations.registry.catalog import (
    catalog_snapshot,
    catalog_state_lock,
    clear_catalog,
    read_catalog_revision_under_lock,
    register_integration,
    set_catalog_generation_for_tests,
)


@pytest.fixture()
def cleared_catalog() -> Iterator[None]:
    """Empty catalog for the test; restore the previous process state afterwards."""
    with catalog_state_lock():
        saved_entries = catalog_snapshot()
        saved_generation = read_catalog_revision_under_lock().generation
    clear_catalog()
    try:
        yield
    finally:
        clear_catalog()
        for entry in saved_entries.values():
            register_integration(entry, override=True)
        set_catalog_generation_for_tests(saved_generation)
