# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import IntegrationStatus
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.harness_lab_stack import (
    HARNESS_LAB_STABLE_SLUGS,
    harness_lab_stack_metadata,
    list_harness_lab_stable_slugs,
    validate_harness_lab_stable_stack,
)


pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_harness_lab_stable_slug_list_is_documented_set() -> None:
    slugs = list_harness_lab_stable_slugs()
    assert slugs == tuple(sorted(HARNESS_LAB_STABLE_SLUGS))
    assert "sqlite" in slugs
    assert "otel" in slugs


def test_harness_lab_stack_entries_are_stable_after_bootstrap() -> None:
    register_default_integrations()
    validate_harness_lab_stable_stack()
    for meta in harness_lab_stack_metadata():
        assert meta.status is IntegrationStatus.STABLE
