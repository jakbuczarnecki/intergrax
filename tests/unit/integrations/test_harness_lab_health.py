# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations._shared.circuit_breaker_registry import reset_circuit_breaker_registry_for_tests
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.harness_lab_health import health_check_harness_lab_stack
from intergrax.integrations.registry.harness_lab_stack import HARNESS_LAB_STABLE_SLUGS

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset_breakers() -> None:
    reset_circuit_breaker_registry_for_tests()
    yield
    reset_circuit_breaker_registry_for_tests()


def test_health_check_harness_lab_stack_covers_stable_slugs() -> None:
    register_default_integrations()
    results = health_check_harness_lab_stack()
    slugs = {item.slug for item in results}
    assert slugs == set(HARNESS_LAB_STABLE_SLUGS)
    assert len(results) == len(HARNESS_LAB_STABLE_SLUGS)
