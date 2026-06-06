# © Artur Czarnecki. All rights reserved.

"""Tests for harness.reliability_smoke P6 tool expansion."""

from __future__ import annotations

import pytest

from intergrax.skills.providers.harness.manifests import HARNESS_RELIABILITY_SMOKE
from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_default_skills_for_tests()
    register_default_skills()
    yield
    reset_default_skills_for_tests()


def test_harness_reliability_smoke_includes_p6_tools() -> None:
    assert "security.scan" in HARNESS_RELIABILITY_SMOKE.tool_ids
    assert "workflow.trigger" in HARNESS_RELIABILITY_SMOKE.tool_ids
    assert "observability.query_traces" in HARNESS_RELIABILITY_SMOKE.tool_ids
