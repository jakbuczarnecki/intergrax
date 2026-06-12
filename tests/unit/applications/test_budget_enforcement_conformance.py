# © Artur Czarnecki. All rights reserved.

"""APP-PROD-7 — STRICT product manifests declare budget governance."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.budget_wiring import check_manifest_budget_enforcement
from intergrax.applications._shared.product_manifest_registry import iter_strict_product_manifests
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_budget import BudgetLimitEnforcement

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.parametrize(
    ("product_id", "manifest"),
    list(iter_strict_product_manifests()),
    ids=[product_id for product_id, _ in iter_strict_product_manifests()],
)
def test_strict_product_manifest_budget_conformance(
    product_id: str,
    manifest,
) -> None:
    violations = check_manifest_budget_enforcement(product_id, manifest)
    assert violations == [], "\n".join(violations)


def test_product_defaults_include_budget_reaction() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    assert env.cost_profile.budget_reaction is not None
    assert env.cost_profile.max_total_tokens is not None
    assert env.cost_profile.budget_enforcement_enabled is True


def test_product_agent_budget_slice_is_hard() -> None:
    from intergrax.applications._shared.budget_wiring import product_agent_budget_slice

    slice_ = product_agent_budget_slice()
    assert slice_.enforcement is BudgetLimitEnforcement.HARD
    assert slice_.max_total_tokens is not None
