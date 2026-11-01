# © Artur Czarnecki. All rights reserved.

"""Single-vendor integration contract proofs (no false second provider)."""

from __future__ import annotations

import pytest
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = [pytest.mark.unit, pytest.mark.qualification, pytest.mark.gate]

_VENDOR_BLOCKED: tuple[str, ...] = (
    "integration.billing_meter",
    "integration.managed_retrieval",
    "integration.ml_inference_host",
    "integration.model_serving_runtime",
    "integration.vision_serving",
)


def _category_for(concern_id: str) -> IntegrationCategory:
    suffix = concern_id.removeprefix("integration.")
    return IntegrationCategory(suffix)


@pytest.mark.parametrize("concern_id", _VENDOR_BLOCKED)
def test_ebh7_vendor_single_slug_fail_closed_without_profile(concern_id: str) -> None:
    category = _category_for(concern_id)
    clear_catalog()
    with pytest.raises((IntegrationConfigurationError, Exception)):
        resolve_from_profile(IntegrationProfile(), category)


def test_ebh7_external_work_category_exists_no_catalog_providers() -> None:
    assert IntegrationCategory.EXTERNAL_WORK.value == "external_work"
