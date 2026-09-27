# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pre-built integration category contract validation (contract-adjacent helper)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory


def validated_prebuilt_instance_for_category(
    category: IntegrationCategory,
    instance: object,
) -> object:
    from intergrax.runtime.integrations.contract_metadata import contract_for_category

    expected_contract = contract_for_category(category.value)
    if not isinstance(instance, expected_contract):
        raise TypeError(
            f"Pre-built integration for category {category.value!r} is "
            f"{type(instance).__name__}, expected {expected_contract.__name__}."
        )
    return instance


__all__ = ["validated_prebuilt_instance_for_category"]
