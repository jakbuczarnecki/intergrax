# © Artur Czarnecki. All rights reserved.

"""Data classification enforcement hooks (IDEAL-23.1)."""

from __future__ import annotations

from intergrax.contracts.data_classification import DataClassification


class DataClassificationPolicyError(PermissionError):
    """Raised when data export violates classification policy."""


def assert_data_export_allowed(
    classification: DataClassification,
    *,
    external_llm: bool = True,
    external_tool: bool = False,
) -> None:
    """Block confidential/restricted payloads from leaving trust boundary."""
    if external_llm and not classification.allows_export():
        raise DataClassificationPolicyError(
            f"DataClassification {classification.value} cannot be sent to external LLM"
        )
    if external_tool and classification is DataClassification.RESTRICTED:
        raise DataClassificationPolicyError(
            "RESTRICTED data cannot be passed to external tools without explicit policy"
        )


def max_classification(
    left: DataClassification,
    right: DataClassification,
) -> DataClassification:
    """Return the more sensitive classification."""
    order = (
        DataClassification.PUBLIC,
        DataClassification.INTERNAL,
        DataClassification.CONFIDENTIAL,
        DataClassification.RESTRICTED,
    )
    return order[max(order.index(left), order.index(right))]
