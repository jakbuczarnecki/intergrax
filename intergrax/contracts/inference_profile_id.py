# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Logical inference profile identity for neutral execution requests."""

from __future__ import annotations

from typing import NewType

InferenceProfileId = NewType("InferenceProfileId", str)


def validate_inference_profile_id(value: object) -> InferenceProfileId:
    """Validate a logical inference profile identity for execution requests."""
    if type(value) is not str:
        raise TypeError(
            f"InferenceProfileId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "InferenceProfileId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "InferenceProfileId must not contain leading or trailing whitespace",
        )
    return InferenceProfileId(value)
