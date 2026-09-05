# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Caller-supplied availability evidence for discovery (Stage 3)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey

SCHEMA_CAPABILITY_DISCOVERY_AVAILABILITY_EVIDENCE_V1: Final = (
    "capability_discovery_availability_evidence.v1"
)


class CapabilityDiscoveryAvailabilityEvidence(BaseModel):
    """Read-only availability facts — Stage 3 does not compute governance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_discovery_availability_evidence.v1"] = (
        SCHEMA_CAPABILITY_DISCOVERY_AVAILABILITY_EVIDENCE_V1
    )
    host_available_keys: tuple[CapabilityIdentityKey, ...] = ()
    blocked_keys: tuple[CapabilityIdentityKey, ...] = ()
    unavailable_keys: tuple[CapabilityIdentityKey, ...] = ()
    scope_visible_keys: tuple[CapabilityIdentityKey, ...] | None = None

    @model_validator(mode="after")
    def _validate_evidence_consistency(
        self,
    ) -> CapabilityDiscoveryAvailabilityEvidence:
        for label, keys in (
            ("host_available_keys", self.host_available_keys),
            ("blocked_keys", self.blocked_keys),
            ("unavailable_keys", self.unavailable_keys),
            ("scope_visible_keys", self.scope_visible_keys or ()),
        ):
            seen: set[tuple[str, str, str, str]] = set()
            for key in keys:
                sort_key = key.sort_key
                if sort_key in seen:
                    raise ValueError(
                        f"{label} must not repeat the same identity key",
                    )
                seen.add(sort_key)

        host_keys = {key.sort_key for key in self.host_available_keys}
        blocked_keys = {key.sort_key for key in self.blocked_keys}
        unavailable_keys = {key.sort_key for key in self.unavailable_keys}
        for left_label, right_label, left, right in (
            ("host_available_keys", "blocked_keys", host_keys, blocked_keys),
            ("host_available_keys", "unavailable_keys", host_keys, unavailable_keys),
            ("blocked_keys", "unavailable_keys", blocked_keys, unavailable_keys),
        ):
            if left & right:
                raise ValueError(
                    f"{left_label} and {right_label} must not contain "
                    "the same identity key",
                )
        return self
