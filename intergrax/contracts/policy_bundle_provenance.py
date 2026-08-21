# © Artur Czarnecki. All rights reserved.

"""Shared complete-or-absent validation for canonical policy bundle provenance."""

from __future__ import annotations


def strip_policy_bundle_provenance_identifier(value: str) -> str:
    return value.strip()


def validate_policy_bundle_provenance_complete_or_absent(
    bundle_id: str,
    bundle_version: str,
    bundle_digest: str,
) -> None:
    """Raise when bundle provenance is partial or digest format is invalid."""
    present = (bool(bundle_id), bool(bundle_version), bool(bundle_digest))
    if any(present) and not all(present):
        raise ValueError("policy_bundle_provenance_incomplete")
    if bundle_digest and not bundle_digest.startswith("sha256:"):
        raise ValueError("policy_bundle_digest_must_be_sha256")


def has_attested_policy_bundle_provenance(
    bundle_id: str,
    bundle_version: str,
    bundle_digest: str,
) -> bool:
    """True when canonical bundle id, version, and digest are all non-empty."""
    return bool(bundle_id and bundle_version and bundle_digest)
