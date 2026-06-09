# © Artur Czarnecki. All rights reserved.

"""Policy bundle version pinning on run trace metadata (IDEAL-5.2)."""

from __future__ import annotations

POLICY_BUNDLE_VERSION_KEY = "policy_bundle_version"


def attach_policy_bundle_version(metadata: dict[str, object], version: str) -> None:
    metadata[POLICY_BUNDLE_VERSION_KEY] = version


def read_policy_bundle_version(metadata: dict[str, object]) -> str | None:
    raw = metadata.get(POLICY_BUNDLE_VERSION_KEY)
    return raw if isinstance(raw, str) and raw else None
