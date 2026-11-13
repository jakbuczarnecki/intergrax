# © Artur Czarnecki. All rights reserved.

"""PDC-D4-R014 (APP-HOST-9F): stable public hosting API manifest and COMPAT-X version gate."""

from __future__ import annotations

import intergrax.hosting as hosting
import pytest
from intergrax.hosting.public_api import (
    HOSTING_PUBLIC_API_FINGERPRINTS,
    HOSTING_PUBLIC_API_MANIFEST_SCHEMA_ID,
    HOSTING_PUBLIC_API_VERSION,
    HOSTING_PUBLIC_API_VERSION_HISTORY,
    compute_public_api_fingerprints,
    compute_surface_digest,
    verify_public_api,
)

pytestmark = [pytest.mark.unit]


def test_r014_public_surface_matches_the_declared_manifest() -> None:
    violations = verify_public_api(
        compute_public_api_fingerprints(),
        HOSTING_PUBLIC_API_FINGERPRINTS,
        HOSTING_PUBLIC_API_VERSION_HISTORY,
        HOSTING_PUBLIC_API_VERSION,
    )
    assert violations == [], (
        "The public hosting API changed without a version decision. Review the change, then run "
        "`uv run python -m intergrax.hosting.public_api`, update HOSTING_PUBLIC_API_FINGERPRINTS, "
        "bump HOSTING_PUBLIC_API_VERSION and add a HOSTING_PUBLIC_API_VERSION_HISTORY entry: " + "; ".join(violations)
    )


def test_r014_every_exported_name_resolves_and_is_unique() -> None:
    exported = list(hosting.__all__)
    assert len(exported) == len(set(exported)), "duplicate names in intergrax.hosting.__all__"
    for name in exported:
        assert hasattr(hosting, name), name
    assert set(exported) == set(HOSTING_PUBLIC_API_FINGERPRINTS)
    assert HOSTING_PUBLIC_API_MANIFEST_SCHEMA_ID == "intergrax.hosting.public_api"


def test_r014_gate_fails_on_an_undeclared_public_symbol() -> None:
    live = dict(compute_public_api_fingerprints())
    live["HostedApplicationSecretNewThing"] = "0" * 16
    violations = verify_public_api(
        live,
        HOSTING_PUBLIC_API_FINGERPRINTS,
        HOSTING_PUBLIC_API_VERSION_HISTORY,
        HOSTING_PUBLIC_API_VERSION,
    )
    assert "undeclared public symbol: HostedApplicationSecretNewThing" in violations
    assert any("surface digest differs" in item for item in violations)


def test_r014_gate_fails_on_a_removed_or_renamed_public_symbol() -> None:
    live = dict(compute_public_api_fingerprints())
    removed = sorted(live)[0]
    del live[removed]
    violations = verify_public_api(
        live,
        HOSTING_PUBLIC_API_FINGERPRINTS,
        HOSTING_PUBLIC_API_VERSION_HISTORY,
        HOSTING_PUBLIC_API_VERSION,
    )
    assert f"declared public symbol is no longer exported: {removed}" in violations


def test_r014_gate_fails_on_a_changed_symbol_without_a_version_bump() -> None:
    live = dict(compute_public_api_fingerprints())
    changed = "HostedApplicationEngine"
    live[changed] = "f" * 16
    violations = verify_public_api(
        live,
        HOSTING_PUBLIC_API_FINGERPRINTS,
        HOSTING_PUBLIC_API_VERSION_HISTORY,
        HOSTING_PUBLIC_API_VERSION,
    )
    assert f"public symbol changed without a declared version: {changed}" in violations


def test_r014_gate_accepts_a_change_only_with_a_new_higher_version_entry() -> None:
    live = dict(compute_public_api_fingerprints())
    live["HostedApplicationEngine"] = "f" * 16
    new_declared = dict(HOSTING_PUBLIC_API_FINGERPRINTS)
    new_declared["HostedApplicationEngine"] = "f" * 16
    history = dict(HOSTING_PUBLIC_API_VERSION_HISTORY)
    history["1.1.0"] = compute_surface_digest(live)
    assert verify_public_api(live, new_declared, history, "1.1.0") == []
    # Reusing the old version for the changed surface is rejected.
    assert verify_public_api(live, new_declared, HOSTING_PUBLIC_API_VERSION_HISTORY, "1.0.0") != []
    # A history whose newest entry is not the current version is rejected.
    assert any("not the highest" in item for item in verify_public_api(live, new_declared, history, "1.0.0"))


def test_r014_version_must_be_semantic() -> None:
    violations = verify_public_api(
        compute_public_api_fingerprints(),
        HOSTING_PUBLIC_API_FINGERPRINTS,
        HOSTING_PUBLIC_API_VERSION_HISTORY,
        "v1",
    )
    assert violations == ["public API version is not a semantic version: v1"]
