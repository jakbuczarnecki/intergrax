# © Artur Czarnecki. All rights reserved.

"""ApplicationManifest round-trip conformance (Phase H-APP.0.5)."""

from __future__ import annotations

import pytest

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import ApplicationManifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_lab_manifest_round_trip() -> None:
    from lab_application.manifest import build_lab_manifest_default

    manifest = build_lab_manifest_default()
    ctx = ApplicationBuildContext.for_manifest(manifest)
    assert ctx.manifest.app_id == "lab"
    assert manifest.resolved_environment().profile_id


def test_legal_manifest_round_trip() -> None:
    from legal_application.manifest import LEGAL_APPLICATION_MANIFEST

    ctx = ApplicationBuildContext.for_manifest(LEGAL_APPLICATION_MANIFEST)
    assert ctx.integration_profile is not None
    assert LEGAL_APPLICATION_MANIFEST.profile.value == "product"


def test_poc_template_manifest_round_trip() -> None:
    from poc_template_application.manifest import build_poc_template_manifest

    manifest = build_poc_template_manifest()
    ctx = ApplicationBuildContext.for_manifest(manifest)
    assert manifest.app_id == "poc_template"
    assert ctx.manifest.route_prefix.startswith("/v1/")
