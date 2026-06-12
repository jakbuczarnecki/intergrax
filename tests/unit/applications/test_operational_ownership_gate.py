# © Artur Czarnecki. All rights reserved.

"""APP-OPS-2 — ApplicationOperationalOwnership on product manifests."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.ownership_wiring import (
    check_manifest_operational_ownership,
    evaluate_application_ownership,
    standard_product_operational_ownership,
)
from intergrax.applications._shared.product_manifest_registry import iter_product_manifests
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.parametrize(
    ("product_id", "manifest"),
    list(iter_product_manifests()),
    ids=[product_id for product_id, _ in iter_product_manifests()],
)
def test_product_manifest_declares_operational_ownership(
    product_id: str,
    manifest: ApplicationManifest,
) -> None:
    violations = check_manifest_operational_ownership(product_id, manifest)
    assert violations == [], "\n".join(violations)
    decision = evaluate_application_ownership(manifest)
    assert decision.approved


def test_product_manifest_without_ownership_fails_gate() -> None:
    manifest = ApplicationManifest.product(
        app_id="ownership_missing",
        name="Ownership Missing",
        route_prefix="/v1/ownership_missing",
        env_prefix="OWNERSHIP_MISSING_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
    )
    violations = check_manifest_operational_ownership("ownership_missing", manifest)
    assert any("ownership must be declared" in item for item in violations)


def test_standard_product_operational_ownership_matches_app_id() -> None:
    ownership = standard_product_operational_ownership("legal")
    assert ownership.app_id == "legal"
    assert ownership.architecture_ref.endswith("legal_application/ARCHITECTURE.md")
    assert ownership.maintainer.repo_path == "applications/legal_application/"
