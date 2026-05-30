# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.applications.contracts.manifest import (
    AgentBinding,
    ApplicationFeatures,
    ApplicationManifest,
    ApplicationProfile,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

pytestmark = pytest.mark.unit


def test_agent_binding_validates_import_path() -> None:
    binding = AgentBinding(import_path="echo.echo_agent.EchoAgent")
    assert binding.import_path == "echo.echo_agent.EchoAgent"


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "EchoAgent",
        "echo.EchoAgent",
        "echo.echo_agent.echo_agent",
        "Echo.echo_agent.EchoAgent",
    ],
)
def test_agent_binding_rejects_bad_import_path(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        AgentBinding(import_path=bad_path)


def test_application_manifest_lab_factory() -> None:
    manifest = ApplicationManifest.lab(
        app_id="my_lab",
        name="My Lab",
        agents=[
            AgentBinding(
                import_path="echo.echo_agent.EchoAgent",
                capabilities=["echo.basic"],
            )
        ],
    )
    assert manifest.app_id == "my_lab"
    assert manifest.profile == ApplicationProfile.LAB
    assert manifest.route_prefix == "/v1/lab"
    assert manifest.env_prefix == "LAB_"
    assert manifest.integration_profile.relational_store == IntegrationSlug.SQLITE
    assert manifest.features.debug_surface is True
    assert len(manifest.enabled_agents()) == 1


def test_application_manifest_coerces_env_prefix() -> None:
    manifest = ApplicationManifest.lab(
        app_id="demo",
        name="Demo",
        env_prefix="demo",
        agents=[AgentBinding(import_path="echo.echo_agent.EchoAgent")],
    )
    assert manifest.env_prefix == "DEMO_"


def test_application_manifest_integration_profile_from_mapping() -> None:
    manifest = ApplicationManifest.lab(
        app_id="cache_lab",
        name="Cache Lab",
        agents=[AgentBinding(import_path="echo.echo_agent.EchoAgent")],
        integration_profile=IntegrationProfile(
            relational_store=IntegrationSlug.SQLITE,
            key_value_cache=IntegrationSlug.REDIS,
        ),
    )
    assert manifest.integration_profile.key_value_cache == IntegrationSlug.REDIS


def test_application_manifest_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ApplicationManifest.lab(
            app_id="x",
            name="X",
            agents=[AgentBinding(import_path="echo.echo_agent.EchoAgent")],
            unknown_field=True,
        )


def test_require_enabled_agents_raises_when_empty() -> None:
    manifest = ApplicationManifest.lab(
        app_id="empty",
        name="Empty",
        agents=[AgentBinding(import_path="echo.echo_agent.EchoAgent", enabled=False)],
    )
    with pytest.raises(ValueError, match="no enabled agents"):
        manifest.require_enabled_agents()


def test_at_most_one_default_agent() -> None:
    with pytest.raises(ValidationError, match="at most one"):
        ApplicationManifest.lab(
            app_id="multi",
            name="Multi",
            agents=[
                AgentBinding(import_path="echo.echo_agent.EchoAgent", default=True),
                AgentBinding(
                    import_path="research.research_agent.ResearchAgent",
                    default=True,
                ),
            ],
        )


def test_product_profile_defaults() -> None:
    manifest = ApplicationManifest.product(
        app_id="legal_app",
        name="Legal",
        route_prefix="/v1/legal",
        env_prefix="LEGAL_",
        agents=[
            AgentBinding(
                import_path="legal.legal_agent.LegalAgent",
                contract_id="legal-default",
                default=True,
            )
        ],
    )
    assert manifest.profile == ApplicationProfile.PRODUCT
    assert manifest.features.debug_surface is False
    assert manifest.default_agent() is not None
    assert manifest.default_agent().contract_id == "legal-default"
