# © Artur Czarnecki. All rights reserved.

"""Research production factory revision-bound registry projection E2E (AC-3)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from research_application.host.factory import create_research_backend_app
from research_application.host.settings import ResearchBackendSettings
from research_application.host.wiring import build_research_environment_profile
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST
from research_application.tests.research_ac3_projection import build_research_test_registry_projection

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_research_factory_uses_projected_registry_not_manifest_extra_agents() -> None:
    settings = ResearchBackendSettings(use_nexus_loop=True)
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    projection = build_research_test_registry_projection(
        settings,
        revision_id="rev-research-e2e",
        enabled_contract_ids=("research",),
    )
    app = create_research_backend_app(
        registry_projection=projection,
        settings=settings,
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code in {200, 204}

    manifest_stems = {_binding_stem(binding) for binding in manifest.enabled_agents()}
    projected_ids = set(projection.agent_registry.list_agent_ids())
    assert projected_ids == {"research"}
    assert "summary" in manifest_stems
    assert "summary" not in projected_ids

    runtime = build_harness_host_runtime(
        manifest.model_copy(update={"environment": env}),
        env,
        settings=settings,
        registry_projection=projection,
        use_in_memory_trace=True,
    )
    assert runtime.registry_projection_evidence is not None
    assert runtime.registry_projection_evidence.runtime_revision_id == "rev-research-e2e"
    assert runtime.nexus_loop.registry.list_agent_ids() == ["research"]
    assert "summary" not in runtime.nexus_loop.registry.list_agent_ids()


def _binding_stem(binding) -> str:
    if binding.contract_id is not None:
        return binding.contract_id
    import_path = binding.import_path or ""
    stem = import_path.rsplit(".", 1)[-1]
    if stem.endswith("Agent") and len(stem) > 5:
        return stem[:-5].lower()
    return stem.lower()
