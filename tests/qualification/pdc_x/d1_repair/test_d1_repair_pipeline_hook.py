# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-A / F-D1-02 second activation path: the deploy/activate pipeline trust hook.

``run_deploy_and_activate`` takes a caller-supplied projection input and commits the activation
after control-plane authorization only. The optional ``revision_trust_admission`` hook of
``DeployActivatePipelineContext`` lets a composition refuse a revision whose packages were
revoked. These tests drive the real pipeline over the enterprise proof stack (SQLite, no network).
The production composition does not pass the hook yet (excluded wiring, see the ticket).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.agent_distribution.errors import AgentPackageTrustError
from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from intergrax.applications._shared.deploy_activate_pipeline import (
    DeployActivatePipelineContext,
    run_deploy_and_activate,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from testing_support.canonical_agent_lifecycle_composition import default_stage15_proof_config
from testing_support.enterprise_agent_lifecycle_composition import EnterpriseAgentLifecycleProofStack

pytestmark = [pytest.mark.integration, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr("intergrax.applications._shared.llm_resolver.resolve_llm_adapter", _resolve)


class _Scenario:
    def __init__(self, tmp_path: Path) -> None:
        self.config = default_stage15_proof_config()
        self.stack = EnterpriseAgentLifecycleProofStack.build(tmp_path, db_path=tmp_path / "d1-pipeline.db")
        self.stack.install_from_catalog()
        self.stack.bind_enabled_agent()
        built = self.stack.build_revision()
        before = self.stack.admin.inspect_serving(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
        )
        assert built.artifact_locator is not None and built.materialization_artifact_digest is not None
        self.revision_id = built.runtime_revision_id
        self.input_bundle = build_production_registry_projection_input_bundle_for_revision(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
            runtime_revision_id=built.runtime_revision_id,
            manifest=self.stack.canonical.manifest,
            build_context=ApplicationBuildContext.for_manifest(self.stack.canonical.manifest),
            authority=self.stack.durable_runtime.registry_projection_authority,
        )
        self.activation_request = ActivateRuntimeRevisionRequest(
            mutation_id="mut-d1-pipeline-hook",
            runtime_revision_id=built.runtime_revision_id,
            artifact_locator=built.artifact_locator,
            expected_artifact_digest=built.materialization_artifact_digest,
            expected_serving_pointer_revision=before.serving_pointer_revision,
            expected_prior_traffic_revision_id=before.traffic_serving_revision_id,
        )

    def context(self, **extra: object) -> DeployActivatePipelineContext:
        launcher = self.stack.launcher
        return DeployActivatePipelineContext(
            composition=launcher._composition,
            services=launcher._services,
            mutation_authorization_boundary=launcher._mutation_authorization_boundary,  # type: ignore[arg-type]
            environment_tenant_resolver=launcher._environment_tenant_resolver,  # type: ignore[arg-type]
            **extra,  # type: ignore[arg-type]
        )

    def run(self, ctx: DeployActivatePipelineContext):
        return run_deploy_and_activate(
            ctx,
            self.input_bundle,
            self.activation_request,
            principal=self.stack.canonical.governance.principal,
            admission_mutation_id=reference_admission_mutation_id(self.revision_id),
        )

    def traffic_revision(self) -> str | None:
        return self.stack.admin.inspect_serving(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
        ).traffic_serving_revision_id


def test_pipeline_refuses_activation_when_the_trust_hook_denies_the_revision(tmp_path: Path) -> None:
    scenario = _Scenario(tmp_path)
    seen: list[str] = []

    def _deny(revision: RuntimeRevision) -> None:
        seen.append(revision.runtime_revision_id)
        raise AgentPackageTrustError(
            "package digest is revoked at install admission",
            reason_code="package_digest_revoked",
        )

    with pytest.raises(AgentPackageTrustError):
        scenario.run(scenario.context(revision_trust_admission=_deny))

    assert seen == [scenario.revision_id]
    assert scenario.traffic_revision() is None  # nothing was committed


def test_pipeline_activates_when_the_trust_hook_allows_the_revision(tmp_path: Path) -> None:
    scenario = _Scenario(tmp_path)
    seen: list[str] = []

    def _allow(revision: RuntimeRevision) -> None:
        seen.append(revision.runtime_revision_id)

    result = scenario.run(scenario.context(revision_trust_admission=_allow))

    assert seen == [scenario.revision_id]
    assert result.runtime_revision_id == scenario.revision_id
    assert scenario.traffic_revision() == scenario.revision_id


def test_limitation_pipeline_without_a_hook_activates_as_before(tmp_path: Path) -> None:
    """Named limitation: the hook is optional, so a composition that does not pass it has no check."""
    scenario = _Scenario(tmp_path)

    result = scenario.run(scenario.context())

    assert result.runtime_revision_id == scenario.revision_id
