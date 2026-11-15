# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-A / F-D1-11 (durable CAS e2e): re-registering a persisted revision is idempotent.

A second composition that opens the same durable store and registers the projection input of an
already persisted revision must be judged by the persisted authority, not by the in-memory shape of
the manifest. The persisted descriptor decodes ``integration_profile`` bindings to slug references
while the production builder emits manifest-carrying bindings: same slug, different JSON shape.
A genuinely different manifest must still be refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications._shared.registry_projection import RegistryProjectionError
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


def _persist_one_revision(tmp_path: Path):
    config = default_stage15_proof_config()
    db_path = tmp_path / "durable-inputs.db"
    stack_a = EnterpriseAgentLifecycleProofStack.build(tmp_path / "a", db_path=db_path)
    stack_a.install_from_catalog()
    stack_a.bind_enabled_agent()
    built = stack_a.build_revision()
    serving = stack_a.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    bundle = build_production_registry_projection_input_bundle_for_revision(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
        runtime_revision_id=built.runtime_revision_id,
        manifest=stack_a.canonical.manifest,
        build_context=ApplicationBuildContext.for_manifest(stack_a.canonical.manifest),
        authority=stack_a.durable_runtime.registry_projection_authority,
    )
    assert built.artifact_locator is not None and built.materialization_artifact_digest is not None
    stack_a.launcher.deploy_and_activate(
        projection_input=bundle,
        activation_request=ActivateRuntimeRevisionRequest(
            mutation_id="mut-d1-durable-inputs",
            runtime_revision_id=built.runtime_revision_id,
            artifact_locator=built.artifact_locator,
            expected_artifact_digest=built.materialization_artifact_digest,
            expected_serving_pointer_revision=serving.serving_pointer_revision,
            expected_prior_traffic_revision_id=serving.traffic_serving_revision_id,
        ),
        principal=stack_a.canonical.governance.principal,
        admission_mutation_id=reference_admission_mutation_id(built.runtime_revision_id),
    )
    stack_b = EnterpriseAgentLifecycleProofStack.reopen(tmp_path / "b", db_path, config)
    return stack_b, bundle


def test_second_composition_can_reregister_the_inputs_of_a_persisted_revision(tmp_path: Path) -> None:
    stack_b, bundle = _persist_one_revision(tmp_path)

    stack_b.launcher._services.projection_input_store.register(bundle)  # idempotent, must not raise


def test_second_composition_still_refuses_inputs_with_a_different_manifest(tmp_path: Path) -> None:
    stack_b, bundle = _persist_one_revision(tmp_path)
    tampered_manifest = bundle.manifest.model_copy(update={"description": "tampered after persistence"})
    tampered = type(bundle)(
        runtime_revision=bundle.runtime_revision,
        effective_roster=bundle.effective_roster,
        manifest=tampered_manifest,
        build_context=bundle.build_context,
        composition=bundle.composition,
        factory_resolver=bundle.factory_resolver,
        builders=bundle.builders,
        materialization_artifact_digest=bundle.materialization_artifact_digest,
        roster_contract_authority=bundle.roster_contract_authority,
    )

    with pytest.raises(RegistryProjectionError, match="durable descriptor authority mismatch"):
        stack_b.launcher._services.projection_input_store.register(tampered)
