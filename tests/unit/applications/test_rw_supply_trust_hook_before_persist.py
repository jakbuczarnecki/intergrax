# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-B Part C: the trust hook runs before the revision is persisted (NT-RW-SUP-05, A-01).

Drives the real deploy/activate pipeline over the enterprise proof stack (SQLite under ``tmp_path``,
no network). The refusals are in-process stubs of a trust decision (labelled as such): they stand in
for a revoked package, no real revocation feed is used.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.agent_distribution.errors import AgentPackageTrustError
from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from intergrax.applications._shared import deploy_activate_pipeline as pipeline
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications._shared.registry_projection_input_bundle import reference_admission_mutation_id
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from testing_support.canonical_agent_lifecycle_composition import default_stage15_proof_config
from testing_support.enterprise_agent_lifecycle_composition import EnterpriseAgentLifecycleProofStack

pytestmark = [pytest.mark.integration]


@pytest.fixture(autouse=True)
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr("intergrax.applications._shared.llm_resolver.resolve_llm_adapter", _resolve)


def _refuse(seen: list[str]) -> Callable[[RuntimeRevision], None]:
    def _deny(revision: RuntimeRevision) -> None:
        seen.append(revision.runtime_revision_id)
        raise AgentPackageTrustError("stub: package digest revoked", reason_code="package_digest_revoked")

    return _deny


class _Scenario:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.config = default_stage15_proof_config()
        self.stack = EnterpriseAgentLifecycleProofStack.build(tmp_path, db_path=tmp_path / "rw-supply.db")
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
            mutation_id="mut-rw-supply-hook",
            runtime_revision_id=built.runtime_revision_id,
            artifact_locator=built.artifact_locator,
            expected_artifact_digest=built.materialization_artifact_digest,
            expected_serving_pointer_revision=before.serving_pointer_revision,
            expected_prior_traffic_revision_id=before.traffic_serving_revision_id,
        )
        self.writes: list[str] = []
        services = self.stack.launcher.services
        for owner, name in (
            (services.revision_service, "persist_candidate_revision"),
            (services.revision_service, "mark_validated"),
            (services.projection_input_store, "register"),
            (services.activation_service, "prepare_candidate"),
        ):
            monkeypatch.setattr(owner, name, self._recording(name, getattr(owner, name)))

    def _recording(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def _call(*args: Any, **kwargs: Any) -> Any:
            self.writes.append(name)
            return original(*args, **kwargs)

        return _call

    def context(self, **changes: Any) -> pipeline.DeployActivatePipelineContext:
        launcher = self.stack.launcher
        boundary = launcher._mutation_authorization_boundary
        resolver = launcher._environment_tenant_resolver
        assert boundary is not None and resolver is not None
        base = pipeline.DeployActivatePipelineContext(
            composition=launcher.process_composition,
            services=launcher.services,
            mutation_authorization_boundary=boundary,
            environment_tenant_resolver=resolver,
        )
        return dataclasses.replace(base, **changes)

    def run(self, ctx: pipeline.DeployActivatePipelineContext) -> pipeline.ReferenceProductionLifecycleResult:
        return pipeline.run_deploy_and_activate(
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


def test_sup_05a_refused_revision_not_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _Scenario(tmp_path, monkeypatch)
    seen: list[str] = []
    with pytest.raises(AgentPackageTrustError):
        scenario.run(scenario.context(revision_trust_admission=_refuse(seen)))
    assert seen == [scenario.revision_id]
    assert "persist_candidate_revision" not in scenario.writes
    assert scenario.traffic_revision() is None


def test_sup_05b_nothing_validated_registered_prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _Scenario(tmp_path, monkeypatch)
    with pytest.raises(AgentPackageTrustError):
        scenario.run(scenario.context(revision_trust_admission=_refuse([])))
    assert scenario.writes == []


def test_sup_05c_hook_resolved_from_composed_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _Scenario(tmp_path, monkeypatch)
    capability_runtime = scenario.stack.launcher.process_composition.agent_capability_runtime
    assert capability_runtime is not None
    seen: list[str] = []
    monkeypatch.setattr(capability_runtime.admin_service, "assert_revision_trust_admissible", _refuse(seen))
    with pytest.raises(AgentPackageTrustError):
        scenario.run(scenario.context())
    assert seen == [scenario.revision_id]
    assert scenario.writes == []
    assert scenario.traffic_revision() is None


def test_sup_05d_no_runtime_pass_through_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _Scenario(tmp_path, monkeypatch)
    composition = dataclasses.replace(scenario.stack.launcher.process_composition, agent_capability_runtime=None)
    result = scenario.run(scenario.context(composition=composition))
    assert result.runtime_revision_id == scenario.revision_id
    assert getattr(result, "trust_admission_state", None) == "trust_admission_not_composed"
    assert scenario.traffic_revision() == scenario.revision_id
