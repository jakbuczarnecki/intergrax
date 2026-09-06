# © Artur Czarnecki. All rights reserved.

"""P1.4 — runtime inspection read model."""

from __future__ import annotations

import json

import pytest

from intergrax.applications._shared.capability_dependency import validate_capability_dependencies
from intergrax.applications._shared.profile_resolution import (
    InMemoryEffectiveProfileExecutionPinningStore,
    InMemoryEffectiveProfileRevisionStore,
    materialize_effective_profile_revision,
    pin_effective_profile_revision_for_execution,
    resolve_profile,
)
from intergrax.applications._shared.profile_resolution.redaction import encode_provenance_value
from intergrax.applications._shared.runtime_inspection import (
    RuntimeInspectionService,
    profile_contains_no_raw_secrets,
    redacted_profile_snapshot,
)
from intergrax.applications.contracts.capability_dependency import (
    CapabilityDependency,
    CapabilityDependencyAvailabilityStatus,
    CapabilityDependencyKind,
    CapabilityDependencyRequirement,
    CapabilityDependencyValidationContext,
)
from intergrax.applications.contracts.capability_dependency.dependency import CapabilityRef
from intergrax.applications.contracts.capability_health import CapabilityHealthStatus
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import CostProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.profile_resolution import (
    EffectiveProfileRevisionScope,
    ProfileDelta,
    ProfileFieldUpdate,
    ProfileLayer,
    ProfileLayerInput,
    ProfileResolutionDecisionKind,
)
from intergrax.applications.contracts.runtime_inspection import (
    InspectionCompleteness,
    InspectionExtensionEvidence,
    InspectionInconsistencyKind,
    InspectionProviderContribution,
    InspectionScope,
    RuntimeInspectionProvider,
)
from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.tools.registry.profile import ToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_SCOPE = EffectiveProfileRevisionScope(application_id="inspection.test", tenant_id="tenant-a")
_SCOPE_B = EffectiveProfileRevisionScope(application_id="inspection.test", tenant_id="tenant-b")
_RAW_SECRET = "RAW_SECRET_123"


def _application(
    *,
    tools: list[str] | None = None,
    execution_mode: ExecutionMode = ExecutionMode.BALANCED,
    max_tool_calls: int | None = None,
) -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="inspection.test")
    updates: dict[str, object] = {
        "meta": profile.meta.model_copy(update={"execution_mode": execution_mode}),
        "capabilities": profile.capabilities.model_copy(
            update={
                "llm": LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
                "tools": ToolProfile(enabled=tools or ["search", "calculator"]),
            },
        ),
    }
    if max_tool_calls is not None:
        updates["governance"] = profile.governance.model_copy(
            update={"cost": CostProfile(max_tool_calls=max_tool_calls)},
        )
    return profile.model_copy(update=updates)


def _application_with_secret(*, secret: str = _RAW_SECRET) -> ApplicationEnvironmentProfile:
    profile = _application()
    integration = profile.capabilities.integrations.model_copy(
        update={
            "options": {
                **profile.capabilities.integrations.options,
                "inspection.test.secret": {"api_token": secret},
            },
        },
    )
    return profile.model_copy(
        update={
            "capabilities": profile.capabilities.model_copy(
                update={"integrations": integration},
            ),
        },
    )


def _assert_direct_serialization_excludes_secret(result: object, *, raw_secret: str) -> None:
    dumped = result.model_dump(mode="json")
    assert profile_contains_no_raw_secrets(dumped, raw_secret=raw_secret)
    assert profile_contains_no_raw_secrets(result.model_dump_json(), raw_secret=raw_secret)
    assert profile_contains_no_raw_secrets(
        json.dumps(dumped, sort_keys=True),
        raw_secret=raw_secret,
    )


def _service(
    *,
    revision_store: InMemoryEffectiveProfileRevisionStore | None = None,
    pinning_store: InMemoryEffectiveProfileExecutionPinningStore | None = None,
    providers: tuple[RuntimeInspectionProvider, ...] | None = None,
) -> RuntimeInspectionService:
    return RuntimeInspectionService(
        revision_store=revision_store,
        pinning_store=pinning_store,
        providers=providers,
    )


def _revision(
    application: ApplicationEnvironmentProfile,
    layers: tuple[ProfileLayerInput, ...],
    *,
    store: InMemoryEffectiveProfileRevisionStore | None = None,
):
    resolution = resolve_profile(application, layers=layers)
    return resolution, materialize_effective_profile_revision(
        resolution,
        scope=_SCOPE,
        store=store,
    )


class _SyntheticDependencyProvider:
    def __init__(
        self,
        *,
        provider_id: str | None = None,
        source_domain: str,
        declarations: tuple[CapabilityDependency, ...],
        availability: dict[tuple[str, str, str], tuple[CapabilityDependencyAvailabilityStatus, str]],
    ) -> None:
        self._provider_id = provider_id or source_domain
        self._source_domain = source_domain
        self._declarations = declarations
        self._availability = availability

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def source_domain(self) -> str:
        return self._source_domain

    def dependencies_for(
        self,
        context: CapabilityDependencyValidationContext,
    ) -> tuple[CapabilityDependency, ...]:
        del context
        return self._declarations

    def evaluate_availability(
        self,
        dependency: CapabilityDependency,
        context: CapabilityDependencyValidationContext,
    ) -> tuple[CapabilityDependencyAvailabilityStatus, str]:
        del context
        return self._availability[dependency.dedup_key]


class _CustomInspectionProvider:
    @property
    def provider_id(self) -> str:
        return "custom.inspection"

    def contribute_profile(
        self,
        *,
        resolution,
        configured_profile_ref: str | None,
    ) -> InspectionProviderContribution:
        del resolution, configured_profile_ref
        return InspectionProviderContribution(
            provider_id=self.provider_id,
            extension_evidence=(
                InspectionExtensionEvidence(
                    provider_id=self.provider_id,
                    scope=InspectionScope.PROFILE,
                    subject="custom",
                    payload={"marker": "present"},
                ),
            ),
        )

    def contribute_revision(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_execution(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_capability(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_revision_compare(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)


class _FailingInspectionProvider:
    @property
    def provider_id(self) -> str:
        return "failing.inspection"

    def contribute_profile(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        raise RuntimeError("provider failed")

    def contribute_revision(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_execution(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_capability(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_revision_compare(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)


class _SecretFailingInspectionProvider:
    @property
    def provider_id(self) -> str:
        return "secret.failing.inspection"

    def contribute_profile(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        raise RuntimeError(f"token={_RAW_SECRET}")

    def contribute_revision(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_execution(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_capability(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_revision_compare(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)


class _SecretPayloadInspectionProvider:
    @property
    def provider_id(self) -> str:
        return "secret.payload.inspection"

    def contribute_profile(
        self,
        *,
        resolution,
        configured_profile_ref: str | None,
    ) -> InspectionProviderContribution:
        del resolution, configured_profile_ref
        return InspectionProviderContribution(
            provider_id=self.provider_id,
            extension_evidence=(
                InspectionExtensionEvidence(
                    provider_id=self.provider_id,
                    scope=InspectionScope.PROFILE,
                    subject="secret_payload",
                    payload={"marker": _RAW_SECRET},
                ),
            ),
        )

    def contribute_revision(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_execution(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_capability(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_revision_compare(self, **kwargs) -> InspectionProviderContribution:
        del kwargs
        return InspectionProviderContribution(provider_id=self.provider_id)


def test_profile_explain_clamped_decision_from_existing_resolution() -> None:
    application = _application(tools=["search"], max_tool_calls=10)
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    cost_profile=ProfileFieldUpdate(value=CostProfile(max_tool_calls=5)),
                ),
            ),
            ProfileLayerInput(
                layer=ProfileLayer.EXECUTION,
                delta=ProfileDelta(
                    cost_profile=ProfileFieldUpdate(value=CostProfile(max_tool_calls=25)),
                ),
            ),
        ),
    )
    result = _service().inspect_profile(
        resolution,
        configured_profile_ref="configured/profile",
    )
    assert result.resolution is resolution
    assert result.resolution.decisions is resolution.decisions
    clamped = [
        item
        for item in result.resolution.decisions
        if item.path == "governance.cost.max_tool_calls"
        and item.decision is ProfileResolutionDecisionKind.CLAMPED
        and item.source_layer is ProfileLayer.EXECUTION
    ]
    assert clamped
    assert clamped[0].requested_value == "25"
    assert clamped[0].effective_value == "5"
    assert clamped[0].source_layer is ProfileLayer.EXECUTION
    assert any(item.layer is ProfileLayer.APPLICATION for item in resolution.layers)


def test_profile_explain_rejected_overlay_authority_widen() -> None:
    application = _application(tools=["search"], max_tool_calls=10)
    resolution = resolve_profile(
        application,
        layers=(
            ProfileLayerInput(
                layer=ProfileLayer.EXECUTION,
                delta=ProfileDelta(
                    cost_profile=ProfileFieldUpdate(value=CostProfile(max_tool_calls=25)),
                ),
            ),
        ),
    )
    result = _service().inspect_profile(resolution)
    decision = next(
        item
        for item in result.resolution.decisions
        if item.path == "governance.cost.max_tool_calls"
        and item.source_layer is ProfileLayer.EXECUTION
    )
    assert decision.decision is ProfileResolutionDecisionKind.CLAMPED
    assert decision.effective_value == "10"
    assert decision.requested_value == "25"
    assert decision.source_layer is ProfileLayer.EXECUTION


def test_revision_inspect_exact_snapshot() -> None:
    store = InMemoryEffectiveProfileRevisionStore()
    _, revision = _revision(_application(), (), store=store)
    result = _service(revision_store=store).inspect_revision(
        revision.revision_id,
        scope=_SCOPE,
    )
    assert result.revision is not None
    assert result.revision.revision_id == revision.revision_id
    assert result.revision.fingerprint == revision.fingerprint
    assert result.revision.effective_profile == revision.effective_profile
    assert result.completeness is InspectionCompleteness.COMPLETE


def test_revision_diff_reuses_semantic_paths() -> None:
    store = InMemoryEffectiveProfileRevisionStore()
    _, revision_a = _revision(_application(max_tool_calls=10), (), store=store)
    _, revision_b = _revision(
        _application(max_tool_calls=5),
        (
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    cost_profile=ProfileFieldUpdate(value=CostProfile(max_tool_calls=5)),
                ),
            ),
        ),
        store=store,
    )
    compare = _service(revision_store=store).compare_revisions(revision_a, revision_b)
    paths = {entry.path for entry in compare.safe_diff.entries}
    assert "governance.cost.max_tool_calls" in paths
    assert compare.safe_diff.from_revision_id == revision_a.revision_id
    assert compare.safe_diff.to_revision_id == revision_b.revision_id


def test_execution_inspect_returns_pinned_revision_not_latest() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    _, revision_a = _revision(_application(), (), store=revision_store)
    _, revision_b = _revision(
        _application(max_tool_calls=3),
        (
            ProfileLayerInput(
                layer=ProfileLayer.PLATFORM,
                delta=ProfileDelta(
                    cost_profile=ProfileFieldUpdate(value=CostProfile(max_tool_calls=3)),
                ),
            ),
        ),
        store=revision_store,
    )
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision_a,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    result = _service(
        revision_store=revision_store,
        pinning_store=pinning_store,
    ).inspect_execution(
        tenant_id="tenant-a",
        execution_id=execution_id,
        scope_application_id=_SCOPE.application_id,
        scope_tenant_id=_SCOPE.tenant_id,
    )
    assert result.pinned_revision is not None
    assert result.pinned_revision.revision_id == revision_a.revision_id
    assert result.pinned_revision.revision_id != revision_b.revision_id


def test_execution_missing_revision_reports_inconsistency() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    _, revision = _revision(_application(), (), store=revision_store)
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    empty_store = InMemoryEffectiveProfileRevisionStore()
    result = _service(
        revision_store=empty_store,
        pinning_store=pinning_store,
    ).inspect_execution(
        tenant_id="tenant-a",
        execution_id=execution_id,
        scope_application_id=_SCOPE.application_id,
        scope_tenant_id=_SCOPE.tenant_id,
    )
    assert any(
        item.kind is InspectionInconsistencyKind.MISSING_REVISION
        for item in result.inconsistencies
    )
    assert result.pinned_revision is None


def test_execution_fingerprint_mismatch_reports_inconsistency() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    _, revision = _revision(_application(), (), store=revision_store)
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    binding = pinning_store.get(tenant_id="tenant-a", execution_id=execution_id)
    assert binding is not None
    pinning_store._bindings[(binding.tenant_id, binding.execution_id)] = (  # noqa: SLF001
        binding.model_copy(update={"fingerprint": "mismatch"})
    )
    result = _service(
        revision_store=revision_store,
        pinning_store=pinning_store,
    ).inspect_execution(
        tenant_id="tenant-a",
        execution_id=execution_id,
        scope_application_id=_SCOPE.application_id,
        scope_tenant_id=_SCOPE.tenant_id,
    )
    assert any(
        item.kind is InspectionInconsistencyKind.FINGERPRINT_MISMATCH
        for item in result.inconsistencies
    )


def test_capability_required_dependency_failure_evidence() -> None:
    owner = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.x")
    dependency = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.y")
    declaration = CapabilityDependency(
        owner=owner,
        dependency=dependency,
        requirement=CapabilityDependencyRequirement.REQUIRED,
        source_domains=("synthetic",),
    )
    provider = _SyntheticDependencyProvider(
        source_domain="synthetic",
        declarations=(declaration,),
        availability={
            declaration.dedup_key: (
                CapabilityDependencyAvailabilityStatus.UNAVAILABLE,
                "tool missing",
            ),
        },
    )
    validation = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=_application()),
        providers=(provider,),
    )
    result = _service().inspect_capability(owner, validation)
    assert len(result.required_failures) == 1
    failure = result.required_failures[0]
    assert failure.owner == owner
    assert failure.dependency == dependency
    assert failure.requirement is CapabilityDependencyRequirement.REQUIRED
    assert failure.status is CapabilityDependencyAvailabilityStatus.UNAVAILABLE
    assert failure.source_domains == ("synthetic",)


def test_capability_optional_degradation_evidence() -> None:
    owner = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.x")
    dependency = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.opt")
    declaration = CapabilityDependency(
        owner=owner,
        dependency=dependency,
        requirement=CapabilityDependencyRequirement.OPTIONAL,
        source_domains=("synthetic",),
    )
    provider = _SyntheticDependencyProvider(
        source_domain="synthetic",
        declarations=(declaration,),
        availability={
            declaration.dedup_key: (
                CapabilityDependencyAvailabilityStatus.UNKNOWN,
                "optional unknown",
            ),
        },
    )
    validation = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=_application()),
        providers=(provider,),
    )
    result = _service().inspect_capability(owner, validation)
    assert len(result.optional_degradations) == 1
    assert result.optional_degradations[0].requirement is CapabilityDependencyRequirement.OPTIONAL
    assert result.outcome is not None
    assert result.outcome.degraded is True
    assert result.health.status.value == "degraded"
    assert result.completeness is InspectionCompleteness.COMPLETE


def test_capability_multi_source_provenance_preserved() -> None:
    owner = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.x")
    dependency = CapabilityRef(kind=CapabilityDependencyKind.TOOL, capability_id="tool.y")
    declaration_a = CapabilityDependency(
        owner=owner,
        dependency=dependency,
        requirement=CapabilityDependencyRequirement.REQUIRED,
        source_domains=("domain.a",),
    )
    declaration_b = CapabilityDependency(
        owner=owner,
        dependency=dependency,
        requirement=CapabilityDependencyRequirement.REQUIRED,
        source_domains=("domain.b",),
    )
    provider_a = _SyntheticDependencyProvider(
        provider_id="provider.a",
        source_domain="domain.a",
        declarations=(declaration_a,),
        availability={
            declaration_a.dedup_key: (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "ok",
            ),
        },
    )
    provider_b = _SyntheticDependencyProvider(
        provider_id="provider.b",
        source_domain="domain.b",
        declarations=(declaration_b,),
        availability={
            declaration_b.dedup_key: (
                CapabilityDependencyAvailabilityStatus.UNAVAILABLE,
                "missing in b",
            ),
        },
    )
    validation = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=_application()),
        providers=(provider_b, provider_a),
    )
    result = _service().inspect_capability(owner, validation)
    failure = result.required_failures[0]
    assert failure.source_domains == ("domain.a", "domain.b")


def test_tenant_isolation_for_execution_inspection() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    _, revision = _revision(_application(), (), store=revision_store)
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    result = _service(
        revision_store=revision_store,
        pinning_store=pinning_store,
    ).inspect_execution(
        tenant_id="tenant-b",
        execution_id=execution_id,
        scope_application_id=_SCOPE.application_id,
        scope_tenant_id="tenant-b",
    )
    assert result.binding is None
    assert any(
        item.kind is InspectionInconsistencyKind.NOT_FOUND
        for item in result.inconsistencies
    )


def test_custom_inspection_provider_appends_evidence() -> None:
    resolution = resolve_profile(_application(), layers=())
    result = _service(providers=(_CustomInspectionProvider(),)).inspect_profile(resolution)
    assert any(item.provider_id == "custom.inspection" for item in result.extension_evidence)


def test_optional_provider_failure_marks_partial() -> None:
    resolution = resolve_profile(_application(), layers=())
    result = _service(providers=(_FailingInspectionProvider(),)).inspect_profile(resolution)
    assert result.completeness is InspectionCompleteness.PARTIAL
    assert len(result.provider_failures) == 1
    assert result.provider_failures[0].provider_id == "failing.inspection"


def test_deterministic_ordering_with_reversed_providers() -> None:
    resolution = resolve_profile(_application(), layers=())
    providers = (_CustomInspectionProvider(), _FailingInspectionProvider())
    first = _service(providers=providers).inspect_profile(resolution)
    second = _service(providers=tuple(reversed(providers))).inspect_profile(resolution)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_sensitive_redaction_does_not_emit_raw_secret() -> None:
    encoded = encode_provenance_value("capabilities.llm.api_key", "raw-secret-value")
    assert encoded is not None
    assert "raw-secret-value" not in encoded
    assert encoded.startswith("hash:")
    profile = _application()
    snapshot = redacted_profile_snapshot(profile)
    assert profile_contains_no_raw_secrets(snapshot, raw_secret="raw-secret-value")


def test_inspection_is_read_only_no_store_mutation() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    _, revision = _revision(_application(), (), store=revision_store)
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    service = _service(revision_store=revision_store, pinning_store=pinning_store)
    before_revision_keys = set(revision_store._revisions.keys())  # noqa: SLF001
    before_binding_keys = set(pinning_store._bindings.keys())  # noqa: SLF001
    service.inspect_revision(revision.revision_id, scope=_SCOPE)
    service.inspect_execution(
        tenant_id="tenant-a",
        execution_id=execution_id,
        scope_application_id=_SCOPE.application_id,
        scope_tenant_id=_SCOPE.tenant_id,
    )
    assert set(revision_store._revisions.keys()) == before_revision_keys  # noqa: SLF001
    assert set(pinning_store._bindings.keys()) == before_binding_keys  # noqa: SLF001


def test_profile_direct_serialization_excludes_raw_secret() -> None:
    resolution = resolve_profile(_application_with_secret(), layers=())
    result = _service().inspect_profile(resolution)
    assert result.resolution is resolution
    _assert_direct_serialization_excludes_secret(result, raw_secret=_RAW_SECRET)
    assert result.safe_resolution.fingerprint == resolution.fingerprint


def test_revision_direct_serialization_excludes_raw_secret() -> None:
    store = InMemoryEffectiveProfileRevisionStore()
    _, revision = _revision(_application_with_secret(), (), store=store)
    result = _service(revision_store=store).inspect_revision(
        revision.revision_id,
        scope=_SCOPE,
    )
    assert result.revision is revision
    _assert_direct_serialization_excludes_secret(result, raw_secret=_RAW_SECRET)
    assert result.safe_revision is not None
    assert result.safe_revision.revision_id == revision.revision_id


def test_execution_direct_serialization_excludes_raw_secret() -> None:
    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    _, revision = _revision(_application_with_secret(), (), store=revision_store)
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    result = _service(
        revision_store=revision_store,
        pinning_store=pinning_store,
    ).inspect_execution(
        tenant_id="tenant-a",
        execution_id=execution_id,
        scope_application_id=_SCOPE.application_id,
        scope_tenant_id=_SCOPE.tenant_id,
    )
    assert result.pinned_revision is revision
    _assert_direct_serialization_excludes_secret(result, raw_secret=_RAW_SECRET)
    assert result.safe_pinned_revision is not None
    assert result.safe_pinned_revision.fingerprint == revision.fingerprint


def test_revision_compare_direct_serialization_redacts_sensitive_diff_values() -> None:
    store = InMemoryEffectiveProfileRevisionStore()
    profile_a = _application()
    profile_a = profile_a.model_copy(
        update={
            "capabilities": profile_a.capabilities.model_copy(
                update={
                    "llm": LLMProfile(provider=LLMProvider.OPENAI, model=_RAW_SECRET),
                },
            ),
        },
    )
    profile_b = _application()
    _, revision_a = _revision(profile_a, (), store=store)
    _, revision_b = _revision(profile_b, (), store=store)
    compare = _service(revision_store=store).compare_revisions(revision_a, revision_b)
    _assert_direct_serialization_excludes_secret(compare, raw_secret=_RAW_SECRET)
    assert compare.safe_diff.from_fingerprint == revision_a.fingerprint
    assert any(entry.path == "capabilities.llm.model" for entry in compare.safe_diff.entries)


def test_safe_serialization_retains_non_sensitive_facts() -> None:
    resolution = resolve_profile(_application_with_secret(), layers=())
    result = _service().inspect_profile(resolution)
    payload = result.model_dump(mode="json")
    assert payload["safe_resolution"]["fingerprint"] == resolution.fingerprint
    assert payload["safe_resolution"]["decisions"]
    assert payload["completeness"] == InspectionCompleteness.COMPLETE.value


def test_provider_exception_secret_is_sanitized_in_serialized_result() -> None:
    resolution = resolve_profile(_application(), layers=())
    result = _service(providers=(_SecretFailingInspectionProvider(),)).inspect_profile(resolution)
    _assert_direct_serialization_excludes_secret(result, raw_secret=_RAW_SECRET)
    assert result.completeness is InspectionCompleteness.PARTIAL
    assert result.provider_failures[0].provider_id == "secret.failing.inspection"
    assert result.provider_failures[0].reason.startswith("RuntimeError:")


def test_provider_extension_payload_is_defensively_redacted() -> None:
    resolution = resolve_profile(_application(), layers=())
    result = _service(providers=(_SecretPayloadInspectionProvider(),)).inspect_profile(resolution)
    _assert_direct_serialization_excludes_secret(result, raw_secret=_RAW_SECRET)
    assert result.extension_evidence[0].subject == "secret_payload"


def test_internal_canonical_fields_excluded_from_serialization() -> None:
    resolution = resolve_profile(_application_with_secret(), layers=())
    result = _service().inspect_profile(resolution)
    payload = result.model_dump(mode="json")
    assert "resolution" not in payload
    assert result.resolution is resolution


def test_inspect_capability_no_evidence_unavailable_with_safe_reason() -> None:
    capability = CapabilityRef(kind=CapabilityDependencyKind.SKILL, capability_id="skill.orphan")
    validation = validate_capability_dependencies(
        CapabilityDependencyValidationContext(environment_profile=_application()),
    )
    result = _service().inspect_capability(capability, validation)
    assert result.health.status is CapabilityHealthStatus.UNAVAILABLE
    assert result.safe_health.status is CapabilityHealthStatus.UNAVAILABLE
    assert any(
        item.reason_code == "capability.health.evidence_missing"
        for item in result.safe_health.reasons
    )
    serialized = result.model_dump_json()
    assert "capability.health.evidence_missing" in serialized
    payload = json.loads(serialized)
    assert payload["safe_health"]["status"] == CapabilityHealthStatus.UNAVAILABLE.value
