# © Artur Czarnecki. All rights reserved.

"""DS-PLUGIN platform integration tests for Decision domain registry composition."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.decision_artifact_registry import (
    decision_artifact_kind_registry,
    is_decision_artifact_kind_registered,
    require_registered_decision_artifact_kind,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifact,
    DecisionVersionLineage,
    candidate_decision_ref,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_strategy import (
    DecisionStrategy,
    DecisionStrategyKind,
    DecisionStrategyRegistration,
    decision_strategy_registry,
    is_decision_strategy_registered,
    register_decision_strategy,
    require_registered_decision_strategy,
    validate_decision_strategy_kind,
)
from intergrax.contracts.decision_verification import (
    VerificationStageOutcome,
    validate_verification_stage_kind,
    verification_stage_record,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStage,
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    verification_stage_registry,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.hybrid_strategy import (
    HybridPhase,
    hybrid_strategy,
    validate_hybrid_phase_id,
    validate_hybrid_strategy_registry_bindings,
)
from intergrax.core.distribution import PlatformCompatibility, check_platform_compatibility
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import (
    EP_DECISION_ARTIFACT_KINDS,
    EP_DECISION_STRATEGIES,
    EP_DECISION_VERIFICATION_STAGES,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.core.plugins.package_contract import CapabilityDescriptor
from intergrax.core.plugins.platform_qualification import (
    PluginQualificationEvidenceKind,
    PluginQualificationLevel,
    build_external_package_subject,
    build_qualification_result,
    compatibility_evidence,
)
from intergrax.core.qualification import QualificationEvidence, QualificationStatus
from intergrax.runtime.decision_plugin_composition import (
    DECISION_ARTIFACT_KIND_CAPABILITY_ID,
    DECISION_PLUGIN_DOMAIN,
    DECISION_STRATEGY_CAPABILITY_ID,
    DECISION_VERIFICATION_STAGE_CAPABILITY_ID,
    DecisionArtifactKindContribution,
    DecisionPluginLoadPolicy,
    load_decision_artifact_kind_plugins,
    load_decision_strategy_plugins,
    load_verification_stage_plugins,
)
from intergrax.runtime.decision_verification import VerificationPipeline

pytestmark = pytest.mark.unit

_PACKAGE_NAME = "acme-decision-plugin"
_PACKAGE_VERSION = "1.0.0"
_PLATFORM_VERSION = "0.1.0"


class _Dist:
    def __init__(self, name: str, *, files: object | None = None) -> None:
        self.name = name
        self.version = _PACKAGE_VERSION
        self.files = files


class _EntryPoint:
    def __init__(
        self,
        name: str,
        value: str,
        group: str,
        *,
        distribution: str | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.group = group
        self.dist = _Dist(distribution) if distribution is not None else None


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


@dataclass(frozen=True, slots=True)
class _ExternalCouncilStrategy:
    kind: DecisionStrategyKind = validate_decision_strategy_kind("external_council_variant")


@dataclass(frozen=True, slots=True)
class _DuplicateKindStrategyA:
    kind: DecisionStrategyKind = validate_decision_strategy_kind("same_kind")


@dataclass(frozen=True, slots=True)
class _DuplicateKindStrategyB:
    kind: DecisionStrategyKind = validate_decision_strategy_kind("same_kind")


class _NotAStrategy:
    pass


@dataclass(frozen=True, slots=True)
class _Payload:
    value: str


@dataclass(frozen=True, slots=True)
class _ExternalVerificationStage:
    kind: str = validate_verification_stage_kind("external_domain_check")
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(
        self,
        candidate: CandidateDecision[_Payload],
    ) -> object:
        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=self.kind,
            outcome=VerificationStageOutcome.PASSED,
        )


@dataclass(frozen=True, slots=True)
class _ZzzVerificationStage:
    kind: str = validate_verification_stage_kind("zzz_plugin_stage")
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(
        self,
        candidate: CandidateDecision[_Payload],
    ) -> object:
        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=self.kind,
            outcome=VerificationStageOutcome.PASSED,
        )


@dataclass(frozen=True, slots=True)
class _AaaVerificationStage:
    kind: str = validate_verification_stage_kind("aaa_plugin_stage")
    execution_class: VerificationStageExecutionClass = (
        VerificationStageExecutionClass.DETERMINISTIC
    )

    async def verify(
        self,
        candidate: CandidateDecision[_Payload],
    ) -> object:
        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=self.kind,
            outcome=VerificationStageOutcome.PASSED,
        )


@dataclass(frozen=True, slots=True)
class _ExternalRiskArtifactKindContribution(DecisionArtifactKindContribution):
    kind: DecisionArtifactKind = validate_decision_artifact_kind("external_risk_decision")


@dataclass(frozen=True, slots=True)
class _DuplicateArtifactKindContribution(DecisionArtifactKindContribution):
    kind: DecisionArtifactKind = validate_decision_artifact_kind("external_risk_decision")


@dataclass(frozen=True, slots=True)
class _WhitespaceArtifactKindContribution(DecisionArtifactKindContribution):
    kind: DecisionArtifactKind = "   "


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _strategy_ep(name: str, attr: str, *, distribution: str | None = None) -> _EntryPoint:
    return _EntryPoint(name, f"{__name__}:{attr}", EP_DECISION_STRATEGIES, distribution=distribution)


def _verification_ep(name: str, attr: str, *, distribution: str | None = None) -> _EntryPoint:
    return _EntryPoint(
        name,
        f"{__name__}:{attr}",
        EP_DECISION_VERIFICATION_STAGES,
        distribution=distribution,
    )


def _artifact_ep(name: str, attr: str, *, distribution: str | None = None) -> _EntryPoint:
    return _EntryPoint(
        name,
        f"{__name__}:{attr}",
        EP_DECISION_ARTIFACT_KINDS,
        distribution=distribution,
    )


def _builtin_strategy_registry() -> object:
    alpha = DecisionStrategyRegistration(
        kind=validate_decision_strategy_kind("builtin_alpha"),
        strategy=_ExternalCouncilStrategy(
            kind=validate_decision_strategy_kind("builtin_alpha"),
        ),
    )
    return decision_strategy_registry((alpha,))


def _candidate() -> CandidateDecision[_Payload]:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="demo", subject="subject-1"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("demo.payload"),
        content=_Payload(value="ok"),
    )
    lineage = DecisionVersionLineage(current=decision_lineage_ref(identity.version))
    return CandidateDecision(identity=identity, artifact=artifact, lineage=lineage)


def _compatible_platform() -> object:
    return check_platform_compatibility(
        PlatformCompatibility(intergrax_version=">=0.1,<2"),
        _PLATFORM_VERSION,
    )


def _production_package_qualification(*, compatibility: object, group: str, name: str) -> object:
    return build_qualification_result(
        subject=build_external_package_subject(
            level=PluginQualificationLevel.PACKAGE,
            package_name=_PACKAGE_NAME,
            package_version=_PACKAGE_VERSION,
            domain=DECISION_PLUGIN_DOMAIN,
            entry_point_group=group,
            entry_point_name=name,
        ),
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(
            compatibility_evidence(compatibility),
            QualificationEvidence(
                kind=PluginQualificationEvidenceKind.DOMAIN_QUALIFICATION,
                code="decision.tests.passed",
                ref=__file__,
            ),
        ),
        reason="external decision plugin production-qualified",
    )


def _lookup_for(qualification: object | None) -> object:
    return lambda spec: qualification


def _mock_installed_distribution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest_toml: str | None = None,
) -> None:
    dist = MagicMock()
    dist.version = _PACKAGE_VERSION
    if manifest_toml is None:
        dist.files = None
    else:
        file = MagicMock()
        file.name = "pyproject.toml"
        dist.read_text.return_value = manifest_toml
        dist.files = [file]
    intergrax_dist = MagicMock()
    intergrax_dist.version = _PLATFORM_VERSION

    def _distribution(name: str) -> MagicMock:
        if name == _PACKAGE_NAME:
            return dist
        if name in ("intergrax", "Intergrax-ai"):
            return intergrax_dist
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", _distribution)


def _manifest_toml(
    *,
    capability: CapabilityDescriptor,
) -> str:
    return f"""
[project]
name = "{_PACKAGE_NAME}"
version = "{_PACKAGE_VERSION}"

[tool.intergrax.plugin]
name = "{_PACKAGE_NAME}"
version = "{_PACKAGE_VERSION}"
intergrax_version = ">=0.1,<2"

[[tool.intergrax.plugin.capabilities]]
domain = "{capability.domain}"
entry_point_group = "{capability.entry_point_group}"
entry_point_name = "{capability.entry_point_name}"
capability_ids = {list(capability.capability_ids)}
"""


# --- DS-PLUGIN-01 strategy matrix ---


def test_valid_strategy_plugin_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_strategy_ep("external_council", "_ExternalCouncilStrategy")])
    base = decision_strategy_registry()
    outcome = load_decision_strategy_plugins(
        base,
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 1
    strategy = require_registered_decision_strategy(
        outcome.registry,
        "external_council_variant",
    )
    assert isinstance(strategy, _ExternalCouncilStrategy)


def test_builtin_strategy_registry_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_strategy_ep("external_council", "_ExternalCouncilStrategy")])
    base = _builtin_strategy_registry()
    outcome = load_decision_strategy_plugins(base, discover_entry_points=True)
    assert is_decision_strategy_registered(outcome.registry, "builtin_alpha")
    assert is_decision_strategy_registered(outcome.registry, "external_council_variant")


def test_duplicate_semantic_strategy_kind_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _strategy_ep("package_a_strategy", "_DuplicateKindStrategyA"),
            _strategy_ep("package_b_strategy", "_DuplicateKindStrategyB"),
        ],
    )
    base = decision_strategy_registry()
    outcome = load_decision_strategy_plugins(base, discover_entry_points=True)
    assert outcome.report.registered_count == 1
    assert outcome.report.rejected[0].reason_code is PluginAdmissionReasonCode.PLUGIN_ID_COLLISION
    assert outcome.report.rejected[0].plugin_id == "same_kind"


def test_strategy_wrong_target_type_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_strategy_ep("bad", "_NotAStrategy")])
    base = decision_strategy_registry()
    outcome = load_decision_strategy_plugins(base, discover_entry_points=True)
    assert outcome.registry is base
    assert outcome.report.rejected[0].reason_code is PluginAdmissionReasonCode.INVALID_TARGET_TYPE


def test_strategy_load_failure_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _strategy_ep("good", "_ExternalCouncilStrategy"),
            _EntryPoint("broken", "not-a-valid-target", EP_DECISION_STRATEGIES),
        ],
    )
    base = decision_strategy_registry()
    outcome = load_decision_strategy_plugins(
        base,
        policy=DecisionPluginLoadPolicy(on_load_failure="isolate"),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 1
    assert [item.spec.name for item in outcome.report.failed] == ["broken"]


def test_strategy_production_admission_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [_strategy_ep("external_council", "_ExternalCouncilStrategy", distribution=_PACKAGE_NAME)],
    )
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: _compatible_platform(),
    )
    base = decision_strategy_registry()
    outcome = load_decision_strategy_plugins(
        base,
        policy=DecisionPluginLoadPolicy(
            require_production_admission=True,
            package_qualification_lookup=None,
            platform_version=_PLATFORM_VERSION,
        ),
        discover_entry_points=True,
    )
    assert outcome.registry is base
    assert outcome.report.rejected[0].reason_code is PluginAdmissionReasonCode.PRODUCTION_ADMISSION_DENIED


def test_hybrid_can_reference_plugin_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_strategy_ep("external_council", "_ExternalCouncilStrategy")])
    base = decision_strategy_registry()
    outcome = load_decision_strategy_plugins(base, discover_entry_points=True)
    hybrid = hybrid_strategy(
        phases=(
            HybridPhase(
                phase_id=validate_hybrid_phase_id("phase-1"),
                strategy_kind=validate_decision_strategy_kind("external_council_variant"),
            ),
        ),
    )
    validate_hybrid_strategy_registry_bindings(strategy=hybrid, registry=outcome.registry)


# --- DS-PLUGIN-02 verification matrix ---


@pytest.mark.asyncio
async def test_valid_verification_stage_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_verification_ep("external_check", "_ExternalVerificationStage")])
    base = verification_stage_registry()
    outcome = load_verification_stage_plugins(base, discover_entry_points=True)
    assert outcome.report.registered_count == 1
    pipeline = VerificationPipeline(registry=outcome.registry)
    result = await pipeline.verify(_candidate())
    assert result.disposition.value == "passed"
    assert any(record.stage == "external_domain_check" for record in result.stage_records)


def test_verification_duplicate_kind_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _verification_ep("stage_a", "_ExternalVerificationStage"),
            _verification_ep("stage_b", "_ExternalVerificationStage"),
        ],
    )
    base = verification_stage_registry()
    outcome = load_verification_stage_plugins(base, discover_entry_points=True)
    assert outcome.report.registered_count == 1
    assert outcome.report.rejected[0].reason_code is PluginAdmissionReasonCode.PLUGIN_ID_COLLISION


def test_verification_wrong_target_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_verification_ep("bad", "_NotAStrategy")])
    base = verification_stage_registry()
    outcome = load_verification_stage_plugins(base, discover_entry_points=True)
    assert outcome.registry is base
    assert outcome.report.rejected[0].reason_code is PluginAdmissionReasonCode.INVALID_TARGET_TYPE


@pytest.mark.asyncio
async def test_verification_pipeline_order_independent_of_entry_point_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [
            _verification_ep("zzz", "_ZzzVerificationStage"),
            _verification_ep("aaa", "_AaaVerificationStage"),
        ],
    )
    base = verification_stage_registry()
    outcome = load_verification_stage_plugins(base, discover_entry_points=True)
    kinds = tuple(registration.kind for registration in outcome.registry.registrations)
    assert kinds == (
        validate_verification_stage_kind("aaa_plugin_stage"),
        validate_verification_stage_kind("zzz_plugin_stage"),
    )
    pipeline = VerificationPipeline(registry=outcome.registry)
    result = await pipeline.verify(_candidate())
    assert list(record.stage for record in result.stage_records) == list(kinds)


# --- DS-PLUGIN-03 artifact matrix ---


def test_valid_artifact_kind_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [_artifact_ep("external_risk", "_ExternalRiskArtifactKindContribution")],
    )
    base = decision_artifact_kind_registry()
    outcome = load_decision_artifact_kind_plugins(base, discover_entry_points=True)
    assert outcome.report.registered_count == 1
    assert is_decision_artifact_kind_registered(outcome.registry, "external_risk_decision")
    assert (
        require_registered_decision_artifact_kind(outcome.registry, "external_risk_decision")
        == validate_decision_artifact_kind("external_risk_decision")
    )


def test_artifact_duplicate_kind_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _artifact_ep("artifact_a", "_DuplicateArtifactKindContribution"),
            _artifact_ep("artifact_b", "_DuplicateArtifactKindContribution"),
        ],
    )
    base = decision_artifact_kind_registry()
    outcome = load_decision_artifact_kind_plugins(base, discover_entry_points=True)
    assert outcome.report.registered_count == 1
    assert outcome.report.rejected[0].reason_code is PluginAdmissionReasonCode.PLUGIN_ID_COLLISION


def test_artifact_invalid_kind_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_artifact_ep("bad", "_WhitespaceArtifactKindContribution")])
    base = decision_artifact_kind_registry()
    outcome = load_decision_artifact_kind_plugins(base, discover_entry_points=True)
    assert outcome.report.registered_count == 0
    assert outcome.report.rejected


def test_artifact_wrong_target_type_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_artifact_ep("bad", "_NotAStrategy")])
    base = decision_artifact_kind_registry()
    outcome = load_decision_artifact_kind_plugins(base, discover_entry_points=True)
    assert outcome.registry is base
    assert outcome.report.rejected[0].reason_code is PluginAdmissionReasonCode.INVALID_TARGET_TYPE


# --- explicit opt-in ---


def test_discovery_disabled_leaves_builtin_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_strategy_ep("external_council", "_ExternalCouncilStrategy")])
    base = _builtin_strategy_registry()
    outcome = load_decision_strategy_plugins(base, discover_entry_points=False)
    assert outcome.registry is base
    assert outcome.report.registered_count == 0
    assert not is_decision_strategy_registered(outcome.registry, "external_council_variant")


def test_discovery_enabled_registers_only_valid_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _strategy_ep("external_council", "_ExternalCouncilStrategy"),
            _strategy_ep("bad", "_NotAStrategy"),
        ],
    )
    base = decision_strategy_registry()
    outcome = load_decision_strategy_plugins(base, discover_entry_points=True)
    assert outcome.report.registered_count == 1
    assert is_decision_strategy_registered(outcome.registry, "external_council_variant")


# --- platform admission ---


def test_production_qualified_compatible_package_admits_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility = _compatible_platform()
    qualification = _production_package_qualification(
        compatibility=compatibility,
        group=EP_DECISION_STRATEGIES,
        name="external_council",
    )
    _install_eps(
        monkeypatch,
        [_strategy_ep("external_council", "_ExternalCouncilStrategy", distribution=_PACKAGE_NAME)],
    )
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: compatibility,
    )
    outcome = load_decision_strategy_plugins(
        decision_strategy_registry(),
        policy=DecisionPluginLoadPolicy(
            require_production_admission=True,
            package_qualification_lookup=_lookup_for(qualification),
            platform_version=_PLATFORM_VERSION,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 1


def test_incompatible_platform_rejects_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    incompatible = check_platform_compatibility(
        PlatformCompatibility(intergrax_version=">=2,<3"),
        _PLATFORM_VERSION,
    )
    qualification = _production_package_qualification(
        compatibility=incompatible,
        group=EP_DECISION_STRATEGIES,
        name="external_council",
    )
    _install_eps(
        monkeypatch,
        [_strategy_ep("external_council", "_ExternalCouncilStrategy", distribution=_PACKAGE_NAME)],
    )
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: incompatible,
    )
    outcome = load_decision_strategy_plugins(
        decision_strategy_registry(),
        policy=DecisionPluginLoadPolicy(
            require_production_admission=True,
            package_qualification_lookup=_lookup_for(qualification),
            platform_version=_PLATFORM_VERSION,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 0
    assert outcome.report.rejected[0].reason_code is PluginAdmissionReasonCode.PRODUCTION_ADMISSION_DENIED


def test_manifest_capability_mismatch_rejects_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest_toml(
        capability=CapabilityDescriptor(
            domain=DECISION_PLUGIN_DOMAIN,
            entry_point_group=EP_DECISION_STRATEGIES,
            entry_point_name="external_council",
            capability_ids=("decision.wrong_capability",),
        ),
    )
    _install_eps(
        monkeypatch,
        [_strategy_ep("external_council", "_ExternalCouncilStrategy", distribution=_PACKAGE_NAME)],
    )
    _mock_installed_distribution(monkeypatch, manifest_toml=manifest)
    outcome = load_decision_strategy_plugins(
        decision_strategy_registry(),
        policy=DecisionPluginLoadPolicy(require_manifest_capability_binding=True),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 0
    assert outcome.report.rejected[0].reason_code is PluginAdmissionReasonCode.INVALID_TARGET_TYPE


def test_manifest_capability_binding_accepts_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest_toml(
        capability=CapabilityDescriptor(
            domain=DECISION_PLUGIN_DOMAIN,
            entry_point_group=EP_DECISION_STRATEGIES,
            entry_point_name="external_council",
            capability_ids=(DECISION_STRATEGY_CAPABILITY_ID,),
        ),
    )
    _install_eps(
        monkeypatch,
        [_strategy_ep("external_council", "_ExternalCouncilStrategy", distribution=_PACKAGE_NAME)],
    )
    _mock_installed_distribution(monkeypatch, manifest_toml=manifest)
    outcome = load_decision_strategy_plugins(
        decision_strategy_registry(),
        policy=DecisionPluginLoadPolicy(require_manifest_capability_binding=True),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 1


def test_manifest_missing_entry_point_rejects_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest_toml(
        capability=CapabilityDescriptor(
            domain=DECISION_PLUGIN_DOMAIN,
            entry_point_group=EP_DECISION_VERIFICATION_STAGES,
            entry_point_name="other",
            capability_ids=(DECISION_VERIFICATION_STAGE_CAPABILITY_ID,),
        ),
    )
    _install_eps(
        monkeypatch,
        [_strategy_ep("external_council", "_ExternalCouncilStrategy", distribution=_PACKAGE_NAME)],
    )
    _mock_installed_distribution(monkeypatch, manifest_toml=manifest)
    outcome = load_decision_strategy_plugins(
        decision_strategy_registry(),
        policy=DecisionPluginLoadPolicy(require_manifest_capability_binding=True),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 0
    assert "not declared" in outcome.report.rejected[0].reason


def test_registry_immutability_on_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_strategy_ep("bad", "_NotAStrategy")])
    base = _builtin_strategy_registry()
    outcome = load_decision_strategy_plugins(base, discover_entry_points=True)
    assert outcome.registry is base
    assert outcome.registry.registrations == base.registrations
