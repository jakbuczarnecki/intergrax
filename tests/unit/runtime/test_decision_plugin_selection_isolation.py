# © Artur Czarnecki. All rights reserved.

"""P0-A-R3 Decision plugin selection isolation tests."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.application_decision_composition import (
    ApplicationDecisionCompositionError,
    compose_application_decision,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    DecisionPluginProfile,
    DecisionProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications._shared.decision_wiring import application_decision_wiring_spec
from intergrax.contracts.decision_verification_stage import verification_stage_registry
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins import discovery as plugin_discovery
from intergrax.core.plugins.discovery import (
    EP_DECISION_ARTIFACT_KINDS,
    EP_DECISION_STRATEGIES,
    EP_DECISION_VERIFICATION_STAGES,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.core.plugins.selection_ref import PlatformPluginSelectionRef
from intergrax.contracts.decision_artifact_registry import (
    decision_artifact_kind_registry,
    is_decision_artifact_kind_registered,
)
from intergrax.runtime.decision_plugin_composition import (
    DECISION_ARTIFACT_KIND_CAPABILITY_ID,
    DECISION_PLUGIN_DOMAIN,
    DECISION_STRATEGY_CAPABILITY_ID,
    DECISION_VERIFICATION_STAGE_CAPABILITY_ID,
    DecisionPluginLoadPolicy,
    load_decision_artifact_kind_plugins,
    load_decision_strategy_plugins,
    load_verification_stage_plugins,
)
from intergrax.runtime.decision_plugin_pre_load import plan_decision_plugin_admission
from intergrax.runtime.registry.agent_registry import AgentRegistry
from echo.echo_agent import EchoAgent

pytestmark = pytest.mark.unit

_PACKAGE_A = "decision-isolation-pkg-a"
_PACKAGE_B = "decision-isolation-pkg-b"
_PACKAGE_VERSION = "1.0.0"
_COUNTER_MODULE = "testing_support.decision_plugins.preload_trust_counter"
_MALICIOUS_MODULE = "testing_support.decision_plugins.preload_trust_malicious"
_REJECT_MODULE = "testing_support.decision_plugins.preload_trust_reject_on_import"
_STRATEGY_MODULE = "testing_support.decision_plugins.preload_trust_counter"
_ARTIFACT_CONTRIBUTION = (
    "tests.unit.runtime.test_decision_plugin_composition:_ExternalRiskArtifactKindContribution"
)
_ARTIFACT_PLUGIN_ID = "external_risk_decision"


class _Dist:
    def __init__(self, name: str) -> None:
        self.name = name
        self.version = _PACKAGE_VERSION


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str, *, distribution: str) -> None:
        self.name = name
        self.value = value
        self.group = group
        self.dist = _Dist(distribution)


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _verification_ep(name: str, value: str, distribution: str) -> _EntryPoint:
    return _EntryPoint(name, value, EP_DECISION_VERIFICATION_STAGES, distribution=distribution)


def _strategy_ep(name: str, value: str, distribution: str) -> _EntryPoint:
    return _EntryPoint(name, value, EP_DECISION_STRATEGIES, distribution=distribution)


def _artifact_ep(name: str, value: str, distribution: str) -> _EntryPoint:
    return _EntryPoint(name, value, EP_DECISION_ARTIFACT_KINDS, distribution=distribution)


def _manifest_for(
    package: str,
    *,
    group: str,
    capability_id: str,
    entries: tuple[tuple[str, str], ...],
) -> str:
    blocks: list[str] = []
    for ep_name, plugin_id in entries:
        blocks.append(
            f"""
[[tool.intergrax.plugin.capabilities]]
domain = "{DECISION_PLUGIN_DOMAIN}"
entry_point_group = "{group}"
entry_point_name = "{ep_name}"
capability_ids = ["{capability_id}"]
plugin_id = "{plugin_id}"
"""
        )
    return f"""
[project]
name = "{package}"
version = "{_PACKAGE_VERSION}"

[tool.intergrax.plugin]
name = "{package}"
version = "{_PACKAGE_VERSION}"
intergrax_version = ">=0.1,<2"
{"".join(blocks)}
"""


def _mock_distribution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifests: dict[str, str],
) -> None:
    def _distribution(name: str) -> MagicMock:
        if name not in manifests:
            raise importlib.metadata.PackageNotFoundError(name)
        dist = MagicMock()
        dist.version = _PACKAGE_VERSION
        file = MagicMock()
        file.name = "pyproject.toml"
        dist.read_text.return_value = manifests[name]
        dist.files = [file]
        return dist

    monkeypatch.setattr(importlib.metadata, "distribution", _distribution)


def _ref(
    plugin_id: str,
    *,
    entry_point_name: str,
    distribution: str,
    group: str = EP_DECISION_VERIFICATION_STAGES,
) -> PlatformPluginSelectionRef:
    return PlatformPluginSelectionRef(
        plugin_id=plugin_id,
        entry_point_group=group,
        entry_point_name=entry_point_name,
        distribution=distribution,
    )


def test_requested_valid_plus_unrelated_invalid_manifest_host_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [
            _verification_ep("good", f"{_COUNTER_MODULE}:PreloadTrustCounterStage", _PACKAGE_A),
            _verification_ep("bad", f"{_REJECT_MODULE}:PreloadTrustCounterStage", _PACKAGE_B),
        ],
    )
    _mock_distribution(
        monkeypatch,
        manifests={
            _PACKAGE_A: _manifest_for(
                _PACKAGE_A,
                group=EP_DECISION_VERIFICATION_STAGES,
                capability_id=DECISION_VERIFICATION_STAGE_CAPABILITY_ID,
                entries=(("good", "preload.trust.counter"),),
            ),
            _PACKAGE_B: "not-valid-manifest",
        },
    )
    outcome = load_verification_stage_plugins(
        verification_stage_registry(),
        policy=DecisionPluginLoadPolicy(
            requested_verification_stage_plugins=(
                _ref("preload.trust.counter", entry_point_name="good", distribution=_PACKAGE_A),
            ),
            require_manifest_capability_binding=True,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.critical_bootstrap_acceptable
    assert outcome.report.registered_count == 1


def test_requested_manifest_plugin_id_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [
            _verification_ep("stage", f"{_COUNTER_MODULE}:PreloadTrustCounterStage", _PACKAGE_A),
        ],
    )
    _mock_distribution(
        monkeypatch,
        manifests={
            _PACKAGE_A: _manifest_for(
                _PACKAGE_A,
                group=EP_DECISION_VERIFICATION_STAGES,
                capability_id=DECISION_VERIFICATION_STAGE_CAPABILITY_ID,
                entries=(("stage", "manifest.other"),),
            ),
        },
    )
    plan = plan_decision_plugin_admission(
        EP_DECISION_VERIFICATION_STAGES,
        domain=DECISION_PLUGIN_DOMAIN,
        required_capability_id=DECISION_VERIFICATION_STAGE_CAPABILITY_ID,
        policy=DecisionPluginLoadPolicy(
            requested_verification_stage_plugins=(
                _ref("requested.id", entry_point_name="stage", distribution=_PACKAGE_A),
            ),
            require_manifest_capability_binding=True,
        ),
        requested_plugins=(
            _ref("requested.id", entry_point_name="stage", distribution=_PACKAGE_A),
        ),
    )
    assert not plan.admitted
    assert plan.rejected[0].reason_code is PluginAdmissionReasonCode.MANIFEST_PLUGIN_ID_MISMATCH


def test_requested_missing_locator_fails_closed() -> None:
    plan = plan_decision_plugin_admission(
        EP_DECISION_VERIFICATION_STAGES,
        domain=DECISION_PLUGIN_DOMAIN,
        required_capability_id=DECISION_VERIFICATION_STAGE_CAPABILITY_ID,
        policy=DecisionPluginLoadPolicy(
            require_manifest_capability_binding=True,
        ),
        requested_plugins=(
            _ref("missing.plugin", entry_point_name="absent", distribution=_PACKAGE_A),
        ),
    )
    assert not plan.admitted
    assert (
        plan.rejected[0].reason_code
        is PluginAdmissionReasonCode.REQUESTED_PLUGIN_LOCATOR_NOT_FOUND
    )


def test_strategy_family_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _strategy_ep("wanted", "tests.unit.runtime.test_decision_plugin_composition:_ExternalCouncilStrategy", _PACKAGE_A),
            _strategy_ep("noise", f"{_MALICIOUS_MODULE}:PreloadTrustCounterStage", _PACKAGE_B),
        ],
    )
    _mock_distribution(
        monkeypatch,
        manifests={
            _PACKAGE_A: _manifest_for(
                _PACKAGE_A,
                group=EP_DECISION_STRATEGIES,
                capability_id=DECISION_STRATEGY_CAPABILITY_ID,
                entries=(("wanted", "isolation.strategy"),),
            ),
            _PACKAGE_B: "broken",
        },
    )
    plan = plan_decision_plugin_admission(
        EP_DECISION_STRATEGIES,
        domain=DECISION_PLUGIN_DOMAIN,
        required_capability_id=DECISION_STRATEGY_CAPABILITY_ID,
        policy=DecisionPluginLoadPolicy(
            require_manifest_capability_binding=True,
        ),
        requested_plugins=(
            _ref(
                "isolation.strategy",
                entry_point_name="wanted",
                distribution=_PACKAGE_A,
                group=EP_DECISION_STRATEGIES,
            ),
        ),
    )
    assert len(plan.admitted) == 1
    assert not any(item.fail_closed for item in plan.rejected)


def test_application_composition_selection_isolation_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [
            _verification_ep("good", f"{_COUNTER_MODULE}:PreloadTrustCounterStage", _PACKAGE_A),
            _verification_ep("evil", f"{_MALICIOUS_MODULE}:PreloadTrustCounterStage", _PACKAGE_B),
        ],
    )
    _mock_distribution(
        monkeypatch,
        manifests={
            _PACKAGE_A: _manifest_for(
                _PACKAGE_A,
                group=EP_DECISION_VERIFICATION_STAGES,
                capability_id=DECISION_VERIFICATION_STAGE_CAPABILITY_ID,
                entries=(("good", "preload.trust.counter"),),
            ),
            _PACKAGE_B: "broken-manifest",
        },
    )
    registry = AgentRegistry()
    registry.register(EchoAgent())
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="isolation.compose")
    env.execution_mode = ExecutionMode.STRICT
    env.decision_profile = DecisionProfile(
        plugins=DecisionPluginProfile(
            discover_entry_points=True,
            verification_stage_plugins=[
                _ref("preload.trust.counter", entry_point_name="good", distribution=_PACKAGE_A),
            ],
        ),
    )
    composition = compose_application_decision(
        environment=env,
        contract=registry.get_contract("echo"),
        spec=application_decision_wiring_spec(),
    )
    assert "preload.trust.counter" in composition.activated_verification_stage_kinds


def test_no_external_plugins_requested_ignores_installed_manifest_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [
            _verification_ep("orphan", f"{_REJECT_MODULE}:PreloadTrustCounterStage", _PACKAGE_B),
        ],
    )
    _mock_distribution(monkeypatch, manifests={_PACKAGE_B: "broken"})
    registry = AgentRegistry()
    registry.register(EchoAgent())
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="isolation.none")
    env.execution_mode = ExecutionMode.STRICT
    env.decision_profile = DecisionProfile(
        plugins=DecisionPluginProfile(
            discover_entry_points=True,
            verification_stage_plugins=[],
        ),
    )
    compose_application_decision(
        environment=env,
        contract=registry.get_contract("echo"),
        spec=application_decision_wiring_spec(),
    )


def test_requested_invalid_manifest_composition_fails_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [
            _verification_ep("bad", f"{_REJECT_MODULE}:PreloadTrustCounterStage", _PACKAGE_A),
        ],
    )
    _mock_distribution(monkeypatch, manifests={_PACKAGE_A: "broken"})
    registry = AgentRegistry()
    registry.register(EchoAgent())
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="isolation.fail")
    env.execution_mode = ExecutionMode.STRICT
    env.decision_profile = DecisionProfile(
        plugins=DecisionPluginProfile(
            discover_entry_points=True,
            verification_stage_plugins=[
                _ref("isolation.bad", entry_point_name="bad", distribution=_PACKAGE_A),
            ],
        ),
    )
    with pytest.raises(ApplicationDecisionCompositionError):
        compose_application_decision(
            environment=env,
            contract=registry.get_contract("echo"),
            spec=application_decision_wiring_spec(),
        )


# --- P0-A-Q1 artifact family selection matrix ---


def _artifact_ref(
    plugin_id: str,
    *,
    entry_point_name: str,
    distribution: str,
) -> PlatformPluginSelectionRef:
    return PlatformPluginSelectionRef(
        plugin_id=plugin_id,
        entry_point_group=EP_DECISION_ARTIFACT_KINDS,
        entry_point_name=entry_point_name,
        distribution=distribution,
    )


def test_artifact_requested_valid_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _artifact_ep("risk", _ARTIFACT_CONTRIBUTION, _PACKAGE_A),
        ],
    )
    _mock_distribution(
        monkeypatch,
        manifests={
            _PACKAGE_A: _manifest_for(
                _PACKAGE_A,
                group=EP_DECISION_ARTIFACT_KINDS,
                capability_id=DECISION_ARTIFACT_KIND_CAPABILITY_ID,
                entries=(("risk", _ARTIFACT_PLUGIN_ID),),
            ),
        },
    )
    outcome = load_decision_artifact_kind_plugins(
        decision_artifact_kind_registry(),
        policy=DecisionPluginLoadPolicy(
            requested_artifact_plugins=(
                _artifact_ref(_ARTIFACT_PLUGIN_ID, entry_point_name="risk", distribution=_PACKAGE_A),
            ),
            require_manifest_capability_binding=True,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 1
    assert is_decision_artifact_kind_registered(outcome.registry, _ARTIFACT_PLUGIN_ID)


def test_artifact_requested_valid_plus_unrelated_invalid_manifest_host_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [
            _artifact_ep("risk", _ARTIFACT_CONTRIBUTION, _PACKAGE_A),
            _artifact_ep("noise", _ARTIFACT_CONTRIBUTION, _PACKAGE_B),
        ],
    )
    _mock_distribution(
        monkeypatch,
        manifests={
            _PACKAGE_A: _manifest_for(
                _PACKAGE_A,
                group=EP_DECISION_ARTIFACT_KINDS,
                capability_id=DECISION_ARTIFACT_KIND_CAPABILITY_ID,
                entries=(("risk", _ARTIFACT_PLUGIN_ID),),
            ),
            _PACKAGE_B: "not-valid-manifest",
        },
    )
    outcome = load_decision_artifact_kind_plugins(
        decision_artifact_kind_registry(),
        policy=DecisionPluginLoadPolicy(
            requested_artifact_plugins=(
                _artifact_ref(_ARTIFACT_PLUGIN_ID, entry_point_name="risk", distribution=_PACKAGE_A),
            ),
            require_manifest_capability_binding=True,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.critical_bootstrap_acceptable
    assert outcome.report.registered_count == 1


def test_artifact_malicious_unselected_plugin_never_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _artifact_ep("risk", _ARTIFACT_CONTRIBUTION, _PACKAGE_A),
            _artifact_ep("evil", f"{_MALICIOUS_MODULE}:PreloadTrustCounterStage", _PACKAGE_B),
        ],
    )
    _mock_distribution(
        monkeypatch,
        manifests={
            _PACKAGE_A: _manifest_for(
                _PACKAGE_A,
                group=EP_DECISION_ARTIFACT_KINDS,
                capability_id=DECISION_ARTIFACT_KIND_CAPABILITY_ID,
                entries=(("risk", _ARTIFACT_PLUGIN_ID),),
            ),
            _PACKAGE_B: _manifest_for(
                _PACKAGE_B,
                group=EP_DECISION_ARTIFACT_KINDS,
                capability_id=DECISION_ARTIFACT_KIND_CAPABILITY_ID,
                entries=(("evil", "preload.trust.evil"),),
            ),
        },
    )
    outcome = load_decision_artifact_kind_plugins(
        decision_artifact_kind_registry(),
        policy=DecisionPluginLoadPolicy(
            requested_artifact_plugins=(
                _artifact_ref(_ARTIFACT_PLUGIN_ID, entry_point_name="risk", distribution=_PACKAGE_A),
            ),
            require_manifest_capability_binding=True,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.critical_bootstrap_acceptable
    assert outcome.report.registered_count == 1


def test_artifact_requested_missing_locator_fails_closed() -> None:
    plan = plan_decision_plugin_admission(
        EP_DECISION_ARTIFACT_KINDS,
        domain=DECISION_PLUGIN_DOMAIN,
        required_capability_id=DECISION_ARTIFACT_KIND_CAPABILITY_ID,
        policy=DecisionPluginLoadPolicy(require_manifest_capability_binding=True),
        requested_plugins=(
            _artifact_ref("missing.artifact", entry_point_name="absent", distribution=_PACKAGE_A),
        ),
    )
    assert not plan.admitted
    assert (
        plan.rejected[0].reason_code
        is PluginAdmissionReasonCode.REQUESTED_PLUGIN_LOCATOR_NOT_FOUND
    )


def test_artifact_requested_invalid_manifest_fails_closed_with_zero_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = _track_entry_point_loads(monkeypatch)
    _install_eps(
        monkeypatch,
        [
            _artifact_ep("risk", _ARTIFACT_CONTRIBUTION, _PACKAGE_A),
        ],
    )
    _mock_distribution(monkeypatch, manifests={_PACKAGE_A: "broken-manifest"})
    outcome = load_decision_artifact_kind_plugins(
        decision_artifact_kind_registry(),
        policy=DecisionPluginLoadPolicy(
            requested_artifact_plugins=(
                _artifact_ref(_ARTIFACT_PLUGIN_ID, entry_point_name="risk", distribution=_PACKAGE_A),
            ),
            require_manifest_capability_binding=True,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 0
    assert outcome.report.rejected[0].reason_code is PluginAdmissionReasonCode.MANIFEST_INVALID
    assert len(loads) == 0


def test_artifact_requested_manifest_plugin_id_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [
            _artifact_ep("risk", _ARTIFACT_CONTRIBUTION, _PACKAGE_A),
        ],
    )
    _mock_distribution(
        monkeypatch,
        manifests={
            _PACKAGE_A: _manifest_for(
                _PACKAGE_A,
                group=EP_DECISION_ARTIFACT_KINDS,
                capability_id=DECISION_ARTIFACT_KIND_CAPABILITY_ID,
                entries=(("risk", "manifest.other"),),
            ),
        },
    )
    plan = plan_decision_plugin_admission(
        EP_DECISION_ARTIFACT_KINDS,
        domain=DECISION_PLUGIN_DOMAIN,
        required_capability_id=DECISION_ARTIFACT_KIND_CAPABILITY_ID,
        policy=DecisionPluginLoadPolicy(
            requested_artifact_plugins=(
                _artifact_ref("requested.artifact", entry_point_name="risk", distribution=_PACKAGE_A),
            ),
            require_manifest_capability_binding=True,
        ),
        requested_plugins=(
            _artifact_ref("requested.artifact", entry_point_name="risk", distribution=_PACKAGE_A),
        ),
    )
    assert not plan.admitted
    assert plan.rejected[0].reason_code is PluginAdmissionReasonCode.MANIFEST_PLUGIN_ID_MISMATCH


def test_artifact_runtime_kind_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _artifact_ep("risk", _ARTIFACT_CONTRIBUTION, _PACKAGE_A),
        ],
    )
    _mock_distribution(
        monkeypatch,
        manifests={
            _PACKAGE_A: _manifest_for(
                _PACKAGE_A,
                group=EP_DECISION_ARTIFACT_KINDS,
                capability_id=DECISION_ARTIFACT_KIND_CAPABILITY_ID,
                entries=(("risk", "artifact.runtime.mismatch"),),
            ),
        },
    )
    outcome = load_decision_artifact_kind_plugins(
        decision_artifact_kind_registry(),
        policy=DecisionPluginLoadPolicy(
            requested_artifact_plugins=(
                _artifact_ref(
                    "artifact.runtime.mismatch",
                    entry_point_name="risk",
                    distribution=_PACKAGE_A,
                ),
            ),
            require_manifest_capability_binding=True,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 0
    assert outcome.report.rejected[0].reason_code is PluginAdmissionReasonCode.PLUGIN_IDENTITY_MISMATCH


def test_artifact_duplicate_plugin_id_blocked_before_load(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = _track_entry_point_loads(monkeypatch)
    _install_eps(
        monkeypatch,
        [
            _artifact_ep("ep_one", _ARTIFACT_CONTRIBUTION, _PACKAGE_A),
            _artifact_ep("ep_two", _ARTIFACT_CONTRIBUTION, _PACKAGE_A),
        ],
    )
    _mock_distribution(
        monkeypatch,
        manifests={
            _PACKAGE_A: _manifest_for(
                _PACKAGE_A,
                group=EP_DECISION_ARTIFACT_KINDS,
                capability_id=DECISION_ARTIFACT_KIND_CAPABILITY_ID,
                entries=(
                    ("ep_one", "artifact.duplicate"),
                    ("ep_two", "artifact.duplicate"),
                ),
            ),
        },
    )
    outcome = load_decision_artifact_kind_plugins(
        decision_artifact_kind_registry(),
        policy=DecisionPluginLoadPolicy(
            requested_artifact_plugins=(
                _artifact_ref("artifact.duplicate", entry_point_name="ep_one", distribution=_PACKAGE_A),
                _artifact_ref("artifact.duplicate", entry_point_name="ep_two", distribution=_PACKAGE_A),
            ),
            require_manifest_capability_binding=True,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 0
    assert len(loads) == 0
    assert (
        outcome.report.rejected[-1].reason_code
        is PluginAdmissionReasonCode.METADATA_PLUGIN_ID_COLLISION
    )


def _combined_manifest(
    package: str,
    *,
    capability_blocks: str,
) -> str:
    return f"""
[project]
name = "{package}"
version = "{_PACKAGE_VERSION}"

[tool.intergrax.plugin]
name = "{package}"
version = "{_PACKAGE_VERSION}"
intergrax_version = ">=0.1,<2"
{capability_blocks}
"""


def _capability_sections_from_manifest(manifest: str) -> str:
    parts = manifest.split("[[tool.intergrax.plugin.capabilities]]")
    return "".join(
        "[[tool.intergrax.plugin.capabilities]]" + part for part in parts[1:]
    )


def _track_entry_point_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    loads: list[str] = []
    original = plugin_discovery.load_entry_point_value

    def _counting_load(value: str) -> object:
        loads.append(value)
        return original(value)

    monkeypatch.setattr(plugin_discovery, "load_entry_point_value", _counting_load)
    return loads


def test_artifact_application_composition_activates_requested_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verification_block = _manifest_for(
        _PACKAGE_A,
        group=EP_DECISION_VERIFICATION_STAGES,
        capability_id=DECISION_VERIFICATION_STAGE_CAPABILITY_ID,
        entries=(("good", "preload.trust.counter"),),
    )
    artifact_block = _manifest_for(
        _PACKAGE_A,
        group=EP_DECISION_ARTIFACT_KINDS,
        capability_id=DECISION_ARTIFACT_KIND_CAPABILITY_ID,
        entries=(("risk", _ARTIFACT_PLUGIN_ID),),
    )
    capability_blocks = _capability_sections_from_manifest(verification_block)
    capability_blocks += _capability_sections_from_manifest(artifact_block)
    _install_eps(
        monkeypatch,
        [
            _verification_ep("good", f"{_COUNTER_MODULE}:PreloadTrustCounterStage", _PACKAGE_A),
            _artifact_ep("risk", _ARTIFACT_CONTRIBUTION, _PACKAGE_A),
        ],
    )
    _mock_distribution(
        monkeypatch,
        manifests={
            _PACKAGE_A: _combined_manifest(_PACKAGE_A, capability_blocks=capability_blocks),
        },
    )
    registry = AgentRegistry()
    registry.register(EchoAgent())
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="isolation.artifact.compose")
    env.execution_mode = ExecutionMode.STRICT
    env.decision_profile = DecisionProfile(
        plugins=DecisionPluginProfile(
            discover_entry_points=True,
            verification_stage_plugins=[
                _ref("preload.trust.counter", entry_point_name="good", distribution=_PACKAGE_A),
            ],
            artifact_plugins=[
                _artifact_ref(_ARTIFACT_PLUGIN_ID, entry_point_name="risk", distribution=_PACKAGE_A),
            ],
        ),
    )
    composition = compose_application_decision(
        environment=env,
        contract=registry.get_contract("echo"),
        spec=application_decision_wiring_spec(),
    )
    assert "preload.trust.counter" in composition.activated_verification_stage_kinds
    assert _ARTIFACT_PLUGIN_ID in composition.activated_artifact_kinds
