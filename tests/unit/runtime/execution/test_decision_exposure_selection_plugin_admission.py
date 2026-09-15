# © Artur Czarnecki. All rights reserved.

"""I1-A-R1 exposure selection plugin admission and post-selection trust tests."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.application_decision_composition import (
    compose_application_decision_exposure_selection,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    DecisionPluginProfile,
    DecisionProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.decision_authoritative_exposure import DecisionEvaluationScope
from intergrax.contracts.decision_authoritative_exposure import ExposureAccepted
from intergrax.contracts.decision_exposure_selection import (
    DecisionExposureCandidate,
    DecisionExposurePublicationPolicy,
    DecisionExposureSelectionDecision,
    DecisionExposureSelectionFailure,
    DecisionExposureSelectionFailureCode,
    DecisionExposureSelectionStrategy,
    DecisionExposureSelectionSuccessReason,
    HostPublicationClass,
)
from intergrax.contracts.execution_identity import mint_attempt_id
from intergrax.core.plugins import discovery as plugin_discovery
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import (
    EP_DECISION_EXPOSURE_SELECTION_STRATEGIES,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.core.plugins.selection_ref import PlatformPluginSelectionRef
from intergrax.runtime.decision_plugin_composition import (
    DECISION_EXPOSURE_SELECTION_STRATEGY_CAPABILITY_ID,
    DECISION_PLUGIN_DOMAIN,
    DecisionPluginLoadPolicy,
    load_decision_exposure_selection_strategy_plugin,
)
from intergrax.runtime.execution.decision_exposure_selection_composition import (
    DecisionExposureSelectionCompositionError,
    compose_decision_exposure_selection,
)
from intergrax.runtime.execution.decision_exposure_selection_validation import (
    run_validated_decision_exposure_selection,
)
from intergrax.runtime.execution.host_terminal_decision_exposure_selector import (
    HOST_TERMINAL_DECISION_EXPOSURE_SELECTOR_ID,
    HostTerminalDecisionExposureSelector,
    default_decision_exposure_selection_strategy,
)
from tests.unit.runtime.execution.test_decision_exposure_selection import (
    _accepted,
    _candidate,
    _graph_policy,
)

pytestmark = pytest.mark.unit

_PACKAGE = "exposure-selection-trust-pkg"
_PACKAGE_VERSION = "1.0.0"
_DELEGATE_MODULE = "testing_support.decision_plugins.exposure_selection_trust"
_MALICIOUS_MODULE = "testing_support.decision_plugins.preload_trust_malicious"
_WRONG_ID_MODULE = f"{_DELEGATE_MODULE}:ExposureSelectionTrustWrongRuntimeId"
_DELEGATE_TARGET = f"{_DELEGATE_MODULE}:ExposureSelectionTrustDelegate"
_PLUGIN_ID = "preload.exposure.trust.delegate"
_WRONG_PLUGIN_ID = "preload.exposure.trust.wrong_id"


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


def _exposure_ep(name: str, value: str, distribution: str = _PACKAGE) -> _EntryPoint:
    return _EntryPoint(name, value, EP_DECISION_EXPOSURE_SELECTION_STRATEGIES, distribution=distribution)


def _selection_ref(
    plugin_id: str,
    *,
    entry_point_name: str = "external_selector",
) -> PlatformPluginSelectionRef:
    return PlatformPluginSelectionRef(
        plugin_id=plugin_id,
        entry_point_group=EP_DECISION_EXPOSURE_SELECTION_STRATEGIES,
        entry_point_name=entry_point_name,
        distribution=_PACKAGE,
    )


def _manifest_for(
    *,
    entries: tuple[tuple[str, str, str], ...],
    capability_id: str = DECISION_EXPOSURE_SELECTION_STRATEGY_CAPABILITY_ID,
) -> str:
    blocks: list[str] = []
    for ep_name, plugin_id, capability in entries:
        blocks.append(
            f"""
[[tool.intergrax.plugin.capabilities]]
domain = "{DECISION_PLUGIN_DOMAIN}"
entry_point_group = "{EP_DECISION_EXPOSURE_SELECTION_STRATEGIES}"
entry_point_name = "{ep_name}"
capability_ids = ["{capability}"]
plugin_id = "{plugin_id}"
"""
        )
    return f"""
[project]
name = "{_PACKAGE}"
version = "{_PACKAGE_VERSION}"

[tool.intergrax.plugin]
name = "{_PACKAGE}"
version = "{_PACKAGE_VERSION}"
intergrax_version = ">=0.1,<2"
{"".join(blocks)}
"""


def _mock_distribution(monkeypatch: pytest.MonkeyPatch, manifest_toml: str) -> None:
    dist = MagicMock()
    dist.version = _PACKAGE_VERSION
    file = MagicMock()
    file.name = "pyproject.toml"
    dist.read_text.return_value = manifest_toml
    dist.files = [file]
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: dist if name == _PACKAGE else (_ for _ in ()).throw(
            importlib.metadata.PackageNotFoundError(name),
        ),
    )


def _track_entry_point_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    loads: list[str] = []
    original = plugin_discovery.load_entry_point_value

    def _counting_load(value: str) -> object:
        loads.append(value)
        return original(value)

    monkeypatch.setattr(plugin_discovery, "load_entry_point_value", _counting_load)
    return loads


def _env_with_selection(
    ref: PlatformPluginSelectionRef | None,
    *,
    mode: ExecutionMode = ExecutionMode.BALANCED,
) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="exposure.selection.compose")
    env.execution_mode = mode
    env.decision_profile = DecisionProfile(
        plugins=DecisionPluginProfile(
            discover_entry_points=True,
            exposure_selection_strategy_plugin=ref,
        ),
    )
    return env


def test_builtin_default_without_external_plugin() -> None:
    composition = compose_decision_exposure_selection(selection_ref=None)
    assert composition.activated_plugin_id is None
    assert composition.strategy.strategy_id == HOST_TERMINAL_DECISION_EXPOSURE_SELECTOR_ID
    assert composition.report.critical_bootstrap_acceptable


def test_unrelated_malicious_plugin_not_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = _track_entry_point_loads(monkeypatch)
    _install_eps(
        monkeypatch,
        [_exposure_ep("evil", f"{_MALICIOUS_MODULE}:IMPORT_COUNTER")],
    )
    compose_decision_exposure_selection(selection_ref=None)
    assert loads == []


def test_requested_valid_external_selector_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = _selection_ref(_PLUGIN_ID)
    _install_eps(monkeypatch, [_exposure_ep("external_selector", _DELEGATE_TARGET)])
    _mock_distribution(
        monkeypatch,
        _manifest_for(entries=(("external_selector", _PLUGIN_ID, DECISION_EXPOSURE_SELECTION_STRATEGY_CAPABILITY_ID),)),
    )
    policy = DecisionPluginLoadPolicy(
        require_manifest_capability_binding=True,
        requested_exposure_selection_strategy_plugins=(ref,),
    )
    composition = compose_decision_exposure_selection(policy=policy, selection_ref=ref)
    assert composition.activated_plugin_id == _PLUGIN_ID
    assert isinstance(composition.strategy, DecisionExposureSelectionStrategy)


def test_requested_missing_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = _selection_ref("missing.plugin")
    _install_eps(monkeypatch, [])
    with pytest.raises(DecisionExposureSelectionCompositionError):
        compose_decision_exposure_selection(
            policy=DecisionPluginLoadPolicy(
                requested_exposure_selection_strategy_plugins=(ref,),
            ),
            selection_ref=ref,
        )


def test_invalid_manifest_fail_before_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = _track_entry_point_loads(monkeypatch)
    ref = _selection_ref(_PLUGIN_ID)
    _install_eps(monkeypatch, [_exposure_ep("external_selector", _DELEGATE_TARGET)])
    _mock_distribution(monkeypatch, "[project]\nname = \"broken\"\n")
    with pytest.raises(DecisionExposureSelectionCompositionError):
        compose_decision_exposure_selection(
            policy=DecisionPluginLoadPolicy(
                require_manifest_capability_binding=True,
                requested_exposure_selection_strategy_plugins=(ref,),
            ),
            selection_ref=ref,
        )
    assert loads == []


def test_manifest_plugin_id_mismatch_fail_before_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = _track_entry_point_loads(monkeypatch)
    ref = _selection_ref("expected.id")
    _install_eps(monkeypatch, [_exposure_ep("external_selector", _DELEGATE_TARGET)])
    _mock_distribution(
        monkeypatch,
        _manifest_for(entries=(("external_selector", "manifest.other", DECISION_EXPOSURE_SELECTION_STRATEGY_CAPABILITY_ID),)),
    )
    with pytest.raises(DecisionExposureSelectionCompositionError):
        compose_decision_exposure_selection(
            policy=DecisionPluginLoadPolicy(
                require_manifest_capability_binding=True,
                requested_exposure_selection_strategy_plugins=(ref,),
            ),
            selection_ref=ref,
        )
    assert loads == []


def test_capability_mismatch_fail_before_load(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = _track_entry_point_loads(monkeypatch)
    ref = _selection_ref(_PLUGIN_ID)
    _install_eps(monkeypatch, [_exposure_ep("external_selector", _DELEGATE_TARGET)])
    _mock_distribution(
        monkeypatch,
        _manifest_for(
            entries=(("external_selector", _PLUGIN_ID, "decision.strategy"),),
        ),
    )
    with pytest.raises(DecisionExposureSelectionCompositionError):
        compose_decision_exposure_selection(
            policy=DecisionPluginLoadPolicy(
                require_manifest_capability_binding=True,
                requested_exposure_selection_strategy_plugins=(ref,),
            ),
            selection_ref=ref,
        )
    assert loads == []


def test_runtime_strategy_id_mismatch_after_load(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = _selection_ref(_PLUGIN_ID)
    _install_eps(monkeypatch, [_exposure_ep("external_selector", _WRONG_ID_MODULE)])
    _mock_distribution(
        monkeypatch,
        _manifest_for(entries=(("external_selector", _PLUGIN_ID, DECISION_EXPOSURE_SELECTION_STRATEGY_CAPABILITY_ID),)),
    )
    outcome = load_decision_exposure_selection_strategy_plugin(
        policy=DecisionPluginLoadPolicy(
            require_manifest_capability_binding=True,
            requested_exposure_selection_strategy_plugins=(ref,),
        ),
        selection_ref=ref,
    )
    assert outcome.strategy is None
    assert any(
        item.reason_code is PluginAdmissionReasonCode.PLUGIN_IDENTITY_MISMATCH
        for item in outcome.report.rejected
    )


def test_strict_requires_manifest_for_external(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = _selection_ref(_PLUGIN_ID)
    _install_eps(monkeypatch, [_exposure_ep("external_selector", _DELEGATE_TARGET)])
    env = _env_with_selection(ref, mode=ExecutionMode.STRICT)
    with pytest.raises(DecisionExposureSelectionCompositionError):
        compose_application_decision_exposure_selection(env)


class _MintingStrategy:
    @property
    def strategy_id(self) -> str:
        return "mint.attempt"

    def select(
        self,
        policy: DecisionExposurePublicationPolicy,
        candidates: tuple[DecisionExposureCandidate[object], ...],
    ) -> DecisionExposureSelectionDecision[object]:
        exposure = _accepted("fabricated")
        return DecisionExposureSelectionDecision(
            selected=exposure,
            reason_code=DecisionExposureSelectionSuccessReason.HOST_TERMINAL_SINGLE_ELIGIBLE_CANDIDATE,
            considered_candidates=len(candidates),
        )


def test_malicious_strategy_cannot_mint_exposure() -> None:
    attempt = mint_attempt_id()
    exposure_a = _accepted("a")
    candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure_a,
            ordinal=0,
            attempt_id=attempt,
        ),
    )
    outcome = run_validated_decision_exposure_selection(
        _MintingStrategy(),
        _graph_policy(),
        candidates,
    )
    assert isinstance(outcome, DecisionExposureSelectionFailure)
    assert outcome.reason_code is DecisionExposureSelectionFailureCode.INVARIANT_VIOLATION


class _IntermediatePicker:
    @property
    def strategy_id(self) -> str:
        return "pick.intermediate"

    def select(
        self,
        policy: DecisionExposurePublicationPolicy,
        candidates: tuple[DecisionExposureCandidate[object], ...],
    ) -> DecisionExposureSelectionDecision[object]:
        intermediate = next(
            c
            for c in candidates
            if c.host_publication_class is HostPublicationClass.INTERMEDIATE
        )
        return DecisionExposureSelectionDecision(
            selected=intermediate.exposure,
            reason_code=DecisionExposureSelectionSuccessReason.HOST_TERMINAL_SINGLE_ELIGIBLE_CANDIDATE,
            considered_candidates=len(candidates),
        )


def test_strategy_cannot_select_intermediate_candidate() -> None:
    attempt = mint_attempt_id()
    graph = _accepted("graph")
    candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.UAEP_STEP,
            publication_class=HostPublicationClass.INTERMEDIATE,
            exposure=graph,
            ordinal=0,
            attempt_id=attempt,
        ),
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=graph,
            ordinal=1,
            attempt_id=attempt,
        ),
    )
    outcome = run_validated_decision_exposure_selection(
        _IntermediatePicker(),
        _graph_policy(),
        candidates,
    )
    assert isinstance(outcome, DecisionExposureSelectionFailure)


class _UaepScopePicker:
    @property
    def strategy_id(self) -> str:
        return "pick.uaep"

    def select(
        self,
        policy: DecisionExposurePublicationPolicy,
        candidates: tuple[DecisionExposureCandidate[object], ...],
    ) -> DecisionExposureSelectionDecision[object]:
        uaep = next(
            c for c in candidates if c.evaluation_scope is DecisionEvaluationScope.UAEP_STEP
        )
        return DecisionExposureSelectionDecision(
            selected=uaep.exposure,
            reason_code=DecisionExposureSelectionSuccessReason.HOST_TERMINAL_SINGLE_ELIGIBLE_CANDIDATE,
            considered_candidates=len(candidates),
        )


def test_strategy_cannot_bypass_graph_final_policy() -> None:
    attempt = mint_attempt_id()
    graph = _accepted("graph")
    uaep_exposure = ExposureAccepted(
        scope=DecisionEvaluationScope.UAEP_STEP,
        accepted=graph.accepted,
    )
    candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=graph,
            ordinal=0,
            attempt_id=attempt,
        ),
        _candidate(
            evaluation_scope=DecisionEvaluationScope.UAEP_STEP,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=uaep_exposure,
            ordinal=1,
            attempt_id=attempt,
        ),
    )
    outcome = run_validated_decision_exposure_selection(
        _UaepScopePicker(),
        _graph_policy(),
        candidates,
    )
    assert isinstance(outcome, DecisionExposureSelectionFailure)


def test_default_selector_uses_same_post_validation_path() -> None:
    attempt = mint_attempt_id()
    exposure = _accepted("ok")
    candidates = (
        _candidate(
            evaluation_scope=DecisionEvaluationScope.GRAPH_FINAL,
            publication_class=HostPublicationClass.HOST_TERMINAL_CANDIDATE,
            exposure=exposure,
            ordinal=0,
            attempt_id=attempt,
        ),
    )
    outcome = run_validated_decision_exposure_selection(
        default_decision_exposure_selection_strategy(),
        _graph_policy(),
        candidates,
    )
    assert isinstance(
        outcome,
        DecisionExposureSelectionDecision,
    )
