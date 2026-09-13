# © Artur Czarnecki. All rights reserved.

"""P0-A-R2 pre-load Decision plugin admission trust boundary tests."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock

import pytest

from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins import discovery as plugin_discovery
from intergrax.core.plugins.discovery import (
    EP_DECISION_VERIFICATION_STAGES,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.core.plugins.package_contract import CapabilityDescriptor
from intergrax.runtime.decision_plugin_composition import (
    DECISION_PLUGIN_DOMAIN,
    DECISION_VERIFICATION_STAGE_CAPABILITY_ID,
    DecisionPluginLoadPolicy,
    load_verification_stage_plugins,
)
from intergrax.contracts.decision_verification_stage import verification_stage_registry
pytestmark = pytest.mark.unit

_PACKAGE_NAME = "preload-trust-pkg"
_PACKAGE_VERSION = "1.0.0"
_COUNTER_MODULE = "testing_support.decision_plugins.preload_trust_counter"
_MALICIOUS_MODULE = "testing_support.decision_plugins.preload_trust_malicious"
_REJECT_MODULE = "testing_support.decision_plugins.preload_trust_reject_on_import"


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


def _verification_ep(name: str, value: str) -> _EntryPoint:
    return _EntryPoint(name, value, EP_DECISION_VERIFICATION_STAGES, distribution=_PACKAGE_NAME)


def _dual_capability_manifest(
    *,
    entries: tuple[tuple[str, str], ...],
) -> str:
    blocks: list[str] = []
    for ep_name, plugin_id in entries:
        blocks.append(
            f"""
[[tool.intergrax.plugin.capabilities]]
domain = "{DECISION_PLUGIN_DOMAIN}"
entry_point_group = "{EP_DECISION_VERIFICATION_STAGES}"
entry_point_name = "{ep_name}"
capability_ids = ["{DECISION_VERIFICATION_STAGE_CAPABILITY_ID}"]
plugin_id = "{plugin_id}"
"""
        )
    return f"""
[project]
name = "{_PACKAGE_NAME}"
version = "{_PACKAGE_VERSION}"

[tool.intergrax.plugin]
name = "{_PACKAGE_NAME}"
version = "{_PACKAGE_VERSION}"
intergrax_version = ">=0.1,<2"
{"".join(blocks)}
"""


def _track_entry_point_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    loads: list[str] = []
    original = plugin_discovery.load_entry_point_value

    def _counting_load(value: str) -> object:
        loads.append(value)
        return original(value)

    monkeypatch.setattr(plugin_discovery, "load_entry_point_value", _counting_load)
    return loads


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
        lambda name: dist if name == _PACKAGE_NAME else (_ for _ in ()).throw(
            importlib.metadata.PackageNotFoundError(name),
        ),
    )


def test_unselected_plugin_never_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = _track_entry_point_loads(monkeypatch)
    _install_eps(
        monkeypatch,
        [
            _verification_ep("stage_a", f"{_COUNTER_MODULE}:PreloadTrustCounterStage"),
            _verification_ep("stage_b", f"{_COUNTER_MODULE}:PreloadTrustCounterStage"),
        ],
    )
    manifest = _dual_capability_manifest(
        entries=(
            ("stage_a", "preload.trust.a"),
            ("stage_b", "preload.trust.counter"),
        ),
    )
    _mock_distribution(monkeypatch, manifest)
    outcome = load_verification_stage_plugins(
        verification_stage_registry(),
        policy=DecisionPluginLoadPolicy(
            allowed_verification_stage_kinds=frozenset({"preload.trust.counter"}),
            require_manifest_capability_binding=True,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 1
    assert len(loads) == 1
    not_selected = [
        item
        for item in outcome.report.rejected
        if item.reason_code is PluginAdmissionReasonCode.PLUGIN_NOT_SELECTED
    ]
    assert len(not_selected) == 1


def test_malicious_unselected_plugin_does_not_import(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _verification_ep("safe", f"{_COUNTER_MODULE}:PreloadTrustCounterStage"),
            _verification_ep("evil", f"{_MALICIOUS_MODULE}:PreloadTrustCounterStage"),
        ],
    )
    manifest = _dual_capability_manifest(
        entries=(
            ("safe", "preload.trust.counter"),
            ("evil", "preload.trust.evil"),
        ),
    )
    _mock_distribution(monkeypatch, manifest)
    outcome = load_verification_stage_plugins(
        verification_stage_registry(),
        policy=DecisionPluginLoadPolicy(
            allowed_verification_stage_kinds=frozenset({"preload.trust.counter"}),
            require_manifest_capability_binding=True,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.critical_bootstrap_acceptable
    assert outcome.report.registered_count == 1


def test_selected_invalid_manifest_never_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _verification_ep(
                "bad",
                f"{_REJECT_MODULE}:PreloadTrustCounterStage",
            ),
        ],
    )
    _mock_distribution(monkeypatch, "not-valid-toml-manifest")
    outcome = load_verification_stage_plugins(
        verification_stage_registry(),
        policy=DecisionPluginLoadPolicy(
            allowed_verification_stage_kinds=frozenset({"preload.trust.counter"}),
            require_manifest_capability_binding=True,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 0
    assert outcome.report.rejected[0].reason_code is PluginAdmissionReasonCode.MANIFEST_INVALID


def test_three_plugins_select_middle_only(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = _track_entry_point_loads(monkeypatch)
    _install_eps(
        monkeypatch,
        [
            _verification_ep("ep_a", f"{_COUNTER_MODULE}:PreloadTrustCounterStage"),
            _verification_ep("ep_b", f"{_COUNTER_MODULE}:PreloadTrustCounterStage"),
            _verification_ep("ep_c", f"{_COUNTER_MODULE}:PreloadTrustCounterStage"),
        ],
    )
    manifest = _dual_capability_manifest(
        entries=(
            ("ep_a", "preload.trust.a"),
            ("ep_b", "preload.trust.counter"),
            ("ep_c", "preload.trust.c"),
        ),
    )
    _mock_distribution(monkeypatch, manifest)
    outcome = load_verification_stage_plugins(
        verification_stage_registry(),
        policy=DecisionPluginLoadPolicy(
            allowed_verification_stage_kinds=frozenset({"preload.trust.counter"}),
            require_manifest_capability_binding=True,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 1
    assert len(loads) == 1
    assert len([r for r in outcome.report.rejected if not r.fail_closed]) == 2


def test_runtime_kind_manifest_plugin_id_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [
            _verification_ep("stage", f"{_COUNTER_MODULE}:PreloadTrustCounterStage"),
        ],
    )
    manifest = _dual_capability_manifest(entries=(("stage", "preload.trust.wrong_id"),))
    _mock_distribution(monkeypatch, manifest)
    outcome = load_verification_stage_plugins(
        verification_stage_registry(),
        policy=DecisionPluginLoadPolicy(
            allowed_verification_stage_kinds=frozenset({"preload.trust.wrong_id"}),
            require_manifest_capability_binding=True,
        ),
        discover_entry_points=True,
    )
    assert outcome.report.registered_count == 0
    assert outcome.report.rejected[0].reason_code is PluginAdmissionReasonCode.PLUGIN_IDENTITY_MISMATCH


def test_duplicate_manifest_plugin_id_rejected_before_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = _track_entry_point_loads(monkeypatch)
    _install_eps(
        monkeypatch,
        [
            _verification_ep("ep_one", f"{_COUNTER_MODULE}:PreloadTrustCounterStage"),
            _verification_ep("ep_two", f"{_COUNTER_MODULE}:PreloadTrustCounterStage"),
        ],
    )
    manifest = _dual_capability_manifest(
        entries=(
            ("ep_one", "preload.trust.duplicate"),
            ("ep_two", "preload.trust.duplicate"),
        ),
    )
    _mock_distribution(monkeypatch, manifest)
    outcome = load_verification_stage_plugins(
        verification_stage_registry(),
        policy=DecisionPluginLoadPolicy(
            allowed_verification_stage_kinds=frozenset({"preload.trust.duplicate"}),
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
