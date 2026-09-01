# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.metadata
from dataclasses import fields

import pytest

from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import reset_entry_point_spec_cache_for_tests
from intergrax.core.plugins.errors import PluginConflictError, PluginLoadError
from intergrax.core.security_bootstrap import bootstrap_security_providers
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.security.defense_plugin import SecurityFailMode, SecurityInspectionResult
from intergrax.runtime.security.defense_plugin_loader import (
    LEGACY_UNCONDITIONAL_OVERRIDE_POLICY,
    SecurityDefenseAdmissionPolicy,
    load_security_defense_plugin_report,
)
from intergrax.runtime.security.defense_registry import (
    get_security_defense_plugin,
    register_security_defense_plugin,
    reset_security_defense_registry_for_tests,
)

pytestmark = pytest.mark.unit

_GROUP = "intergrax.security_defenses"


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


class _AlphaDefense:
    plugin_id = "lab.alpha"
    version = "1.0.0"
    hook_points = frozenset()
    priority = 0
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: object) -> SecurityInspectionResult:
        return SecurityInspectionResult(plugin_id=self.plugin_id, hook_point=str(point))


class _BetaDefense:
    plugin_id = "lab.beta"
    version = "1.0.0"
    hook_points = frozenset()
    priority = 0
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: object) -> SecurityInspectionResult:
        return SecurityInspectionResult(plugin_id=self.plugin_id, hook_point=str(point))


class _GammaDefense:
    plugin_id = "lab.alpha"
    version = "1.0.0"
    hook_points = frozenset()
    priority = 0
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: object) -> SecurityInspectionResult:
        return SecurityInspectionResult(plugin_id=self.plugin_id, hook_point=str(point))


class _ShippedIdDefense:
    plugin_id = "harness.strict_injection"
    version = "9.0.0"
    hook_points = frozenset()
    priority = 0
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: object) -> SecurityInspectionResult:
        return SecurityInspectionResult(plugin_id=self.plugin_id, hook_point=str(point))


class _NotADefense:
    pass


class _PreexistingDefense:
    plugin_id = "lab.preexisting"
    version = "1.0.0"
    hook_points = frozenset()
    priority = 0
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: object) -> SecurityInspectionResult:
        return SecurityInspectionResult(plugin_id=self.plugin_id, hook_point=str(point))


class _CollidingPreexistingDefense:
    plugin_id = "lab.preexisting"
    version = "2.0.0"
    hook_points = frozenset()
    priority = 0
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: object) -> SecurityInspectionResult:
        return SecurityInspectionResult(plugin_id=self.plugin_id, hook_point=str(point))


@pytest.fixture(autouse=True)
def _reset_security_plugin_state() -> None:
    reset_entry_point_spec_cache_for_tests()
    reset_security_defense_registry_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()
    reset_security_defense_registry_for_tests()


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _ep(name: str, attr: str) -> _EntryPoint:
    return _EntryPoint(name, f"{__name__}:{attr}", _GROUP)


def test_oad_002_production_default_is_fail_closed_error() -> None:
    policy = SecurityDefenseAdmissionPolicy()
    assert policy.shipped_id_override == "error"
    assert policy.plugin_id_conflict == "error"
    assert policy.ep_name_conflict == "error"


def test_discovery_disabled_returns_empty_report() -> None:
    report = load_security_defense_plugin_report(discover_entry_points=False)
    assert report.registered_count == 0
    assert report.accepted == ()
    assert report.failed == ()
    assert report.rejected == ()
    assert report.critical_bootstrap_acceptable is True


def test_one_valid_ep_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaDefense")])
    report = load_security_defense_plugin_report(discover_entry_points=True)
    assert report.registered_count == 1
    assert [item.name for item in report.accepted] == ["alpha"]
    plugin = get_security_defense_plugin("lab.alpha")
    assert plugin is not None
    assert isinstance(plugin, _AlphaDefense)


def test_broken_ep_isolated_valid_siblings_register(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("alpha", "_AlphaDefense"),
            _EntryPoint("broken", "not-a-valid-target", _GROUP),
            _ep("beta", "_BetaDefense"),
        ],
    )
    report = load_security_defense_plugin_report(
        discover_entry_points=True,
        admission=SecurityDefenseAdmissionPolicy(on_load_failure="isolate"),
    )
    assert [item.name for item in report.accepted] == ["alpha", "beta"]
    assert [item.spec.name for item in report.failed] == ["broken"]
    assert report.registered_count == 2
    assert get_security_defense_plugin("lab.alpha") is not None
    assert get_security_defense_plugin("lab.beta") is not None


def test_broken_ep_fail_fast_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("alpha", "_AlphaDefense"),
            _EntryPoint("broken", "not-a-valid-target", _GROUP),
        ],
    )
    with pytest.raises(PluginLoadError):
        load_security_defense_plugin_report(
            discover_entry_points=True,
            admission=SecurityDefenseAdmissionPolicy(on_load_failure="fail_fast"),
        )


def test_invalid_target_type_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("alpha", "_AlphaDefense"),
            _ep("nope", "_NotADefense"),
        ],
    )
    report = load_security_defense_plugin_report(discover_entry_points=True)
    assert [item.name for item in report.accepted] == ["alpha"]
    assert report.rejected[0].reason_code is PluginAdmissionReasonCode.INVALID_TARGET_TYPE
    assert report.rejected[0].spec.name == "nope"
    assert report.critical_bootstrap_acceptable is False


def test_invalid_target_type_fail_fast_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_ep("nope", "_NotADefense")])
    with pytest.raises(TypeError, match="SecurityDefensePlugin"):
        load_security_defense_plugin_report(
            discover_entry_points=True,
            admission=SecurityDefenseAdmissionPolicy(on_load_failure="fail_fast"),
        )


def test_duplicate_ep_name_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("dup", "_AlphaDefense"),
            _ep("dup", "_BetaDefense"),
        ],
    )
    with pytest.raises(PluginConflictError):
        load_security_defense_plugin_report(discover_entry_points=True)


def test_duplicate_ep_name_skip_keeps_first(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("dup", "_AlphaDefense"),
            _ep("dup", "_BetaDefense"),
        ],
    )
    report = load_security_defense_plugin_report(
        discover_entry_points=True,
        admission=SecurityDefenseAdmissionPolicy(ep_name_conflict="skip"),
    )
    assert report.registered_count == 1
    first = sorted(
        [f"{__name__}:_AlphaDefense", f"{__name__}:_BetaDefense"],
    )[0]
    if first.endswith("_AlphaDefense"):
        assert get_security_defense_plugin("lab.alpha") is not None
        assert get_security_defense_plugin("lab.beta") is None
    else:
        assert get_security_defense_plugin("lab.beta") is not None
        assert get_security_defense_plugin("lab.alpha") is None


def test_duplicate_ep_name_override_registers_both_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("dup", "_AlphaDefense"),
            _ep("dup", "_BetaDefense"),
        ],
    )
    report = load_security_defense_plugin_report(
        discover_entry_points=True,
        admission=SecurityDefenseAdmissionPolicy(ep_name_conflict="override"),
    )
    assert report.registered_count == 2
    assert get_security_defense_plugin("lab.alpha") is not None
    assert get_security_defense_plugin("lab.beta") is not None


def test_duplicate_plugin_id_unique_ep_names_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("foo", "_AlphaDefense"),
            _ep("bar", "_GammaDefense"),
        ],
    )
    report = load_security_defense_plugin_report(discover_entry_points=True)
    assert report.registered_count == 1
    collision = report.rejected[0]
    assert collision.reason_code is PluginAdmissionReasonCode.PLUGIN_ID_COLLISION
    assert collision.plugin_id == "lab.alpha"
    assert collision.fail_closed is True
    assert get_security_defense_plugin("lab.alpha") is not None
    assert report.critical_bootstrap_acceptable is False


def test_shipped_id_collision_does_not_override(monkeypatch: pytest.MonkeyPatch) -> None:
    shipped = get_security_defense_plugin("harness.strict_injection")
    assert shipped is not None
    _install_eps(monkeypatch, [_ep("evil", "_ShippedIdDefense")])
    report = load_security_defense_plugin_report(discover_entry_points=True)
    assert report.registered_count == 0
    assert report.rejected[0].reason_code is PluginAdmissionReasonCode.SHIPPED_ID_COLLISION
    active = get_security_defense_plugin("harness.strict_injection")
    assert active is shipped
    assert not isinstance(active, _ShippedIdDefense)


def test_authorized_shipped_override_replaces_when_policy_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(monkeypatch, [_ep("patch", "_ShippedIdDefense")])
    report = load_security_defense_plugin_report(
        discover_entry_points=True,
        admission=SecurityDefenseAdmissionPolicy(shipped_id_override="allow"),
    )
    assert report.registered_count == 1
    active = get_security_defense_plugin("harness.strict_injection")
    assert isinstance(active, _ShippedIdDefense)


def test_legacy_policy_restores_unconditional_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_ep("patch", "_ShippedIdDefense")])
    report = load_security_defense_plugin_report(
        discover_entry_points=True,
        admission=LEGACY_UNCONDITIONAL_OVERRIDE_POLICY,
    )
    assert report.registered_count == 1
    assert isinstance(get_security_defense_plugin("harness.strict_injection"), _ShippedIdDefense)


def test_already_registered_plugin_id_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    register_security_defense_plugin(_PreexistingDefense())
    _install_eps(monkeypatch, [_ep("late", "_CollidingPreexistingDefense")])
    report = load_security_defense_plugin_report(discover_entry_points=True)
    assert report.registered_count == 0
    assert report.rejected[0].reason_code is PluginAdmissionReasonCode.ALREADY_REGISTERED
    assert isinstance(get_security_defense_plugin("lab.preexisting"), _PreexistingDefense)


def test_deterministic_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("zeta", "_BetaDefense"),
            _EntryPoint("broken", "not-a-valid-target", _GROUP),
            _ep("alpha", "_AlphaDefense"),
        ],
    )
    report = load_security_defense_plugin_report(discover_entry_points=True)
    assert [item.name for item in report.accepted] == ["alpha", "zeta"]
    assert [item.spec.name for item in report.failed] == ["broken"]


def test_shipped_unaffected_when_discovery_disabled() -> None:
    before = get_security_defense_plugin("harness.strict_injection")
    result = bootstrap_security_providers(discover_entry_points=False)
    after = get_security_defense_plugin("harness.strict_injection")
    assert before is after
    assert result.entry_point_plugins == 0
    assert result.load_report.registered_count == 0
    assert result.critical_bootstrap_acceptable is True
    assert "harness.strict_injection" in result.shipped_bundle_ids


def test_block_c_does_not_represent_qualification_until_host_admission_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BLOCK C defers qualification; no inert require-qualification policy field."""
    policy_field_names = {field.name for field in fields(SecurityDefenseAdmissionPolicy)}
    assert "require_production_qualification" not in policy_field_names

    _install_eps(monkeypatch, [_ep("alpha", "_AlphaDefense")])
    report = load_security_defense_plugin_report(
        discover_entry_points=True,
        admission=SecurityDefenseAdmissionPolicy(),
    )
    assert report.registered_count == 1
    assert report.rejected == ()
