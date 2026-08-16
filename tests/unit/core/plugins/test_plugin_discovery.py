# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.core.plugins.discovery import (
    get_entry_point_spec,
    instantiate_entry_point_target,
    iter_entry_point_specs,
    load_entry_point_plugins,
    load_entry_point_targets,
    load_entry_point_value,
    load_plugin_types,
    reset_entry_point_spec_cache_for_tests,
    resolve_entry_point_plugin_type,
)
from intergrax.core.plugins.errors import PluginConflictError, PluginLoadError
from intergrax.integrations.examples.custom_memory_kv import CustomMemoryKvPlugin
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.policy.rules.plugin_loader import load_policy_rule_plugins
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry
from intergrax.runtime.security.defense_plugin import SecurityFailMode, SecurityInspectionResult
from intergrax.runtime.security.defense_plugin_loader import load_security_defense_plugins
from intergrax.runtime.security.defense_registry import get_security_defense_plugin

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


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


class _DiscoveredPlugin:
    pass


class _CallableObject:
    def __init__(self) -> None:
        self.called = False

    def __call__(self) -> None:
        self.called = True


def _plugin_factory() -> type:
    return _DiscoveredPlugin


def _factory_invocation_probe() -> type:
    global _FACTORY_WAS_INVOKED
    _FACTORY_WAS_INVOKED = True
    return _DiscoveredPlugin


_FACTORY_WAS_INVOKED = False
_CALLABLE_OBJECT_PROBE = _CallableObject()


def test_resolve_entry_point_plugin_type_class_target() -> None:
    assert resolve_entry_point_plugin_type(_DiscoveredPlugin, "mod:Cls") is _DiscoveredPlugin


def test_resolve_entry_point_plugin_type_factory_returns_class() -> None:
    assert resolve_entry_point_plugin_type(_plugin_factory, "mod:factory") is _DiscoveredPlugin


def test_resolve_entry_point_plugin_type_factory_raises() -> None:
    def _broken_factory() -> type:
        raise RuntimeError("boom")

    with pytest.raises(PluginLoadError, match="Failed to call entry point factory"):
        resolve_entry_point_plugin_type(_broken_factory, "mod:broken")


def test_resolve_entry_point_plugin_type_factory_returns_non_class() -> None:
    def _bad_factory() -> object:
        return object()

    with pytest.raises(PluginLoadError, match="must return a plugin class"):
        resolve_entry_point_plugin_type(_bad_factory, "mod:bad")


def test_resolve_entry_point_plugin_type_invalid_target() -> None:
    with pytest.raises(PluginLoadError, match="is not a class or factory"):
        resolve_entry_point_plugin_type(42, "mod:bad")


def test_load_entry_point_value_class_returns_class_not_instance() -> None:
    target = load_entry_point_value(f"{__name__}:_DiscoveredPlugin")
    assert target is _DiscoveredPlugin
    assert not isinstance(target, _DiscoveredPlugin)


def test_load_entry_point_value_function_not_invoked() -> None:
    global _FACTORY_WAS_INVOKED
    _FACTORY_WAS_INVOKED = False
    target = load_entry_point_value(f"{__name__}:_factory_invocation_probe")
    assert callable(target)
    assert target is _factory_invocation_probe
    assert _FACTORY_WAS_INVOKED is False


def test_load_entry_point_value_callable_object_not_invoked() -> None:
    target = load_entry_point_value(f"{__name__}:_CALLABLE_OBJECT_PROBE")
    assert target is _CALLABLE_OBJECT_PROBE
    assert _CALLABLE_OBJECT_PROBE.called is False


def test_load_entry_point_value_module_only_delegates_to_entry_point_load() -> None:
    import tests.unit.core.plugins.test_plugin_discovery as this_module

    target = load_entry_point_value(__name__)
    assert target is this_module


def test_load_entry_point_plugins_supports_callable_factory_returning_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [_EntryPoint("factory", f"{__name__}:_plugin_factory", "intergrax.rag.chunkers")]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)
    loaded = load_entry_point_plugins("intergrax.rag.chunkers")
    assert loaded[0].plugin_type is _DiscoveredPlugin


def test_instantiate_entry_point_target_returns_non_class_callable_unchanged() -> None:
    assert instantiate_entry_point_target(_factory_invocation_probe) is _factory_invocation_probe


def test_load_plugin_types_explicit_only() -> None:
    types = load_plugin_types(
        "intergrax.integrations",
        explicit=(CustomMemoryKvPlugin,),
        discover_entry_points=False,
    )
    assert types == [CustomMemoryKvPlugin]


def test_load_entry_point_value_invalid_target_raises() -> None:
    with pytest.raises(PluginLoadError, match="Failed to load entry point target"):
        load_entry_point_value("not-a-valid-target")


def test_iter_entry_point_specs_is_deterministic_and_scan_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint("b", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers"),
            _EntryPoint("a", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers"),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    specs = iter_entry_point_specs("intergrax.rag.chunkers")

    assert [spec.name for spec in specs] == ["a", "b"]
    assert all(spec.group == "intergrax.rag.chunkers" for spec in specs)


def test_iter_entry_point_specs_reuses_cache_without_rescanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [_EntryPoint("chunker", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers")]
    )
    scan_calls = 0

    def _entry_points() -> _EntryPoints:
        nonlocal scan_calls
        scan_calls += 1
        return entries

    monkeypatch.setattr(importlib.metadata, "entry_points", _entry_points)

    first = iter_entry_point_specs("intergrax.rag.chunkers")
    second = iter_entry_point_specs("intergrax.rag.chunkers")

    assert first == second
    assert scan_calls == 1


def test_reset_entry_point_spec_cache_for_tests_causes_rescan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [_EntryPoint("chunker", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers")]
    )
    scan_calls = 0

    def _entry_points() -> _EntryPoints:
        nonlocal scan_calls
        scan_calls += 1
        return entries

    monkeypatch.setattr(importlib.metadata, "entry_points", _entry_points)

    iter_entry_point_specs("intergrax.rag.chunkers")
    reset_entry_point_spec_cache_for_tests()
    iter_entry_point_specs("intergrax.rag.chunkers")

    assert scan_calls == 2


def test_get_entry_point_spec_duplicate_name_returns_first_sorted_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_value = f"{__name__}:_DiscoveredPlugin"
    second_value = f"{__name__}:_plugin_factory"
    entries = _EntryPoints(
        [
            _EntryPoint("dup", second_value, "intergrax.rag.chunkers"),
            _EntryPoint("dup", first_value, "intergrax.rag.chunkers"),
        ]
    )
    scan_calls = 0

    def _entry_points() -> _EntryPoints:
        nonlocal scan_calls
        scan_calls += 1
        return entries

    def _fail_if_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("entry-point targets must not load during spec lookup")

    monkeypatch.setattr(importlib.metadata, "entry_points", _entry_points)
    monkeypatch.setattr(
        "intergrax.core.plugins.discovery.load_entry_point_value",
        _fail_if_load,
    )

    specs = iter_entry_point_specs("intergrax.rag.chunkers")
    first_lookup = get_entry_point_spec("intergrax.rag.chunkers", "dup")
    second_lookup = get_entry_point_spec("intergrax.rag.chunkers", "dup")

    assert [spec.value for spec in specs] == [first_value, second_value]
    assert first_lookup is not None
    assert first_lookup.value == first_value
    assert second_lookup is first_lookup
    assert scan_calls == 1


def test_get_entry_point_spec_uses_cached_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint("a", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers"),
            _EntryPoint("b", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers"),
        ]
    )
    scan_calls = 0

    def _entry_points() -> _EntryPoints:
        nonlocal scan_calls
        scan_calls += 1
        return entries

    monkeypatch.setattr(importlib.metadata, "entry_points", _entry_points)

    spec = get_entry_point_spec("intergrax.rag.chunkers", "b")
    missing = get_entry_point_spec("intergrax.rag.chunkers", "missing")
    get_entry_point_spec("intergrax.rag.chunkers", "a")

    assert spec is not None
    assert spec.name == "b"
    assert missing is None
    assert scan_calls == 1


def test_load_entry_point_targets_isolates_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint("good", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers"),
            _EntryPoint("bad", "not-a-valid-target", "intergrax.rag.chunkers"),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    results = load_entry_point_targets(
        "intergrax.rag.chunkers",
        on_load_failure="isolate",
    )

    assert len(results) == 2
    assert results[0].spec.name == "bad"
    assert isinstance(results[0].error, PluginLoadError)
    assert results[1].spec.name == "good"
    assert results[1].target is _DiscoveredPlugin


def test_instantiate_entry_point_target_instantiates_classes() -> None:
    instance = instantiate_entry_point_target(_DiscoveredPlugin)
    assert isinstance(instance, _DiscoveredPlugin)


def test_load_entry_point_plugins_scan_does_not_register_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [_EntryPoint("chunker", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers")]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    specs = iter_entry_point_specs("intergrax.rag.chunkers")
    loaded = load_entry_point_plugins("intergrax.rag.chunkers")

    assert len(specs) == 1
    assert loaded[0].plugin_type is _DiscoveredPlugin


def test_load_entry_point_plugins_selects_requested_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "chunker",
                f"{__name__}:_DiscoveredPlugin",
                "intergrax.rag.chunkers",
            ),
            _EntryPoint(
                "other",
                f"{__name__}:_DiscoveredPlugin",
                "intergrax.rag.retrievers",
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    loaded = load_entry_point_plugins("intergrax.rag.chunkers")

    assert [(item.name, item.group, item.plugin_type) for item in loaded] == [
        ("chunker", "intergrax.rag.chunkers", _DiscoveredPlugin)
    ]


def test_load_entry_point_plugins_invalid_target_raises_canonical_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [_EntryPoint("broken", "not-a-valid-target", "intergrax.rag.chunkers")]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    with pytest.raises(PluginLoadError, match="Failed to load entry point target"):
        load_entry_point_plugins("intergrax.rag.chunkers")


def test_load_entry_point_plugins_rejects_duplicate_external_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint("duplicate", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers"),
            _EntryPoint("duplicate", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers"),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    with pytest.raises(PluginConflictError, match="Duplicate entry point"):
        load_entry_point_plugins("intergrax.rag.chunkers")


class _NestedPluginHolder:
    plugin = _DiscoveredPlugin


_NESTED_PLUGIN_HOLDER = _NestedPluginHolder()


class _SecurityDefenseClassPlugin:
    plugin_id = "class-defense"
    version = "1"
    hook_points = frozenset()
    priority = 0
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: object) -> SecurityInspectionResult:
        return SecurityInspectionResult(plugin_id=self.plugin_id, hook_point=str(point))


class _SecurityDefenseInstancePlugin:
    plugin_id = "instance-defense"
    version = "1"
    hook_points = frozenset()
    priority = 0
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: object) -> SecurityInspectionResult:
        return SecurityInspectionResult(plugin_id=self.plugin_id, hook_point=str(point))


_SECURITY_DEFENSE_INSTANCE = _SecurityDefenseInstancePlugin()


class _PolicyRuleClassHandler:
    rule_id = "class-rule"

    def evaluate(self, rule: object, *, context: dict[str, str]) -> object:
        from intergrax.runtime.policy.rules.schema import PolicyRuleAction

        return PolicyRuleAction.ALLOW


class _PolicyRuleInstanceHandler:
    rule_id = "instance-rule"

    def evaluate(self, rule: object, *, context: dict[str, str]) -> object:
        from intergrax.runtime.policy.rules.schema import PolicyRuleAction

        return PolicyRuleAction.ALLOW


_POLICY_RULE_INSTANCE = _PolicyRuleInstanceHandler()


def test_load_entry_point_value_supports_dotted_attribute_targets() -> None:
    target = load_entry_point_value(f"{__name__}:_NESTED_PLUGIN_HOLDER.plugin")
    assert target is _DiscoveredPlugin


def test_security_loader_instantiates_class_targets_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "class-defense",
                f"{__name__}:_SecurityDefenseClassPlugin",
                "intergrax.security_defenses",
            ),
            _EntryPoint(
                "instance-defense",
                f"{__name__}:_SECURITY_DEFENSE_INSTANCE",
                "intergrax.security_defenses",
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    count = load_security_defense_plugins(discover_entry_points=True)

    assert count == 2
    class_plugin = get_security_defense_plugin("class-defense")
    instance_plugin = get_security_defense_plugin("instance-defense")
    assert class_plugin is not None
    assert instance_plugin is not None
    assert isinstance(class_plugin, _SecurityDefenseClassPlugin)
    assert instance_plugin is _SECURITY_DEFENSE_INSTANCE


def test_policy_loader_preserves_class_and_instance_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "class-rule",
                f"{__name__}:_PolicyRuleClassHandler",
                "intergrax.policy_rules",
            ),
            _EntryPoint(
                "instance-rule",
                f"{__name__}:_POLICY_RULE_INSTANCE",
                "intergrax.policy_rules",
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)
    registry = PolicyRuleRegistry()

    count = load_policy_rule_plugins(registry)

    assert count == 2
    class_handler = registry._handlers["class-rule"]
    instance_handler = registry._handlers["instance-rule"]
    assert isinstance(class_handler, _PolicyRuleClassHandler)
    assert instance_handler is _POLICY_RULE_INSTANCE
