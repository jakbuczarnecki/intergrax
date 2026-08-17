# © Artur Czarnecki. All rights reserved.

"""Unit tests for PolicyCatalog resolution core (Governed Execution G2C-1)."""

from __future__ import annotations

import importlib
import sys

import pytest

from intergrax.contracts.policy_catalog import PolicyDefinition, PolicyDefinitionSource
from intergrax.runtime.policy.catalog import (
    PolicyCatalog,
    PolicyCatalogError,
    PolicyDefinitionConflictError,
    UnknownPolicyDefinitionError,
    UnsupportedPolicyDefinitionVersionError,
)

pytestmark = pytest.mark.unit


def _definition(**overrides: object) -> PolicyDefinition:
    payload: dict[str, object] = {
        "policy_id": "example_policy",
        "version": "1",
        "display_name": "Example policy",
        "handler_id": "example_handler",
        "configuration_contract_id": "example.config.v1",
        "source": PolicyDefinitionSource.BUILT_IN,
    }
    payload.update(overrides)
    return PolicyDefinition.model_validate(payload)


def test_empty_catalog_constructs_successfully() -> None:
    catalog = PolicyCatalog()
    assert catalog.definitions() == ()


def test_empty_catalog_unknown_policy_raises() -> None:
    catalog = PolicyCatalog()
    with pytest.raises(UnknownPolicyDefinitionError) as exc:
        catalog.resolve(policy_id="missing", version="1")
    assert exc.value.policy_id == "missing"


def test_exact_single_definition_resolution() -> None:
    definition = _definition()
    catalog = PolicyCatalog((definition,))
    assert catalog.resolve(policy_id="example_policy", version="1") is definition


def test_multiple_different_policies_resolve_independently() -> None:
    first = _definition(policy_id="policy_a", version="1")
    second = _definition(policy_id="policy_b", version="1")
    catalog = PolicyCatalog((first, second))
    assert catalog.resolve(policy_id="policy_a", version="1") is first
    assert catalog.resolve(policy_id="policy_b", version="1") is second


def test_two_versions_of_same_policy_id_coexist() -> None:
    version_one = _definition(version="1")
    version_two = _definition(version="2")
    catalog = PolicyCatalog((version_one, version_two))
    assert catalog.resolve(policy_id="example_policy", version="1") is version_one
    assert catalog.resolve(policy_id="example_policy", version="2") is version_two


def test_exact_version_resolution_returns_requested_object() -> None:
    version_one = _definition(version="1", display_name="Version one")
    version_two = _definition(version="2", display_name="Version two")
    catalog = PolicyCatalog((version_one, version_two))
    resolved = catalog.resolve(policy_id="example_policy", version="2")
    assert resolved is version_two
    assert resolved.display_name == "Version two"


def test_unknown_policy_id_raises_unknown_error() -> None:
    catalog = PolicyCatalog((_definition(),))
    with pytest.raises(UnknownPolicyDefinitionError) as exc:
        catalog.resolve(policy_id="missing", version="1")
    assert exc.value.policy_id == "missing"


def test_known_policy_unsupported_version_raises() -> None:
    catalog = PolicyCatalog((_definition(version="1"),))
    with pytest.raises(UnsupportedPolicyDefinitionVersionError) as exc:
        catalog.resolve(policy_id="example_policy", version="3")
    assert exc.value.policy_id == "example_policy"
    assert exc.value.version == "3"


def test_unsupported_version_never_resolves_another_version() -> None:
    version_one = _definition(version="1")
    version_two = _definition(version="2")
    catalog = PolicyCatalog((version_one, version_two))
    with pytest.raises(UnsupportedPolicyDefinitionVersionError):
        catalog.resolve(policy_id="example_policy", version="3")


@pytest.mark.parametrize(
    ("first_source", "second_source"),
    [
        (PolicyDefinitionSource.BUILT_IN, PolicyDefinitionSource.BUILT_IN),
        (PolicyDefinitionSource.BUILT_IN, PolicyDefinitionSource.PLUGIN),
        (PolicyDefinitionSource.PLUGIN, PolicyDefinitionSource.BUILT_IN),
        (PolicyDefinitionSource.PLUGIN, PolicyDefinitionSource.PLUGIN),
    ],
)
def test_duplicate_exact_identity_raises_conflict(
    first_source: PolicyDefinitionSource,
    second_source: PolicyDefinitionSource,
) -> None:
    first = _definition(source=first_source, display_name="First")
    second = _definition(source=second_source, display_name="Second")
    with pytest.raises(PolicyDefinitionConflictError) as exc:
        PolicyCatalog((first, second))
    assert exc.value.policy_id == "example_policy"
    assert exc.value.version == "1"


def test_duplicate_detection_independent_of_input_ordering() -> None:
    first = _definition(source=PolicyDefinitionSource.BUILT_IN)
    second = _definition(source=PolicyDefinitionSource.PLUGIN)
    with pytest.raises(PolicyDefinitionConflictError):
        PolicyCatalog((second, first))


def test_definitions_listing_is_deterministic() -> None:
    definitions = (
        _definition(policy_id="policy_b", version="2"),
        _definition(policy_id="policy_a", version="2"),
        _definition(policy_id="policy_a", version="1"),
        _definition(policy_id="policy_b", version="1"),
    )
    catalog = PolicyCatalog(definitions)
    listed = catalog.definitions()
    assert listed == tuple(
        sorted(definitions, key=lambda item: (item.policy_id, item.version))
    )


def test_definitions_returns_tuple() -> None:
    catalog = PolicyCatalog((_definition(),))
    listed = catalog.definitions()
    assert isinstance(listed, tuple)
    assert listed == (catalog.resolve(policy_id="example_policy", version="1"),)


def test_catalog_does_not_mutate_supplied_definitions() -> None:
    definition = _definition()
    before = definition.model_dump()
    PolicyCatalog((definition,))
    assert definition.model_dump() == before


def test_lookup_trims_policy_id_and_version() -> None:
    definition = _definition()
    catalog = PolicyCatalog((definition,))
    assert catalog.resolve(policy_id="  example_policy  ", version="  1  ") is definition


def test_whitespace_only_policy_id_lookup_fails() -> None:
    catalog = PolicyCatalog((_definition(),))
    with pytest.raises(UnknownPolicyDefinitionError) as exc:
        catalog.resolve(policy_id="   ", version="1")
    assert exc.value.policy_id == ""


def test_whitespace_only_version_lookup_fails() -> None:
    catalog = PolicyCatalog((_definition(),))
    with pytest.raises(UnsupportedPolicyDefinitionVersionError) as exc:
        catalog.resolve(policy_id="example_policy", version="   ")
    assert exc.value.policy_id == "example_policy"
    assert exc.value.version == ""


def test_no_resolve_latest_api_exists() -> None:
    catalog = PolicyCatalog()
    forbidden = ("latest", "resolve_latest", "default_version", "resolve_default")
    for name in forbidden:
        assert not hasattr(catalog, name)


def test_no_silent_override_api_exists() -> None:
    catalog = PolicyCatalog()
    forbidden = ("register", "unregister", "override", "add", "update")
    for name in forbidden:
        assert not hasattr(catalog, name)


def test_catalog_module_has_no_policy_rule_registry_dependency() -> None:
    module_name = "intergrax.runtime.policy.catalog"
    saved_modules = {
        name: sys.modules.get(name)
        for name in list(sys.modules)
        if name == module_name or name.startswith("intergrax.runtime.policy.rules")
    }
    for name in list(saved_modules):
        if name is not None:
            sys.modules.pop(name, None)

    try:
        importlib.import_module(module_name)
        imported = sys.modules[module_name]
        source_path = imported.__file__
        assert source_path is not None
        with open(source_path, encoding="utf-8") as handle:
            source = handle.read()
        assert "PolicyRuleRegistry" not in source
        assert "intergrax.runtime.policy.rules" not in source
    finally:
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)


def test_exception_hierarchy_is_small() -> None:
    assert issubclass(UnknownPolicyDefinitionError, PolicyCatalogError)
    assert issubclass(UnsupportedPolicyDefinitionVersionError, PolicyCatalogError)
    assert issubclass(PolicyDefinitionConflictError, PolicyCatalogError)
