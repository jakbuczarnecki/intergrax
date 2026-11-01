# © Artur Czarnecki. All rights reserved.

"""W3-01/W3-03: COMPAT-X policy diff over the real contract registry + persisted/event schema diffs."""

from __future__ import annotations

import copy

import pytest
from testing_support.change_safety.assess.contract_diff import (
    assess_persisted_schema_change,
    diff_event_schema,
)
from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult as R
from tests.qualification.evol_x.adapters.compat_x import CompatXRegistryAdapter

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_ADAPTER = CompatXRegistryAdapter()


def test_evol_x_compat_x_adapter_reads_registry() -> None:
    names = _ADAPTER.list_contract_names()
    assert names
    assert _ADAPTER.diff_schemas({"properties": {}}, {"properties": {}}).value == "COMPATIBLE"


@pytest.mark.parametrize("contract_name", sorted(_ADAPTER.list_contract_names()))
def test_evol_x_compat_registry_schema_self_compatible_and_mutations_detected(contract_name: str) -> None:
    schema = _ADAPTER.schema_for(contract_name)
    assert _ADAPTER.diff_schemas(schema, copy.deepcopy(schema)) is R.COMPATIBLE
    required = list(schema.get("required") or ())
    if required:
        dropped = copy.deepcopy(schema)
        dropped["required"] = required[1:]
        dropped["properties"].pop(required[0], None)
        assert _ADAPTER.diff_schemas(schema, dropped) is R.INCOMPATIBLE  # incompatible arm
    widened = copy.deepcopy(schema)
    widened.setdefault("properties", {})["evol_probe_new_required"] = {"type": "string"}
    widened["required"] = [*required, "evol_probe_new_required"]
    assert _ADAPTER.diff_schemas(schema, widened) is R.MIGRATION_REQUIRED


def test_evol_x_compat_type_change_enum_narrowing_removed_optional() -> None:
    base = {
        "properties": {"a": {"type": "string"}, "k": {"enum": ["x", "y"]}, "o": {"type": "integer"}},
        "required": ["a"],
    }
    retyped = copy.deepcopy(base)
    retyped["properties"]["a"]["type"] = "integer"
    assert _ADAPTER.diff_schemas(base, retyped) is R.INCOMPATIBLE
    narrowed = copy.deepcopy(base)
    narrowed["properties"]["k"]["enum"] = ["x"]
    assert _ADAPTER.diff_schemas(base, narrowed) is R.INCOMPATIBLE
    removed_optional = copy.deepcopy(base)
    removed_optional["properties"].pop("o")
    rec = _ADAPTER.diff_records(base, removed_optional)
    assert rec.result is R.INCOMPATIBLE and any("removed optional" in r for r in rec.reasons)
    optional_added = copy.deepcopy(base)
    optional_added["properties"]["extra"] = {"type": "string"}
    assert _ADAPTER.diff_schemas(base, optional_added) is R.COMPATIBLE


def test_evol_x_compat_unknown_when_not_comparable_fail_closed() -> None:
    assert _ADAPTER.diff_records(None, {"properties": {}}).result is R.UNKNOWN
    assert _ADAPTER.diff_records({"properties": {}}, None).result is R.UNKNOWN


def test_evol_x_persisted_schema_bump_without_migration_is_never_silent() -> None:
    before = {"properties": {"a": {"type": "string"}}, "required": ["a"]}
    after_new_required = {"properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}
    silent = assess_persisted_schema_change(before, after_new_required, migration_registered=False)
    assert silent.result is R.INCOMPATIBLE and any("silent break" in r for r in silent.reasons)
    declared = assess_persisted_schema_change(before, after_new_required, migration_registered=True)
    assert declared.result is R.MIGRATION_REQUIRED
    breaking = {"properties": {"a": {"type": "integer"}}, "required": ["a"]}
    assert assess_persisted_schema_change(before, breaking, migration_registered=False).result is R.INCOMPATIBLE
    assert assess_persisted_schema_change(before, breaking, migration_registered=True).result is R.MIGRATION_REQUIRED
    assert assess_persisted_schema_change(before, before, migration_registered=False).result is R.COMPATIBLE


def test_evol_x_event_schema_field_removal_incompatible() -> None:
    before = {"properties": {"id": {"type": "string"}, "ts": {"type": "string"}}, "required": ["id"]}
    after = {"properties": {"id": {"type": "string"}}, "required": ["id"]}
    assert diff_event_schema(before, after).result is R.INCOMPATIBLE
