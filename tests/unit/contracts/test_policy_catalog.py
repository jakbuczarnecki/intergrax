# © Artur Czarnecki. All rights reserved.

"""PolicyDefinition catalog contract (Governed Execution G2B)."""

from __future__ import annotations

from typing import Any, get_args, get_origin

import pytest
from pydantic import ValidationError

from intergrax.utils import attribute_access
from intergrax.contracts.policy_catalog import PolicyDefinition, PolicyDefinitionSource

pytestmark = [pytest.mark.unit]

_REQUIRED_FIELDS = (
    "policy_id",
    "version",
    "display_name",
    "handler_id",
    "configuration_contract_id",
)


def _definition(**overrides: object) -> PolicyDefinition:
    payload: dict[str, object] = {
        "policy_id": "external_commitment_approval",
        "version": "2",
        "display_name": "External commitment approval",
        "handler_id": "meaningful_side_effect_approval",
        "configuration_contract_id": "meaningful_side_effect.external_commitment.v2",
        "source": PolicyDefinitionSource.BUILT_IN,
    }
    payload.update(overrides)
    return PolicyDefinition.model_validate(payload)


def test_valid_built_in_policy_definition() -> None:
    definition = _definition(source=PolicyDefinitionSource.BUILT_IN)
    assert definition.source is PolicyDefinitionSource.BUILT_IN
    assert definition.policy_id == "external_commitment_approval"


def test_valid_plugin_policy_definition() -> None:
    definition = _definition(
        source=PolicyDefinitionSource.PLUGIN,
        policy_id="custom_plugin_policy",
    )
    assert definition.source is PolicyDefinitionSource.PLUGIN


def test_policy_definition_is_immutable() -> None:
    definition = _definition()
    with pytest.raises(ValidationError):
        definition.policy_id = "mutated"


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        PolicyDefinition.model_validate(
            {
                "policy_id": "external_commitment_approval",
                "version": "2",
                "display_name": "External commitment approval",
                "handler_id": "meaningful_side_effect_approval",
                "configuration_contract_id": "meaningful_side_effect.external_commitment.v2",
                "source": "built_in",
                "rule_id": "finance.contracts.require_cfo",
            }
        )


@pytest.mark.parametrize("field_name", _REQUIRED_FIELDS)
def test_whitespace_normalized_on_required_fields(field_name: str) -> None:
    definition = _definition(**{field_name: f"  value-for-{field_name}  "})
    assert attribute_access.optional(definition, field_name) == f"value-for-{field_name}"


def test_description_whitespace_normalized() -> None:
    definition = _definition(description="  optional detail  ")
    assert definition.description == "optional detail"


@pytest.mark.parametrize("field_name", _REQUIRED_FIELDS)
def test_empty_required_field_rejected(field_name: str) -> None:
    with pytest.raises(ValidationError):
        _definition(**{field_name: "   "})


def test_invalid_source_rejected() -> None:
    with pytest.raises(ValidationError):
        _definition(source="external")


def test_schema_version_defaults_to_policy_definition_v1() -> None:
    assert _definition().schema_version == "policy_definition.v1"


def test_unsupported_schema_version_rejected() -> None:
    with pytest.raises(ValidationError):
        _definition(schema_version="policy_definition.v2")


def test_literal_policy_id_equals_handler_id_is_legal() -> None:
    definition = _definition(
        policy_id="deny_tool",
        handler_id="deny_tool",
    )
    assert definition.policy_id == definition.handler_id == "deny_tool"


def test_policy_definition_excludes_rule_and_bundle_identity_fields() -> None:
    forbidden = {"rule_id", "bundle_id", "bundle_version", "bundle_digest"}
    assert forbidden.isdisjoint(PolicyDefinition.model_fields.keys())


def test_policy_definition_has_no_arbitrary_dictionary_fields() -> None:
    dict_origins = {dict, dict[str, Any]}

    def _is_dict_annotation(annotation: object) -> bool:
        origin = get_origin(annotation)
        if origin in dict_origins:
            return True
        if origin is not None:
            return any(arg in dict_origins for arg in get_args(annotation))
        return annotation in dict_origins

    for field_name, field_info in PolicyDefinition.model_fields.items():
        assert not _is_dict_annotation(field_info.annotation), field_name
