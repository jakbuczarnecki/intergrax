# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.payload_registry import (
    EVENT_TYPE_PREFERRED_SCHEMA,
    get_payload_schema,
)
from intergrax.runtime.events.runtime_event_payload_policy import (
    CANONICAL_PRODUCTION_STRICT_SPINE_EVENT_TYPES,
    PayloadWriteMode,
    RuntimeEventTypeClassification,
    get_runtime_event_payload_policy,
    iter_runtime_event_payload_policies,
)

pytestmark = pytest.mark.gate


def test_every_runtime_event_type_has_exactly_one_policy_entry() -> None:
    policies = dict(iter_runtime_event_payload_policies())
    assert set(policies) == set(RuntimeEventType)


def test_canonical_strict_spine_types_have_registered_schema() -> None:
    missing_schema = CANONICAL_PRODUCTION_STRICT_SPINE_EVENT_TYPES - set(
        EVENT_TYPE_PREFERRED_SCHEMA
    )
    assert missing_schema == set()
    for event_type in CANONICAL_PRODUCTION_STRICT_SPINE_EVENT_TYPES:
        policy = get_runtime_event_payload_policy(event_type)
        assert policy.classification == RuntimeEventTypeClassification.CANONICAL_PRODUCTION
        assert policy.write_mode == PayloadWriteMode.STRICT_SPINE_SCHEMA
        schema_id = policy.schema_id
        assert schema_id is not None
        assert get_payload_schema(schema_id) is not None


def test_domain_signal_uses_extension_event_kind_write_mode() -> None:
    policy = get_runtime_event_payload_policy(RuntimeEventType.DOMAIN_SIGNAL)
    assert policy.write_mode == PayloadWriteMode.EXTENSION_EVENT_KIND
    assert policy.schema_id is None
