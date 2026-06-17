# © Artur Czarnecki. All rights reserved.

"""APP-EVOL-8 — hierarchical bundle normalization and digest parity."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.environment_snapshot_wiring import (
    compute_profile_snapshot_id,
    stable_digest_hex,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    bundle_normalized_payload,
    flatten_profile_dict,
    lift_flat_profile_dict,
)
from intergrax.applications.contracts.environment_profile.normalization import (
    BUNDLE_ROOT_KEYS,
    _strip_null_nodes,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.integrations.registry.bootstrap import register_default_integrations

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


@pytest.fixture(autouse=True)
def _integrations_catalog() -> None:
    register_default_integrations(override=True)


def test_root_exposes_nested_bundle_fields() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    assert env.meta.profile_id == env.profile_id
    assert env.capabilities.tools is env.tool_profile
    assert env.cognition.orchestration is env.orchestration_profile
    assert env.governance.reliability is env.reliability_profile


def test_flat_json_round_trips_through_lift() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="bundle.roundtrip")
    flat = env.model_dump(mode="json")
    assert "meta" not in flat
    assert flat["profile_id"] == "bundle.roundtrip"

    restored = ApplicationEnvironmentProfile.model_validate(flat)
    assert restored.profile_id == env.profile_id
    assert restored.tool_profile.enabled == env.tool_profile.enabled
    assert restored.orchestration_profile.long_running_enabled == (
        env.orchestration_profile.long_running_enabled
    )


def test_nested_bundle_dump_round_trips() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="bundle.nested")
    nested = env.bundle_dump(mode="json")
    assert set(nested).issuperset(BUNDLE_ROOT_KEYS)
    restored = ApplicationEnvironmentProfile.model_validate(nested)
    assert restored.profile_id == env.profile_id
    assert restored.cost_profile.max_total_tokens == env.cost_profile.max_total_tokens


def test_flat_and_nested_share_bundle_digest() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="bundle.digest")
    flat_payload = lift_flat_profile_dict(env.model_dump(mode="json"))
    nested_payload = env.bundle_dump(mode="json")
    assert bundle_normalized_payload(flat_payload) == bundle_normalized_payload(nested_payload)


def test_model_copy_flat_update_preserves_bundle_shape() -> None:
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="bundle.copy")
    mutated = base.model_copy(
        update={
            "execution_mode": ExecutionMode.STRICT,
            "tool_invocation_mode": "parallel",
        },
    )
    assert mutated.execution_mode == ExecutionMode.STRICT
    assert mutated.tool_invocation_mode == "parallel"
    assert mutated.capabilities.tool_invocation.mode == "parallel"


def test_snapshot_id_uses_bundle_normalized_payload() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="bundle.snapshot")
    expected = stable_digest_hex(bundle_normalized_payload(env.bundle_dump(mode="json")))
    assert compute_profile_snapshot_id(env) == f"prof_{expected[:24]}"


def test_flatten_inverts_lift_for_wire_keys() -> None:
    nested = ApplicationEnvironmentProfile.lab_defaults().bundle_dump(mode="json")
    flat = flatten_profile_dict(nested)
    relifted = lift_flat_profile_dict(flat)
    assert _strip_null_nodes(relifted) == _strip_null_nodes(nested)
