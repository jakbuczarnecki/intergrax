# © Artur Czarnecki. All rights reserved.
"""T-IGX-051-A Part D: tenant_id on the base online evaluation observation."""

from __future__ import annotations

import pytest
from intergrax.applications._shared.evaluation_assembly_resolver import (
    EvaluationAssemblyError,
    assert_evaluation_assembly_valid,
)
from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import EvaluationProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.runtime.architecture.online_evaluation_models import (
    OnlineEvaluationMode,
    OnlineEvaluationObservation,
)

pytestmark = pytest.mark.unit


def _observation(**overrides: object) -> OnlineEvaluationObservation:
    base = {
        "observation_id": "obs-1",
        "run_id": "run-1",
        "agent_id": "agent-1",
        "mode": OnlineEvaluationMode.SHADOW,
        "scenario_id": "scenario-a",
        "passed": True,
        "score": 0.75,
    }
    base.update(overrides)
    return OnlineEvaluationObservation(**base)


def test_ten_04a_base_model_has_tenant_field() -> None:
    assert "tenant_id" in OnlineEvaluationObservation.model_fields


def test_ten_04b_blank_tenant_rejected() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        _observation(tenant_id="")
    with pytest.raises(ValueError, match="tenant_id"):
        _observation(tenant_id="   ")


def test_ten_04c_valid_tenant_accepted() -> None:
    obs = _observation(tenant_id="tenant-alpha")
    payload = obs.model_dump()
    assert payload["tenant_id"] == "tenant-alpha"
    round_tripped = OnlineEvaluationObservation.model_validate(payload)
    assert round_tripped.tenant_id == "tenant-alpha"


def test_ten_04d_strict_host_without_tenant_refused() -> None:
    """Policy wiring composes evaluation without tenant; STRICT must refuse unscoped registry."""
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="tenant.04d.strict")
    env = env.model_copy(
        update={
            "meta": env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT}),
            "evaluation_profile": EvaluationProfile(
                shadow_eval_enabled=True,
                online_registry_enabled=True,
            ),
        }
    )
    wire_policy_bundle(env)
    evaluation_wiring = wire_application_evaluation(env)
    with pytest.raises(EvaluationAssemblyError, match="G-09"):
        assert_evaluation_assembly_valid(evaluation_wiring, env)
