# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from intergrax.hosting import (
    LifecyclePolicy,
    RestartPolicy,
    ShutdownPolicy,
)
from intergrax.hosting.contracts.policies import RestartMode, ShutdownStrategy

pytestmark = pytest.mark.unit


def test_standard_presets_construct() -> None:
    assert LifecyclePolicy.standard().component_startup_concurrency == 4
    assert ShutdownPolicy.standard().strategy.value == "drain_then_cancel"
    assert RestartPolicy.on_failure().mode.value == "on_failure"
    assert RestartPolicy.never().max_attempts == 0


def test_bounded_timeout_validation() -> None:
    with pytest.raises(ValidationError):
        LifecyclePolicy(default_blocking_hook_timeout_seconds=0)


def test_invalid_shutdown_combinations_rejected() -> None:
    with pytest.raises(ValidationError, match="cancel_immediately"):
        ShutdownPolicy(
            strategy=ShutdownStrategy.CANCEL_IMMEDIATELY,
            drain_timeout_seconds=10,
            cancel_timeout_seconds=5,
            flush_timeout_seconds=5,
        )


def test_restart_max_backoff_validation() -> None:
    with pytest.raises(ValidationError):
        RestartPolicy.on_failure(max_attempts=200)


def test_custom_restart_callback_excluded_from_public_data() -> None:
    def classifier() -> bool:
        return True

    policy = RestartPolicy(
        mode=RestartMode.CUSTOM,
        custom_classifier=classifier,
        custom_classifier_id="tests.policy.classifier",
    )
    payload = policy.public_dict()
    assert "custom_classifier_id" in payload
    assert "custom_classifier" not in payload


def test_policy_serialization_deterministic() -> None:
    first = json.dumps(LifecyclePolicy.standard().model_dump(mode="json"), sort_keys=True)
    second = json.dumps(LifecyclePolicy.standard().model_dump(mode="json"), sort_keys=True)
    assert first == second
