from __future__ import annotations

import pytest

from intergrax.runtime.architecture.evaluation_modes import EvaluationMode, EvaluationModeRequest


def test_offline_mode_requires_dataset_ref() -> None:
    with pytest.raises(ValueError, match="dataset_ref"):
        EvaluationModeRequest(
            run_id="offline",
            target_id="agent:echo",
            mode=EvaluationMode.OFFLINE,
        )


def test_online_mode_requires_traffic_slice_ref() -> None:
    with pytest.raises(ValueError, match="traffic_slice_ref"):
        EvaluationModeRequest(
            run_id="online",
            target_id="agent:echo",
            mode=EvaluationMode.ONLINE,
        )


def test_human_mode_requires_reviewer_ref() -> None:
    with pytest.raises(ValueError, match="reviewer_ref"):
        EvaluationModeRequest(
            run_id="human",
            target_id="agent:echo",
            mode=EvaluationMode.HUMAN,
        )


def test_shadow_mode_with_traffic_is_valid() -> None:
    request = EvaluationModeRequest(
        run_id="shadow",
        target_id="agent:echo",
        mode=EvaluationMode.SHADOW,
        traffic_slice_ref="traffic/shadow/echo",
    )
    assert request.mode == EvaluationMode.SHADOW
