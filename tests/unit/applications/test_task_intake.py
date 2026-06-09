# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.task_intake import (
    apply_long_running_enabled,
    apply_long_running_from_profile,
    apply_orchestration_graph_id,
    apply_orchestration_task_defaults,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_metadata_keys import TaskOrchestrationMetadataKey

pytestmark = pytest.mark.gate


def test_apply_orchestration_graph_id_syncs_metadata() -> None:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hi",
        context=TaskContext(capability="cap.a"),
    )
    updated = apply_orchestration_graph_id(task, "graph-42")
    assert updated.runtime.orchestration.graph_id == "graph-42"
    assert updated.metadata[TaskOrchestrationMetadataKey.GRAPH_ID] == "graph-42"


def test_apply_long_running_enabled_uses_options_only() -> None:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hi",
        context=TaskContext(capability="cap.a"),
    )
    updated = apply_long_running_enabled(task, enabled=True)
    assert updated.options.long_running.enabled is True


def test_apply_long_running_from_profile_respects_profile() -> None:
    base = ApplicationEnvironmentProfile.lab_defaults()
    env = base.model_copy(
        update={
            "orchestration_profile": base.orchestration_profile.model_copy(
                update={"long_running_enabled": True},
            ),
        },
    )
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hi",
        context=TaskContext(capability="cap.a"),
    )
    updated = apply_orchestration_task_defaults(task, env)
    assert updated.options.long_running.enabled is True
