# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.execution_identity import mint_run_id
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskGovernanceOptions,
    TaskHumanInput,
    TaskIsolationOptions,
)
from intergrax.runtime.task.task_metadata_bridge import (
    execution_options_from_metadata,
    hydrate_task_from_metadata,
    sync_task_metadata,
)


@pytest.mark.unit
@pytest.mark.gate
def test_execution_options_from_legacy_metadata():
    metadata = {
        "shadow_workspace": True,
        "sandbox_cleanup": True,
        "require_human_approval": True,
        "human_response": "approve",
    }
    opts = execution_options_from_metadata(metadata)
    assert opts.isolation.shadow_workspace is True
    assert opts.isolation.sandbox_cleanup is True
    assert opts.governance.require_human_approval is True
    assert opts.human.verdict == HumanResponseVerdict.APPROVE.value


@pytest.mark.unit
@pytest.mark.gate
def test_task_hydrates_and_syncs_legacy_metadata():
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hello",
        metadata={
            "shadow_workspace": True,
            "human_decision": HumanResponseVerdict.APPROVE.value,
            "classification": "single_agent_default",
        },
    )
    assert task.options.isolation.shadow_workspace is True
    assert task.options.human.verdict == HumanResponseVerdict.APPROVE.value
    assert task.classification == "single_agent_default"

    task.options.isolation.sandbox = True
    sync_task_metadata(task)
    assert task.metadata["sandbox"] is True
    assert task.metadata["task_contract.v1"]["options"]["isolation"]["sandbox"] is True


@pytest.mark.unit
@pytest.mark.gate
def test_task_typed_human_input_roundtrip():
    task = Task(
        tenant_id="t1",
        user_id="u1",
        options=TaskExecutionOptions(
            human=TaskHumanInput(response_text="reject"),
        ),
    )
    HumanPauseCoordinator.record_human_response(task, "reject")
    assert task.options.human.verdict == HumanResponseVerdict.REJECT
    assert task.metadata["human_decision"] == "reject"


@pytest.mark.unit
@pytest.mark.gate
def test_task_typed_governance_options():
    task = Task(
        tenant_id="t1",
        user_id="u1",
        options=TaskExecutionOptions(
            governance=TaskGovernanceOptions(require_human_approval=True, high_risk=True),
            isolation=TaskIsolationOptions(shadow_workspace_cleanup=True),
        ),
    )
    sync_task_metadata(task)
    assert task.metadata["require_human_approval"] is True
    assert task.metadata["high_risk"] is True
    assert task.metadata["shadow_workspace_cleanup"] is True

    reloaded = Task.model_validate(task.model_dump(mode="json"))
    assert reloaded.options.governance.require_human_approval is True
    assert reloaded.options.isolation.shadow_workspace_cleanup is True


@pytest.mark.unit
@pytest.mark.gate
def test_task_to_runtime_request_propagates_workspace_id():
    task = Task(
        tenant_id="t1",
        user_id="u1",
        agent_id="agent-1",
        message="workspace request",
        metadata={"workspace_id": "workspace-a"},
    )

    request = task.to_runtime_request(run_id=mint_run_id())

    assert request.workspace_id == "workspace-a"

    task_without_workspace = Task(
        tenant_id="t1",
        user_id="u1",
        agent_id="agent-1",
        message="tenant request",
    )
    assert task_without_workspace.to_runtime_request(run_id=mint_run_id()).workspace_id is None


@pytest.mark.unit
@pytest.mark.gate
def test_hydrate_task_from_metadata_preserves_extra_keys():
    task = Task(
        tenant_id="t1",
        user_id="u1",
        metadata={"source": "runtime_request", "custom_flag": 42},
    )
    hydrate_task_from_metadata(task)
    assert task.metadata["custom_flag"] == 42
