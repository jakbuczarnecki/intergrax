# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.workflow.contracts import (
    WorkflowFetchLogsInput,
    WorkflowFetchLogsOutput,
    WorkflowPollInput,
    WorkflowPollOutput,
    WorkflowTriggerInput,
    WorkflowTriggerOutput,
)
from intergrax.tools.providers.workflow.service import workflow_fetch_logs, workflow_poll, workflow_trigger


class WorkflowTriggerHandler(ServiceToolHandler[WorkflowTriggerInput, WorkflowTriggerOutput]):
    _service = workflow_trigger


class WorkflowPollHandler(ServiceToolHandler[WorkflowPollInput, WorkflowPollOutput]):
    _service = workflow_poll


class WorkflowFetchLogsHandler(ServiceToolHandler[WorkflowFetchLogsInput, WorkflowFetchLogsOutput]):
    _service = workflow_fetch_logs
