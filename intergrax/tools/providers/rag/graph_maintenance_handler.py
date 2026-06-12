# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.rag.graph_maintenance_contracts import (
    RagScheduleGraphMaintenanceJobInput,
    RagScheduleGraphMaintenanceJobOutput,
)
from intergrax.tools.providers.rag.graph_maintenance_service import (
    perform_rag_schedule_graph_maintenance_job,
)


class RagScheduleGraphMaintenanceJobHandler(
    ServiceToolHandler[RagScheduleGraphMaintenanceJobInput, RagScheduleGraphMaintenanceJobOutput]
):
    _service = perform_rag_schedule_graph_maintenance_job
