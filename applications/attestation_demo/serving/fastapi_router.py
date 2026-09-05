# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from intergrax.applications._shared.harness_auth import require_harness_api_key

from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id


class AttestationPocRunRequestV1(BaseModel):
    tenant_id: str = "default"
    user_id: str = "lab-user"
    session_id: Optional[str] = None
    message: str = Field(default="PoC attestation report", min_length=1)
    capability: str = Field(default="attestation.demo", min_length=1)
    partition_key: str = Field(default="attestation_demo", min_length=1)
    row_key: Optional[str] = None
    record_data: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttestationPocRunResponseV1(BaseModel):
    task_id: str
    run_id: Optional[str] = None
    state: str
    answer: str = ""
    agent_id: Optional[str] = None
    boundary_events: list[dict[str, Any]] = Field(default_factory=list)
    trust_model: dict[str, str] = Field(
        default_factory=lambda: {
            "platform_signed": "false",
            "recommended_receipt_role": "client_observed",
            "note": "Intergrax emits unsigned boundary facts; partner signs receipts locally.",
        }
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class AttestationPocRunService:
    host_execution: HostTaskExecutionPort
    boundary_event_buffer: BoundaryEventBuffer

    async def run_task(self, body: AttestationPocRunRequestV1) -> AttestationPocRunResponseV1:
        run_id = new_run_id()
        metadata = dict(body.metadata)
        metadata["partition_key"] = body.partition_key
        if body.row_key:
            metadata["row_key"] = body.row_key
        if body.record_data is not None:
            metadata["record_data"] = body.record_data
        task = Task(
            task_id=run_id,
            tenant_id=body.tenant_id,
            user_id=body.user_id,
            session_id=body.session_id,
            message=body.message,
            context=TaskContext(capability=body.capability),
            metadata=metadata,
        )
        result = await self.host_execution.execute(task)
        resolved_run_id = result.run_id or run_id
        boundary_events = self.boundary_event_buffer.snapshot_for_run(resolved_run_id)
        host_signed = any(event.get("signed") is True for event in boundary_events)
        trust_model = {
            "platform_signed": "true" if host_signed else "false",
            "recommended_receipt_role": "host_attested" if host_signed else "client_observed",
            "note": (
                "Intergrax emits host-signed boundary facts when EBE-9 is enabled; "
                "BoundaryAttest may add a separate client_observed wrapper after verification."
                if host_signed
                else "Intergrax emits unsigned boundary facts; partner signs receipts locally."
            ),
        }
        return AttestationPocRunResponseV1(
            task_id=result.task_id,
            run_id=resolved_run_id,
            state=result.state.value,
            answer=result.answer,
            agent_id=result.agent_id,
            boundary_events=boundary_events,
            trust_model=trust_model,
            metadata=dict(result.metadata),
        )


def mount_attestation_demo_routes(
    app: FastAPI,
    *,
    host_execution: HostTaskExecutionPort,
    registry: AgentRegistryRead,
    boundary_event_buffer: BoundaryEventBuffer,
    prefix: str = "/v1/attestation_demo",
) -> AttestationPocRunService:
    service = AttestationPocRunService(
        host_execution=host_execution,
        boundary_event_buffer=boundary_event_buffer,
    )
    router = APIRouter(prefix=prefix, tags=["attestation_demo"])

    @router.post(
        "/poc/run",
        response_model=AttestationPocRunResponseV1,
        dependencies=[Depends(require_harness_api_key)],
    )
    async def poc_run(body: AttestationPocRunRequestV1) -> AttestationPocRunResponseV1:
        try:
            return await service.run_task(body)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"run_error: {exc.__class__.__name__}",
            ) from exc

    @router.get(
        "/poc/runs/{run_id}/boundary-events",
        dependencies=[Depends(require_harness_api_key)],
    )
    async def list_boundary_events(run_id: str) -> dict[str, object]:
        events = boundary_event_buffer.snapshot_for_run(run_id)
        return {"run_id": run_id, "boundary_events": events, "count": len(events)}

    @router.get("/agents")
    async def list_agents() -> dict[str, list[dict[str, object]]]:
        agents: list[dict[str, object]] = []
        for agent_id in registry.list_agent_ids():
            contract = registry.get(agent_id).get_contract()
            agents.append(
                {
                    "agent_id": contract.id,
                    "name": contract.name,
                    "capabilities": list(contract.capabilities),
                    "allowed_tools": list(contract.allowed_tools),
                }
            )
        return {"agents": agents}

    app.include_router(router)
    return service
