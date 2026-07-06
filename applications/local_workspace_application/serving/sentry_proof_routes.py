# © Artur Czarnecki. All rights reserved.

"""Local/dev-only LKW Sentry observability proof routes (LKW-OBS-SENTRY-1)."""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel

from intergrax.fastapi_core.config import ApiEnvironment
from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy
from intergrax.runtime.observability.operator_wiring import (
    ObservabilityExportBackendRegistry,
    ObservabilityExportOperatorConfig,
    build_observability_export_integration,
)
from intergrax.runtime.observability.problem_reporter import ProblemReportContext, report_problem
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

_FORBIDDEN_EXPORT_SAMPLES = (
    "secret prompt",
    "raw body",
    "secret-api-key",
    "tool_arguments",
    "raw_chunks",
    "/home/user/",
    "c:\\users\\",
)

_PROOF_PROBLEM_KIND = "lkw.proof_controlled_failure"
_PROOF_ERROR_CODE = "LKW_PROOF_CONTROLLED_FAILURE"


class LocalWorkspaceSentryProofRequestV1(BaseModel):
    run_id: str = ""
    correlation_id: str = ""


class LocalWorkspaceSentryProofResponseV1(BaseModel):
    proof_result: Literal["PASS", "FAIL"]
    backend: Literal["sentry"] = "sentry"
    problem_kind: str = _PROOF_PROBLEM_KIND
    problem_error_code: str = _PROOF_ERROR_CODE
    run_id: str
    correlation_id: str
    safety_check: Literal["passed", "failed"]


def _proof_endpoint_enabled(settings: LocalWorkspaceBackendSettings) -> bool:
    return settings.environment in {ApiEnvironment.DEV, ApiEnvironment.STAGE}


async def emit_local_workspace_sentry_proof_failure(
    *,
    settings: LocalWorkspaceBackendSettings,
    observability_export: ObservabilityExportOperatorConfig | None,
    run_id: str,
    correlation_id: str,
    registry: ObservabilityExportBackendRegistry | None = None,
) -> LocalWorkspaceSentryProofResponseV1:
    """Emit a controlled LKW problem through platform observability export wiring."""
    config = observability_export
    if config is None:
        try:
            config = settings.build_observability_export_config()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"sentry_proof_config_error: {exc.__class__.__name__}",
            ) from exc

    if config is None or not config.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="sentry_proof_disabled: observability export is disabled",
        )
    if config.backend_id != "sentry":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"sentry_proof_backend_mismatch: expected sentry, got {config.backend_id}",
        )

    integration = build_observability_export_integration(config, registry=registry)
    context = ProblemReportContext(
        run_id=run_id,
        agent_id=settings.default_agent_id,
        capability="local.workspace.proof",
        correlation_id=correlation_id,
    )
    result = await report_problem(
        context=context,
        problem_kind=_PROOF_PROBLEM_KIND,
        error_code=_PROOF_ERROR_CODE,
        source_layer="lkw",
        source_component="sentry_proof_endpoint",
        exporter=integration,
        policy=ObservabilityExportPolicy(enabled=True, export_content=False),
    )

    if not result.exported or result.envelope is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="sentry_proof_export_failed",
        )

    serialized = result.envelope.model_dump_json()
    safety_passed = all(sample not in serialized.lower() for sample in _FORBIDDEN_EXPORT_SAMPLES)
    if not safety_passed:
        return LocalWorkspaceSentryProofResponseV1(
            proof_result="FAIL",
            run_id=run_id,
            correlation_id=correlation_id,
            safety_check="failed",
        )

    return LocalWorkspaceSentryProofResponseV1(
        proof_result="PASS",
        run_id=run_id,
        correlation_id=correlation_id,
        safety_check="passed",
    )


def mount_local_workspace_sentry_proof_routes(
    app: FastAPI,
    *,
    settings: LocalWorkspaceBackendSettings,
    observability_export: ObservabilityExportOperatorConfig | None,
    prefix: str,
) -> None:
    """Mount local/dev-only controlled Sentry proof routes."""
    if not _proof_endpoint_enabled(settings):
        return

    router = APIRouter(prefix=prefix, tags=["local_workspace_proof"])

    @router.post("/proof/sentry-error", response_model=LocalWorkspaceSentryProofResponseV1)
    async def sentry_error_proof(
        body: LocalWorkspaceSentryProofRequestV1 | None = None,
    ) -> LocalWorkspaceSentryProofResponseV1:
        request = body or LocalWorkspaceSentryProofRequestV1()
        run_id = request.run_id.strip() or f"lkw-sentry-proof-{uuid.uuid4().hex[:12]}"
        correlation_id = request.correlation_id.strip() or f"corr-{uuid.uuid4().hex[:12]}"
        return await emit_local_workspace_sentry_proof_failure(
            settings=settings,
            observability_export=observability_export,
            run_id=run_id,
            correlation_id=correlation_id,
        )

    app.include_router(router)
