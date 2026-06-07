# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.integrations.contracts.feature_flag import FeatureFlagBackend
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.tools.providers.platform.contracts import (
    PlatformEvaluateFeatureFlagInput,
    PlatformFeatureFlagOutput,
    PlatformGetSecretInput,
    PlatformGetSecretOutput,
    PlatformGetWorkflowRunInput,
    PlatformListCheckSuitesInput,
    PlatformListCheckSuitesOutput,
    PlatformDeleteSecretInput,
    PlatformDeleteSecretOutput,
    PlatformPutSecretInput,
    PlatformPutSecretOutput,
    PlatformCheckSuiteOutput,
    PlatformWorkflowRunOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

PLATFORM_GET_SECRET_TOOL_ID = "platform.get_secret"
PLATFORM_PUT_SECRET_TOOL_ID = "platform.put_secret"
PLATFORM_DELETE_SECRET_TOOL_ID = "platform.delete_secret"
PLATFORM_EVALUATE_FEATURE_FLAG_TOOL_ID = "platform.evaluate_feature_flag"
PLATFORM_GET_WORKFLOW_RUN_TOOL_ID = "platform.get_workflow_run"
PLATFORM_LIST_CHECK_SUITES_TOOL_ID = "platform.list_check_suites"


def platform_get_secret(ctx: ToolWiringContext, params: PlatformGetSecretInput) -> PlatformGetSecretOutput:
    store: SecretsStore | None = ctx.secrets_store
    if store is None:
        raise RuntimeError("secrets_store_not_configured")
    value = store.get_secret(params.path.strip(), version=params.version)
    return PlatformGetSecretOutput(path=params.path.strip(), value=value)


def platform_put_secret(ctx: ToolWiringContext, params: PlatformPutSecretInput) -> PlatformPutSecretOutput:
    store: SecretsStore | None = ctx.secrets_store
    if store is None:
        raise RuntimeError("secrets_store_not_configured")
    store.put_secret(params.path.strip(), params.value)
    return PlatformPutSecretOutput(path=params.path.strip(), stored=True)


def platform_delete_secret(ctx: ToolWiringContext, params: PlatformDeleteSecretInput) -> PlatformDeleteSecretOutput:
    store: SecretsStore | None = ctx.secrets_store
    if store is None:
        raise RuntimeError("secrets_store_not_configured")
    store.delete_secret(params.path.strip())
    return PlatformDeleteSecretOutput(path=params.path.strip(), deleted=True)


def platform_evaluate_feature_flag(
    ctx: ToolWiringContext,
    params: PlatformEvaluateFeatureFlagInput,
) -> PlatformFeatureFlagOutput:
    backend: FeatureFlagBackend | None = ctx.feature_flag_backend
    if backend is None:
        raise RuntimeError("feature_flag_backend_not_configured")
    evaluation = backend.evaluate(
        params.flag_key.strip(),
        tenant_id=params.tenant_id.strip(),
        user_id=params.user_id,
    )
    return PlatformFeatureFlagOutput(
        key=evaluation.key,
        enabled=evaluation.enabled,
        variant=evaluation.variant,
        metadata=dict(evaluation.metadata),
    )


def platform_get_workflow_run(
    ctx: ToolWiringContext,
    params: PlatformGetWorkflowRunInput,
) -> PlatformWorkflowRunOutput:
    backend: CiCdBackend | None = ctx.ci_cd_backend
    if backend is None:
        raise RuntimeError("ci_cd_backend_not_configured")
    record = backend.get_workflow_run(params.run_id.strip())
    return PlatformWorkflowRunOutput(
        id=record.id,
        name=record.name,
        status=record.status,
        conclusion=record.conclusion,
        url=record.url,
    )


def platform_list_check_suites(
    ctx: ToolWiringContext,
    params: PlatformListCheckSuitesInput,
) -> PlatformListCheckSuitesOutput:
    backend: CiCdBackend | None = ctx.ci_cd_backend
    if backend is None:
        raise RuntimeError("ci_cd_backend_not_configured")
    suites = [
        PlatformCheckSuiteOutput(
            id=item.id,
            name=item.name,
            status=item.status,
            conclusion=item.conclusion,
            url=item.url,
        )
        for item in backend.list_check_suites(ref=params.ref.strip(), limit=params.limit)
    ]
    return PlatformListCheckSuitesOutput(ref=params.ref.strip(), suites=suites, total=len(suites))
