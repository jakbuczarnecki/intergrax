# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
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
    PlatformWorkflowRunOutput,
)
from intergrax.tools.providers.platform.service import (
    platform_delete_secret,
    platform_evaluate_feature_flag,
    platform_get_secret,
    platform_get_workflow_run,
    platform_list_check_suites,
    platform_put_secret,
)


class PlatformGetSecretHandler(ServiceToolHandler[PlatformGetSecretInput, PlatformGetSecretOutput]):
    _service = platform_get_secret


class PlatformPutSecretHandler(ServiceToolHandler[PlatformPutSecretInput, PlatformPutSecretOutput]):
    _service = platform_put_secret


class PlatformDeleteSecretHandler(ServiceToolHandler[PlatformDeleteSecretInput, PlatformDeleteSecretOutput]):
    _service = platform_delete_secret


class PlatformEvaluateFeatureFlagHandler(
    ServiceToolHandler[PlatformEvaluateFeatureFlagInput, PlatformFeatureFlagOutput]
):
    _service = platform_evaluate_feature_flag


class PlatformGetWorkflowRunHandler(
    ServiceToolHandler[PlatformGetWorkflowRunInput, PlatformWorkflowRunOutput]
):
    _service = platform_get_workflow_run


class PlatformListCheckSuitesHandler(
    ServiceToolHandler[PlatformListCheckSuitesInput, PlatformListCheckSuitesOutput]
):
    _service = platform_list_check_suites
