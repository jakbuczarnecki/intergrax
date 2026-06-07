# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.harness.contracts import (
    HarnessCompareRunsInput,
    HarnessCompareRunsOutput,
    HarnessExportRunBundleInput,
    HarnessExportRunBundleOutput,
    HarnessGetRunCostInput,
    HarnessGetRunCostOutput,
    HarnessGetRunEventsInput,
    HarnessGetRunEventsOutput,
    HarnessGetRunInput,
    HarnessGetRunOutput,
    HarnessListRunsInput,
    HarnessListRunsOutput,
)
from intergrax.tools.providers.harness.service import (
    harness_compare_runs,
    harness_export_run_bundle,
    harness_get_run,
    harness_get_run_cost,
    harness_get_run_events,
    harness_list_runs,
)


class HarnessGetRunHandler(ServiceToolHandler[HarnessGetRunInput, HarnessGetRunOutput]):
    _service = harness_get_run


class HarnessListRunsHandler(ServiceToolHandler[HarnessListRunsInput, HarnessListRunsOutput]):
    _service = harness_list_runs


class HarnessGetRunCostHandler(ServiceToolHandler[HarnessGetRunCostInput, HarnessGetRunCostOutput]):
    _service = harness_get_run_cost


class HarnessGetRunEventsHandler(ServiceToolHandler[HarnessGetRunEventsInput, HarnessGetRunEventsOutput]):
    _service = harness_get_run_events


class HarnessCompareRunsHandler(ServiceToolHandler[HarnessCompareRunsInput, HarnessCompareRunsOutput]):
    _service = harness_compare_runs


class HarnessExportRunBundleHandler(
    ServiceToolHandler[HarnessExportRunBundleInput, HarnessExportRunBundleOutput]
):
    _service = harness_export_run_bundle
