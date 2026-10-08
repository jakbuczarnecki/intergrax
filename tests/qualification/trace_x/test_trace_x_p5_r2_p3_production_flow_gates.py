# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3 mechanical production-flow qualification gates."""

from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

TRACE_X_P5_R2_P3_START_HEAD = "1b3151eb76e850e71872e10c0860bcc95d2e2831"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_txp5r2p3_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_R2_P3_START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp5r2p3_q02_configured_fulfillment_port_exists() -> None:
    mod = importlib.import_module(
        "intergrax.contracts.autonomous_work.worker_configured_capability_fulfillment",
    )
    assert hasattr(mod, "WorkerConfiguredCapabilityFulfillmentPort")


def test_txp5r2p3_q03_configured_service_depends_on_opportunity_read() -> None:
    from intergrax.autonomous_work.worker_configured_capability_fulfillment_service import (
        WorkerConfiguredCapabilityFulfillmentService,
    )

    params = inspect.signature(WorkerConfiguredCapabilityFulfillmentService.__init__).parameters
    assert "opportunity_read" in params


def test_txp5r2p3_q04_configured_service_depends_on_realization_port() -> None:
    from intergrax.autonomous_work.worker_configured_capability_fulfillment_service import (
        WorkerConfiguredCapabilityFulfillmentService,
    )

    params = inspect.signature(WorkerConfiguredCapabilityFulfillmentService.__init__).parameters
    assert "realization" in params


def test_txp5r2p3_q05_aw_service_has_no_governance_port_import() -> None:
    source = (
        _REPO_ROOT
        / "intergrax/autonomous_work/worker_configured_capability_fulfillment_service.py"
    ).read_text(encoding="utf-8")
    assert "ControlPlaneMutationAuthorizationPort" not in source


def test_txp5r2p3_q06_execution_bound_resolution_exists() -> None:
    mod = importlib.import_module("intergrax.integrations.execution_bound_integration_resolution")
    assert hasattr(mod, "ExecutionBoundIntegrationResolution")


def test_txp5r2p3_q07_pinning_store_protocol_dependency() -> None:
    from intergrax.integrations.execution_bound_integration_resolution import (
        ExecutionBoundIntegrationResolution,
    )

    params = inspect.signature(ExecutionBoundIntegrationResolution.__init__).parameters
    assert "pinning_store" in params


def test_txp5r2p3_q08_coordinator_configured_routing() -> None:
    source = (
        _REPO_ROOT / "intergrax/autonomous_work/worker_capability_fulfillment_coordinator.py"
    ).read_text(encoding="utf-8")
    assert "_fulfill_configure_existing" in source
    assert "CONFIGURE_EXISTING_REQUIRED" in source


def test_txp5r2p3_q09_no_category_value_provider_fallback_in_resolution() -> None:
    source = (
        _REPO_ROOT / "intergrax/integrations/execution_bound_integration_resolution.py"
    ).read_text(encoding="utf-8")
    assert "category.value" not in source


def test_txp5r2p3_q10_runtime_pinning_before_handler() -> None:
    source = (
        _REPO_ROOT
        / "intergrax/runtime/execution/qualified_capability_execution_runtime_delegate.py"
    ).read_text(encoding="utf-8")
    pin_index = source.index("pin_configured_adoption_for_execution")
    handler_index = source.index("handler.dispatch_once")
    assert pin_index < handler_index
