# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-C15 — dependency admission boundary host shutdown regression."""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from intergrax.applications._shared.declarative_tool_wiring import (
    build_declarative_invoker_from_tool_wiring,
    close_declarative_catalog_tool_invoker,
)
from intergrax.applications._shared.harness_host_runtime import close_harness_host_runtime
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications._shared.workspace_cleanup_wiring import (
    build_factory_lifespans,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.runtime.execution.application_declarative_tool_composition import (
    CatalogDeclarativeToolInvoker,
)
from intergrax.runtime.resilience.dependency_attempt_boundary_composition import (
    materialize_tool_dependency_attempt_boundary,
)
from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionRequest,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
)
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
    DependencyAttemptExecutionBoundaryClosedError,
)
from intergrax.runtime.wiring.agent_runtime_governance_factory import (
    build_agent_runtime_governance_boundary,
)
from intergrax.tools.contracts.tool_profile import ToolProfile
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.dependency_concurrency_admission_config import (
    tool_dependency_concurrency_admission_configuration,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _strict_declarative_invoker() -> tuple[
    CatalogDeclarativeToolInvoker,
    DependencyAttemptExecutionBoundary,
]:
    admission = tool_dependency_concurrency_admission_configuration(
        "qualification-http-tool",
        max_concurrent_calls=1,
    )
    boundary = materialize_tool_dependency_attempt_boundary(
        admission,
        production_mode=True,
    )
    assert boundary is not None
    tool_wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=["qualification-http-tool"]),
        wiring_context=ToolWiringContext(),
        registry=ToolRegistry(),
    )
    governance = build_agent_runtime_governance_boundary(capability_grants=())
    invoker = build_declarative_invoker_from_tool_wiring(
        tool_wiring,
        agent_runtime_governance=governance,
        dependency_attempt_boundary=boundary,
        production_mode=True,
    )
    assert invoker is not None
    return invoker, boundary


def _assert_boundary_admission_thread_alive(
    boundary: DependencyAttemptExecutionBoundary,
) -> None:
    assert boundary._thread.is_alive()


def _admission_request() -> DependencyConcurrencyAdmissionRequest:
    return DependencyConcurrencyAdmissionRequest(
        dependency=DependencyConcurrencyIdentity(
            kind=DependencyConcurrencyKind.TOOL,
            value="qualification-http-tool",
        ),
        tenant_id="c15-lifecycle",
    )


def _assert_boundary_closed(boundary: DependencyAttemptExecutionBoundary) -> None:
    assert not boundary._thread.is_alive()
    with pytest.raises(DependencyAttemptExecutionBoundaryClosedError):
        boundary.acquire(_admission_request())


def test_close_declarative_catalog_tool_invoker_stops_admission_thread() -> None:
    invoker, boundary = _strict_declarative_invoker()
    _assert_boundary_admission_thread_alive(boundary)
    try:
        close_declarative_catalog_tool_invoker(invoker)
        _assert_boundary_closed(boundary)
        close_declarative_catalog_tool_invoker(invoker)
        _assert_boundary_closed(boundary)
    except Exception:
        invoker.tool_invoker.close()
        raise


def test_close_declarative_catalog_tool_invoker_none_is_noop() -> None:
    close_declarative_catalog_tool_invoker(None)


def test_close_harness_host_runtime_stops_owned_declarative_admission_thread() -> None:
    from unittest.mock import MagicMock, patch

    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime

    invoker, boundary = _strict_declarative_invoker()
    runtime = HarnessHostRuntime(
        manifest=MagicMock(),
        environment=ApplicationEnvironmentProfile.product_defaults(
            profile_id="c15-lifecycle",
        ).model_copy(update={"execution_mode": ExecutionMode.STRICT}),
        tenant_id="c15-lifecycle",
        env_wiring=MagicMock(),
        registry=MagicMock(),
        observability=MagicMock(),
        reliability=MagicMock(),
        security=MagicMock(),
        guardrail=MagicMock(),
        cost=MagicMock(),
        evaluation=MagicMock(),
        diagnostic_wiring=MagicMock(),
        execution=MagicMock(),
        _internal_composition=MagicMock(),
        application_host=None,
        agent_checkpoint_store=MagicMock(),
        compensation_queue_store=MagicMock(),
        _owned_declarative_tool_invoker=invoker,
    )
    _assert_boundary_admission_thread_alive(boundary)
    try:
        with (
            patch(
                "intergrax.applications._shared.harness_host_composition.resolve_harness_host_event_bus",
                return_value=MagicMock(),
            ),
            patch(
                "intergrax.applications._shared.runtime_event_delivery_wiring.close_application_runtime_event_delivery",
            ),
        ):
            close_harness_host_runtime(runtime)
        _assert_boundary_closed(boundary)
    except Exception:
        invoker.tool_invoker.close()
        raise


def test_close_harness_host_runtime_continues_after_declarative_close_failure() -> None:
    from unittest.mock import MagicMock, patch

    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime

    invoker, boundary = _strict_declarative_invoker()
    runtime = HarnessHostRuntime(
        manifest=MagicMock(),
        environment=ApplicationEnvironmentProfile.product_defaults(
            profile_id="c15-lifecycle",
        ).model_copy(update={"execution_mode": ExecutionMode.STRICT}),
        tenant_id="c15-lifecycle",
        env_wiring=MagicMock(),
        registry=MagicMock(),
        observability=MagicMock(),
        reliability=MagicMock(),
        security=MagicMock(),
        guardrail=MagicMock(),
        cost=MagicMock(),
        evaluation=MagicMock(),
        diagnostic_wiring=MagicMock(),
        execution=MagicMock(),
        _internal_composition=MagicMock(),
        application_host=None,
        agent_checkpoint_store=MagicMock(),
        compensation_queue_store=MagicMock(),
        _owned_declarative_tool_invoker=invoker,
    )
    primary = RuntimeError("declarative close failed")
    event_delivery_close = MagicMock()
    try:
        with (
            patch(
                "intergrax.applications._shared.declarative_tool_wiring.close_declarative_catalog_tool_invoker",
                side_effect=primary,
            ),
            patch(
                "intergrax.applications._shared.harness_host_composition.resolve_harness_host_event_bus",
                return_value=MagicMock(),
            ),
            patch(
                "intergrax.applications._shared.runtime_event_delivery_wiring.close_application_runtime_event_delivery",
                event_delivery_close,
            ),
        ):
            with pytest.raises(RuntimeError, match="declarative close failed") as exc_info:
                close_harness_host_runtime(runtime)
        assert exc_info.value is primary
        event_delivery_close.assert_called_once()
    finally:
        if boundary._thread.is_alive():
            invoker.tool_invoker.close()


def test_factory_lifespan_closes_harness_host_runtime() -> None:
    from unittest.mock import MagicMock, patch

    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime

    runtime = MagicMock(spec=HarnessHostRuntime)
    app = FastAPI()
    lifespans = build_factory_lifespans(runtime, schedulers=[])
    assert lifespans
    from intergrax.applications._shared.fastapi_lifespan import combine_lifespans

    app.router.lifespan_context = combine_lifespans(*lifespans)

    from starlette.testclient import TestClient

    with patch.object(runtime, "close") as close_mock:
        with TestClient(app) as client:
            assert client.app is app
        close_mock.assert_called_once()
