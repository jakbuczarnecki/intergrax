# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-C15 — dependency admission boundary host shutdown regression."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator
from unittest.mock import patch

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
from tests.unit.runtime.nexus.tools.test_gr10_r9_orchestration_mse import (
    _RecordingMseBoundary,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _tool_wiring_for_qualification_http_tool() -> ApplicationToolWiring:
    return ApplicationToolWiring(
        profile=ToolProfile(enabled=["qualification-http-tool"]),
        wiring_context=ToolWiringContext(),
        registry=ToolRegistry(),
    )


def _materialize_strict_admission_boundary() -> DependencyAttemptExecutionBoundary:
    admission = tool_dependency_concurrency_admission_configuration(
        "qualification-http-tool",
        max_concurrent_calls=1,
    )
    boundary = materialize_tool_dependency_attempt_boundary(
        admission,
        production_mode=True,
    )
    assert boundary is not None
    return boundary


def _compose_strict_declarative_invoker(
    boundary: DependencyAttemptExecutionBoundary,
) -> CatalogDeclarativeToolInvoker:
    governance = build_agent_runtime_governance_boundary(capability_grants=())
    meaningful_side_effect_authorization = _RecordingMseBoundary(allow=True)
    invoker = build_declarative_invoker_from_tool_wiring(
        _tool_wiring_for_qualification_http_tool(),
        agent_runtime_governance=governance,
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
        dependency_attempt_boundary=boundary,
        production_mode=True,
    )
    assert invoker is not None
    return invoker


def _compose_strict_declarative_invoker_owned(
    boundary: DependencyAttemptExecutionBoundary,
) -> CatalogDeclarativeToolInvoker:
    try:
        return _compose_strict_declarative_invoker(boundary)
    except BaseException as composition_error:
        cleanup_error: BaseException | None = None
        try:
            boundary.close()
        except BaseException as exc:
            cleanup_error = exc
        if cleanup_error is not None:
            raise BaseExceptionGroup(
                "strict declarative invoker composition failed",
                (composition_error, cleanup_error),
            ) from None
        raise


def _strict_declarative_invoker() -> tuple[
    CatalogDeclarativeToolInvoker,
    DependencyAttemptExecutionBoundary,
]:
    boundary = _materialize_strict_admission_boundary()
    invoker = _compose_strict_declarative_invoker_owned(boundary)
    return invoker, boundary


@contextmanager
def _managed_strict_declarative_invoker() -> Iterator[
    tuple[CatalogDeclarativeToolInvoker, DependencyAttemptExecutionBoundary]
]:
    invoker, boundary = _strict_declarative_invoker()
    try:
        yield invoker, boundary
    finally:
        if boundary._thread.is_alive():
            close_declarative_catalog_tool_invoker(invoker)


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
    with _managed_strict_declarative_invoker() as (invoker, boundary):
        _assert_boundary_admission_thread_alive(boundary)
        close_declarative_catalog_tool_invoker(invoker)
        _assert_boundary_closed(boundary)
        close_declarative_catalog_tool_invoker(invoker)
        _assert_boundary_closed(boundary)


def test_close_declarative_catalog_tool_invoker_none_is_noop() -> None:
    close_declarative_catalog_tool_invoker(None)


def test_strict_declarative_invoker_closes_boundary_when_composition_fails() -> None:
    boundary = _materialize_strict_admission_boundary()
    _assert_boundary_admission_thread_alive(boundary)
    composition_error = ValueError("invoker composition failed")
    with patch(
        "intergrax.applications._shared.declarative_tool_wiring.build_production_runtime_tool_invoker",
        side_effect=composition_error,
    ):
        with pytest.raises(ValueError, match="invoker composition failed") as exc_info:
            _compose_strict_declarative_invoker_owned(boundary)
    assert exc_info.value is composition_error
    _assert_boundary_closed(boundary)


def test_close_harness_host_runtime_stops_owned_declarative_admission_thread() -> None:
    from unittest.mock import MagicMock, patch

    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime

    with _managed_strict_declarative_invoker() as (invoker, boundary):
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


def test_close_harness_host_runtime_continues_after_declarative_close_failure() -> None:
    from unittest.mock import MagicMock, patch

    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime

    with _managed_strict_declarative_invoker() as (invoker, boundary):
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


@dataclass
class _C15DisabledIsolationEnvWiring:
    shadow_manager: None = None
    sandbox_manager: None = None


@dataclass
class _C15FactoryLifespanHarnessHost:
    env_wiring: _C15DisabledIsolationEnvWiring = field(
        default_factory=_C15DisabledIsolationEnvWiring,
    )
    close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


def test_factory_lifespan_closes_harness_host_runtime() -> None:
    from intergrax.applications._shared.fastapi_lifespan import combine_lifespans
    from starlette.testclient import TestClient

    runtime = _C15FactoryLifespanHarnessHost()
    app = FastAPI()
    lifespans = build_factory_lifespans(runtime, schedulers=[])
    assert lifespans
    app.router.lifespan_context = combine_lifespans(*lifespans)

    with TestClient(app) as client:
        assert client.app is app
    assert runtime.close_calls == 1
