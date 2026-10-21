# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P2-R2 — Research + Hosting Engine strict-vs-lab reachability (scoped)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.harness_registry_authority import (
    HarnessHostRegistryAuthorityError,
)
from intergrax.applications._shared.production_host_composition import (
    StrictProductionAsgiPlaceholder,
)
from intergrax.applications._shared.production_process_composition import (
    create_reference_production_process_composition,
)
from intergrax.hosting.errors import HostedApplicationConfigurationError
from intergrax.hosting.engine.runtime import invoke_application_factory
from research_application.host.main import create_app, run
from tests.qualification.prod_q._prod_q_r1_r7_q_p1_strict_reachability import (
    STRICT_PRODUCTION_ROOTS,
    repo_root,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_REPO = repo_root()


def test_p2_r2_research_uvicorn_module_app_is_strict_placeholder() -> None:
    from research_application.host import main as research_main

    assert isinstance(research_main.app, StrictProductionAsgiPlaceholder)


@pytest.mark.asyncio
async def test_p2_r2_research_strict_placeholder_fail_closed_on_request() -> None:
    placeholder = StrictProductionAsgiPlaceholder(application_package="research_application")
    with pytest.raises(HarnessHostRegistryAuthorityError, match="cannot serve STRICT production"):
        await placeholder({}, lambda: None, lambda _: None)  # type: ignore[arg-type]


def test_p2_r2_research_factory_entrypoint_requires_activated_composition() -> None:
    with pytest.raises(HarnessHostRegistryAuthorityError, match="STRICT production requires"):
        create_app()


def test_p2_r2_research_cli_run_without_reference_flag_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RESEARCH_REFERENCE_PRODUCTION", raising=False)
    with pytest.raises(SystemExit) as exc:
        run()
    assert exc.value.code == 1


def test_p2_r2_lkw_cli_run_without_reference_flag_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_workspace_application.host.main import run as lkw_run

    monkeypatch.delenv("LOCAL_WORKSPACE_REFERENCE_PRODUCTION", raising=False)
    with pytest.raises(SystemExit) as exc:
        lkw_run()
    assert exc.value.code == 1


def test_p2_r2_research_fresh_composition_fails_closed_for_process_app() -> None:
    from research_application.host.main import create_research_process_app

    composition = create_reference_production_process_composition()
    with pytest.raises(HarnessHostRegistryAuthorityError, match="no active traffic-serving"):
        create_research_process_app(process_composition=composition)


def test_p2_r2_hosting_engine_not_p1_strict_production_root() -> None:
    engine_roots = {
        (rel, sym)
        for rel, sym in STRICT_PRODUCTION_ROOTS
        if "hosting" in rel or sym == "HostedApplicationEngine"
    }
    assert engine_roots == frozenset()
    engine_path = _REPO / "intergrax" / "hosting" / "engine" / "engine.py"
    assert engine_path.is_file()
    assert not any(root[0].endswith("hosting/engine/engine.py") for root in STRICT_PRODUCTION_ROOTS)


@pytest.mark.asyncio
async def test_p2_r2_hosting_factory_rejects_non_runtime_before_work() -> None:
    from tests.unit.hosting._helpers import build_sample_context

    def bad_factory() -> object:
        return object()

    with pytest.raises(HostedApplicationConfigurationError, match="HostedApplicationRuntime"):
        await invoke_application_factory(bad_factory, build_sample_context())


@pytest.mark.asyncio
async def test_p2_r2_r1_lkw_hosted_runtime_rejects_unactivated_composition_before_work() -> None:
    """PROD-Q alias — canonical behavior owner: test_hosted_runtime (APP-HOST-8B)."""
    from applications.local_workspace_application.tests.hosting.test_hosted_runtime import (
        test_lkw_hosted_runtime_rejects_unactivated_composition_before_start,
    )

    await test_lkw_hosted_runtime_rejects_unactivated_composition_before_start()
