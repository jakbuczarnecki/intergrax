# © Artur Czarnecki. All rights reserved.

"""ME-18 — final enterprise audit vertical and isolation qualification gates."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _me18_host_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter
    from testing_support.host_fixture_wiring import install_diagnostic_cursor_secret

    install_diagnostic_cursor_secret(monkeypatch)
    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def test_me18_agent_vertical_remains_exact_release(tmp_path: Path) -> None:
    from tests.integration.marketplace.test_me13_marketplace_agent_distribution_execution_e2e import (
        test_me13_exact_release_survives_catalog_version_drift,
    )

    test_me13_exact_release_survives_catalog_version_drift(tmp_path)


def test_me18_tool_vertical_remains_exact_release(tmp_path: Path) -> None:
    from tests.integration.marketplace.test_me14_marketplace_tool_execution_e2e import (
        test_me14_exact_tool_release_is_preserved_to_execution,
    )

    test_me14_exact_tool_release_is_preserved_to_execution(tmp_path)


def test_me18_skill_vertical_remains_exact_release() -> None:
    from tests.integration.marketplace.test_me15_marketplace_skill_composition_e2e import (
        test_me15_selected_skill_release_is_preserved_into_binding,
    )

    test_me15_selected_skill_release_is_preserved_into_binding()


def test_me18_mixed_agent_tool_skill_flow_is_green(tmp_path: Path) -> None:
    from tests.integration.marketplace.test_me17_marketplace_production_qualification import (
        test_me17_agent_tool_skill_mixed_flow_still_executes,
    )

    test_me17_agent_tool_skill_mixed_flow_still_executes(tmp_path, None)


def test_me18_machine_api_single_snapshot_is_green() -> None:
    from tests.unit.marketplace.test_me12_machine_capability_acquisition import (
        test_acquire_does_not_read_second_snapshot_for_completeness,
    )

    test_acquire_does_not_read_second_snapshot_for_completeness()


def test_me18_tenant_isolation_is_green() -> None:
    from tests.integration.marketplace.test_me17_marketplace_production_qualification import (
        test_me17_tenant_private_catalog_never_leaks_cross_tenant,
    )

    test_me17_tenant_private_catalog_never_leaks_cross_tenant()


def test_me18_organization_isolation_is_green() -> None:
    from tests.integration.marketplace.test_me17_marketplace_production_qualification import (
        test_me17_organization_private_catalog_never_leaks_cross_org,
    )

    test_me17_organization_private_catalog_never_leaks_cross_org()
