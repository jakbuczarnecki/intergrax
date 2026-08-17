# © Artur Czarnecki. All rights reserved.

"""APP-ADOPTION-1A: Tier-3 context catalog wiring evidence propagation."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.context_wiring import bootstrap_application_context_catalog
from intergrax.context.bootstrap import ContextCatalogBootstrapResult, reset_context_catalog_bootstrap_for_tests
from intergrax.core.plugins.discovery import EP_CONTEXT, load_entry_point_targets, reset_entry_point_spec_cache_for_tests

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset_catalog() -> None:
    reset_context_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_context_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()


def test_bootstrap_application_context_catalog_returns_domain_result() -> None:
    result = bootstrap_application_context_catalog(discover_entry_points=False)

    assert isinstance(result, ContextCatalogBootstrapResult)
    assert result.load_report.group == EP_CONTEXT


def test_bootstrap_application_context_catalog_preserves_domain_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = load_entry_point_targets

    def _count_calls(group: str, **kwargs: object) -> object:
        calls.append(group)
        return original(group, **kwargs)

    monkeypatch.setattr(
        "intergrax.context.bootstrap.load_entry_point_targets",
        _count_calls,
    )

    result = bootstrap_application_context_catalog(discover_entry_points=False)

    assert result.load_report.group == EP_CONTEXT
    assert calls == []
