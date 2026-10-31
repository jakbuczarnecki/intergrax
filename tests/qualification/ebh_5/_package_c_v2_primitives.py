# © Artur Czarnecki. All rights reserved.

"""Package C v2 — shared qualification primitives (fail-closed, no silent skips)."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    UnknownIntegrationError,
)
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.integrations.registry.profile import IntegrationProfile


def require_python_module(module_name: str, *, qualification_context: str) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{qualification_context}: mandatory dependency '{module_name}' missing — "
            f"sync environment with sanctioned dev-ci / integration extras"
        ) from exc


def assert_unregistered_fail_closed(*, category: IntegrationCategory) -> None:
    clear_catalog()
    with pytest.raises((UnknownIntegrationError, IntegrationConfigurationError)):
        resolve_from_profile(IntegrationProfile(), category)


def register_catalog_slug(*, category: IntegrationCategory, slug: str) -> None:
    module_path = f"intergrax.integrations.providers.{category.value}.{slug}.register"
    module = importlib.import_module(module_path)
    register_fn = getattr(module, f"register_{slug}_integration")
    register_fn()


def profile_for_slug(
    *,
    category: IntegrationCategory,
    slug: str,
    tmp_path: Path | None = None,
    extra_options: dict[str, Any] | None = None,
) -> IntegrationProfile:
    from intergrax.integrations.contracts.base import PROFILE_FIELD_BY_CATEGORY

    profile_field = PROFILE_FIELD_BY_CATEGORY[category.value]
    kwargs: dict[str, Any] = {profile_field: slug}
    options: dict[str, Any] = dict(extra_options or {})
    if category is IntegrationCategory.RELATIONAL_STORE and tmp_path is not None:
        options.setdefault(slug, {"data_dir": str(tmp_path / slug)})
    if options:
        kwargs["options"] = options
    return IntegrationProfile(**kwargs)


def exercise_connect_close_if_supported(integration: object) -> None:
    connect = getattr(integration, "connect", None)
    close = getattr(integration, "close", None)
    if callable(connect):
        connect()
    if callable(close):
        close()


def run_dual_slug_resolve_from_profile(
    *,
    category: IntegrationCategory,
    slug_a: str,
    slug_b: str,
    tmp_path: Path | None = None,
    exercise: Callable[[object], None] | None = None,
) -> tuple[object, object]:
    if category is IntegrationCategory.RELATIONAL_STORE:
        require_python_module("duckdb", qualification_context="integration.relational_store")
    clear_catalog()
    register_catalog_slug(category=category, slug=slug_a)
    register_catalog_slug(category=category, slug=slug_b)
    inst_a = resolve_from_profile(profile_for_slug(category=category, slug=slug_a, tmp_path=tmp_path), category)
    inst_b = resolve_from_profile(profile_for_slug(category=category, slug=slug_b, tmp_path=tmp_path), category)
    assert inst_a.provider_id != inst_b.provider_id
    runner = exercise or exercise_connect_close_if_supported
    runner(inst_a)
    runner(inst_b)
    return inst_a, inst_b
