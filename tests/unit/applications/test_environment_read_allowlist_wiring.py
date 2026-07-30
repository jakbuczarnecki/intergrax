# © Artur Czarnecki. All rights reserved.

"""Shared environment wiring — read allowlist root merge semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.environment_wiring import (
    _merge_integration_read_allowlist_roots,
)
from intergrax.applications._shared.harness_host_runtime import (
    build_harness_host_runtime,
)
from intergrax.integrations.registry.catalog_manifests import OTEL, SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _profile_with_read_roots(*blocks: tuple[str, list[str]]) -> IntegrationProfile:
    options: dict[str, dict[str, object]] = {OTEL.slug: {}, SQLITE.slug: {}}
    for slug, roots in blocks:
        options[slug] = {"allowed_read_roots": roots}
    return IntegrationProfile(relational_store=SQLITE, options=options)


def test_merge_single_option_block_contributes_roots() -> None:
    ctx = ToolWiringContext()
    profile = _profile_with_read_roots(("block_a", ["/data/a"]))
    merged = _merge_integration_read_allowlist_roots(ctx, profile)
    assert merged.read_allowlist_roots == frozenset({"/data/a"})


def test_merge_two_option_blocks_preserve_both_root_sets() -> None:
    ctx = ToolWiringContext()
    profile = _profile_with_read_roots(
        ("block_a", ["/data/a"]),
        ("block_b", ["/data/b"]),
    )
    merged = _merge_integration_read_allowlist_roots(ctx, profile)
    assert merged.read_allowlist_roots == frozenset({"/data/a", "/data/b"})


def test_merge_preserves_existing_context_roots() -> None:
    ctx = ToolWiringContext(read_allowlist_roots=frozenset({"/existing"}))
    profile = _profile_with_read_roots(("block_a", ["/data/a"]))
    merged = _merge_integration_read_allowlist_roots(ctx, profile)
    assert merged.read_allowlist_roots == frozenset({"/existing", "/data/a"})


def test_merge_deduplicates_duplicate_roots() -> None:
    ctx = ToolWiringContext(read_allowlist_roots=frozenset({"/shared"}))
    profile = _profile_with_read_roots(
        ("block_a", ["/shared", "/data/a"]),
        ("block_b", ["/shared", "/data/b"]),
    )
    merged = _merge_integration_read_allowlist_roots(ctx, profile)
    assert merged.read_allowlist_roots == frozenset({"/shared", "/data/a", "/data/b"})


def test_merge_without_option_roots_keeps_existing_context() -> None:
    ctx = ToolWiringContext(read_allowlist_roots=frozenset({"/existing"}))
    profile = IntegrationProfile(relational_store=SQLITE, options={OTEL.slug: {}})
    merged = _merge_integration_read_allowlist_roots(ctx, profile)
    assert merged.read_allowlist_roots == frozenset({"/existing"})


def test_merge_ignores_malformed_string_allowed_read_roots() -> None:
    ctx = ToolWiringContext(read_allowlist_roots=frozenset({"/existing"}))
    profile = IntegrationProfile(
        relational_store=SQLITE,
        options={OTEL.slug: {"allowed_read_roots": "/not-a-list"}},
    )
    merged = _merge_integration_read_allowlist_roots(ctx, profile)
    assert merged.read_allowlist_roots == frozenset({"/existing"})


def test_lkw_factory_runtime_includes_staging_and_user_read_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_home = tmp_path / "factory-data"
    user_docs = tmp_path / "user-docs"
    data_home.mkdir()
    user_docs.mkdir()
    monkeypatch.setenv("LKW_DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(user_docs.resolve()))

    settings = LocalWorkspaceBackendSettings.from_env()
    env = build_local_workspace_environment_profile(settings)
    runtime = build_harness_host_runtime(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        env,
        settings=settings,
    )
    roots = runtime.env_wiring.tool_wiring.wiring_context.read_allowlist_roots
    assert settings.managed_upload_staging_dir in roots
    assert settings.web_url_staging_dir in roots
    assert str(user_docs.resolve()) in roots


def test_lkw_factory_runtime_merges_multiple_option_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_home = tmp_path / "factory-data"
    data_home.mkdir()
    monkeypatch.setenv("LKW_DATA_HOME", str(data_home))

    settings = LocalWorkspaceBackendSettings.from_env()
    env = build_local_workspace_environment_profile(settings)
    integration = env.integration_profile.model_copy(
        update={
            "options": {
                **env.integration_profile.options,
                "extra_host": {"allowed_read_roots": ["/extra/read/root"]},
            }
        }
    )
    env = env.model_copy(update={"integration_profile": integration})
    runtime = build_harness_host_runtime(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        env,
        settings=settings,
    )
    roots = runtime.env_wiring.tool_wiring.wiring_context.read_allowlist_roots
    assert settings.managed_upload_staging_dir in roots
    assert settings.web_url_staging_dir in roots
    assert "/extra/read/root" in roots
