# © Artur Czarnecki. All rights reserved.

"""``new-context-bundle`` scaffold must emit a valid ContextPlugin skeleton."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
)
from intergrax.context.plugin import ContextPlugin, register_context_plugin
from intergrax.context.registry import (
    ContextPluginRegistry,
    clear_context_plugin_catalog,
    get_context_plugin,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.scaffold.cli import build_parser, main
from intergrax.scaffold.new_context_bundle import run_new_context_bundle

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _clear_catalog() -> None:
    clear_context_plugin_catalog()
    yield
    clear_context_plugin_catalog()


def _args(
    bundle_id: str,
    *,
    root: Path,
    force: bool = False,
    provider_id: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        bundle_id=bundle_id,
        provider_id=provider_id,
        root=root,
        force=force,
    )


def _provider_dir(root: Path, bundle_id: str) -> Path:
    return root / "intergrax" / "context" / "providers" / bundle_id


def _load_plugin_module(plugin_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assembly_request() -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace",
        run_id="run",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="uaep_turn",
        objective="placeholder",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )


def _load_generated(root: Path, bundle_id: str) -> ModuleType:
    return _load_plugin_module(
        _provider_dir(root, bundle_id) / "plugin.py",
        f"generated_context_plugin_{bundle_id}",
    )


def test_command_creates_expected_directory_and_files(tmp_path: Path) -> None:
    assert run_new_context_bundle(_args("acme_ctx", root=tmp_path)) == 0
    provider_dir = _provider_dir(tmp_path, "acme_ctx")
    assert {path.name for path in provider_dir.iterdir()} == {
        "plugin.py",
        "bundle.py",
        "USAGE.md",
    }


def test_generated_plugin_is_loadable(tmp_path: Path) -> None:
    assert run_new_context_bundle(_args("acme_ctx", root=tmp_path)) == 0
    module = _load_generated(tmp_path, "acme_ctx")
    plugin = module.AcmeCtxContextPlugin
    provider = module.AcmeCtxSourceProvider
    assert plugin is not None
    assert provider is not None
    assert isinstance(plugin, ContextPlugin)


def test_generated_plugin_metadata(tmp_path: Path) -> None:
    assert run_new_context_bundle(_args("acme_ctx", root=tmp_path)) == 0
    plugin = _load_generated(tmp_path, "acme_ctx").AcmeCtxContextPlugin
    assert plugin.plugin_id() == "acme_ctx"
    assert plugin.plugin_version().strip()
    assert plugin.plugin_description().strip()


def test_register_adds_exactly_one_provider(tmp_path: Path) -> None:
    assert run_new_context_bundle(_args("acme_ctx", root=tmp_path)) == 0
    plugin = _load_generated(tmp_path, "acme_ctx").AcmeCtxContextPlugin
    registry = ContextPluginRegistry()
    plugin.register(registry)
    assert len(registry.list_providers()) == 1


def test_provider_id_and_custom_source(tmp_path: Path) -> None:
    assert run_new_context_bundle(_args("acme_ctx", root=tmp_path)) == 0
    module = _load_generated(tmp_path, "acme_ctx")
    registry = ContextPluginRegistry()
    module.AcmeCtxContextPlugin.register(registry)
    provider = registry.list_providers()[0]
    assert provider.provider_id == "acme_ctx.source"
    assert ContextFragmentSource.CUSTOM in provider.supported_sources


def test_collect_returns_valid_context_fragment(tmp_path: Path) -> None:
    assert run_new_context_bundle(_args("acme_ctx", root=tmp_path)) == 0
    module = _load_generated(tmp_path, "acme_ctx")
    registry = ContextPluginRegistry()
    module.AcmeCtxContextPlugin.register(registry)
    fragments = asyncio.run(
        registry.list_providers()[0].collect(_assembly_request(), ContextProviderContext())
    )
    assert len(fragments) == 1
    fragment = fragments[0]
    assert isinstance(fragment, ContextFragment)
    assert fragment.source is ContextFragmentSource.CUSTOM
    assert fragment.content
    assert fragment.token_estimate >= 0


def test_provider_id_override(tmp_path: Path) -> None:
    args = _args("acme_ctx", root=tmp_path, provider_id="acme_ctx.custom_source")
    assert run_new_context_bundle(args) == 0
    module = _load_generated(tmp_path, "acme_ctx")
    registry = ContextPluginRegistry()
    module.AcmeCtxContextPlugin.register(registry)
    assert registry.list_providers()[0].provider_id == "acme_ctx.custom_source"


def test_existing_directory_without_force_fails(tmp_path: Path) -> None:
    provider_dir = _provider_dir(tmp_path, "acme_ctx")
    provider_dir.mkdir(parents=True)
    (provider_dir / "keep_me.txt").write_text("owned", encoding="utf-8")
    assert run_new_context_bundle(_args("acme_ctx", root=tmp_path)) == 1
    assert (provider_dir / "keep_me.txt").read_text(encoding="utf-8") == "owned"
    assert not (provider_dir / "plugin.py").exists()


def test_force_overwrites_scaffold_files_only(tmp_path: Path) -> None:
    provider_dir = _provider_dir(tmp_path, "acme_ctx")
    provider_dir.mkdir(parents=True)
    (provider_dir / "keep_me.txt").write_text("owned", encoding="utf-8")
    (provider_dir / "plugin.py").write_text("stale", encoding="utf-8")
    assert run_new_context_bundle(_args("acme_ctx", root=tmp_path, force=True)) == 0
    assert (provider_dir / "keep_me.txt").read_text(encoding="utf-8") == "owned"
    plugin_source = (provider_dir / "plugin.py").read_text(encoding="utf-8")
    assert plugin_source != "stale"
    assert "class AcmeCtxContextPlugin" in plugin_source


def test_empty_normalized_bundle_id_fails(tmp_path: Path) -> None:
    assert run_new_context_bundle(_args("   ", root=tmp_path)) == 1
    assert run_new_context_bundle(_args("", root=tmp_path)) == 1
    assert run_new_context_bundle(_args("---", root=tmp_path)) == 1
    assert not (tmp_path / "intergrax").exists()


def test_cli_parser_exposes_new_context_bundle() -> None:
    parser = build_parser()
    args = parser.parse_args(["new-context-bundle", "acme_ctx"])
    assert args.command == "new-context-bundle"
    assert args.bundle_id == "acme_ctx"


def test_local_register_context_plugin_catalog(tmp_path: Path) -> None:
    assert run_new_context_bundle(_args("acme_ctx", root=tmp_path)) == 0
    plugin = _load_generated(tmp_path, "acme_ctx").AcmeCtxContextPlugin
    register_context_plugin(plugin)
    entry = get_context_plugin("acme_ctx")
    registry = ContextPluginRegistry()
    entry.register_into(registry)
    assert len(registry.list_providers()) == 1


def test_usage_documents_entry_point_and_profile(tmp_path: Path) -> None:
    assert run_new_context_bundle(_args("acme_ctx", root=tmp_path)) == 0
    usage = (_provider_dir(tmp_path, "acme_ctx") / "USAGE.md").read_text(encoding="utf-8")
    assert '[project.entry-points."intergrax.context"]' in usage
    assert "acme_ctx = \"intergrax.context.providers.acme_ctx.plugin:AcmeCtxContextPlugin\"" in usage
    assert "from intergrax.context.plugin import register_context_plugin" in usage
    assert "register_context_plugin(AcmeCtxContextPlugin)" in usage
    assert "ContextProfile(" in usage
    assert 'context_plugin_ids=["acme_ctx"]' in usage
    assert "installed ≠ enabled" in usage
    assert "discovery enabled" in usage.lower() or "discovery must" in usage.lower() or "Discovery enabled" in usage


def test_hyphen_normalizes_and_preserves_plugin_id_dots(tmp_path: Path) -> None:
    assert run_new_context_bundle(_args("Acme-Ctx", root=tmp_path)) == 0
    plugin = _load_generated(tmp_path, "acme_ctx").AcmeCtxContextPlugin
    assert plugin.plugin_id() == "acme_ctx"


def test_cli_main_dispatches_new_context_bundle(tmp_path: Path) -> None:
    code = main(["new-context-bundle", "acme_ctx", "--root", str(tmp_path)])
    assert code == 0
    assert (_provider_dir(tmp_path, "acme_ctx") / "plugin.py").is_file()
