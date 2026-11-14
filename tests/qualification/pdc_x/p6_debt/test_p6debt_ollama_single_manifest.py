# © Artur Czarnecki. All rights reserved.

"""B-P6-I item 3: ``embedding_provider/ollama`` has no manifest; the single ``ollama`` row owns the slug."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import intergrax.integrations.registry.bootstrap as bootstrap_module
import pytest
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.embedding_provider.ollama.register import (
    register_ollama_embedding_provider_integration,
)
from intergrax.integrations.providers.model_serving_runtime.ollama.register import register_ollama_integration
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import (
    catalog_snapshot,
    catalog_state_lock,
    clear_catalog,
    get_entry,
    list_slugs,
    read_catalog_revision_under_lock,
    register_integration,
    set_catalog_generation_for_tests,
)
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
_PACKAGE = REPO_ROOT / "intergrax" / "integrations" / "providers" / "embedding_provider" / "ollama"
_SOURCE_ROOTS = ("intergrax", "agents", "applications", "scripts")
_MANIFEST_IMPORT = re.compile(
    r"embedding_provider\.ollama\.manifest|embedding_provider\.ollama\s+import\s+[^\n]*\bmanifest\b"
)


@pytest.fixture()
def fresh_catalog() -> Iterator[None]:
    """Empty catalog and bootstrap flag; restore the previous process state afterwards."""
    with catalog_state_lock():
        saved_entries = catalog_snapshot()
        saved_generation = read_catalog_revision_under_lock().generation
    saved_bootstrapped = bootstrap_module._BOOTSTRAPPED
    clear_catalog()
    reset_default_integrations_state()
    try:
        yield
    finally:
        clear_catalog()
        for entry in saved_entries.values():
            register_integration(entry, override=True)
        set_catalog_generation_for_tests(saved_generation)
        bootstrap_module._BOOTSTRAPPED = saved_bootstrapped


def test_embedding_ollama_package_has_no_manifest_and_nothing_names_it() -> None:
    assert not (_PACKAGE / "manifest.py").exists(), "a second ollama manifest is a dead second owner of the slug"
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for root in _SOURCE_ROOTS
        for path in sorted((REPO_ROOT / root).rglob("*.py"))
        if "__pycache__" not in path.parts
        and _MANIFEST_IMPORT.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == []


def test_single_ollama_row_resolves_the_embedding_category(fresh_catalog: None) -> None:
    register_ollama_integration()
    register_ollama_embedding_provider_integration()

    assert list_slugs().count("ollama") == 1
    entry = get_entry("ollama")
    assert IntegrationCategory.MODEL_SERVING_RUNTIME in entry.categories
    assert IntegrationCategory.EMBEDDING_PROVIDER in entry.categories
    resolved = entry.factory(integration_category=IntegrationCategory.EMBEDDING_PROVIDER, enabled=False)
    assert isinstance(resolved, EmbeddingProviderIntegrationContract)
