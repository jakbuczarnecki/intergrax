# © Artur Czarnecki. All rights reserved.

"""EBH-5 Package C v2 — dual-slug integration proofs (qualification integrity)."""

from __future__ import annotations

from pathlib import Path

from intergrax.integrations.contracts.base import IntegrationCategory
from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import IntegrationDualSlugCatalogEntry
from tests.qualification.ebh_5._package_c_v2_primitives import (
    assert_unregistered_fail_closed,
    require_python_module,
    run_dual_slug_resolve_from_profile,
)


def run_package_c_v2_relational_sqlite_duckdb_dual_slug(tmp_path: Path) -> None:
    require_python_module("duckdb", qualification_context="integration.relational_store")
    run_dual_slug_resolve_from_profile(
        category=IntegrationCategory.RELATIONAL_STORE,
        slug_a="sqlite",
        slug_b="duckdb",
        tmp_path=tmp_path,
    )


def run_package_c_v2_relational_unregistered_fail_closed() -> None:
    assert_unregistered_fail_closed(category=IntegrationCategory.RELATIONAL_STORE)


def run_package_c_v2_dual_slug_for_entry(
    entry: IntegrationDualSlugCatalogEntry,
    *,
    tmp_path: Path | None = None,
) -> None:
    if entry.category is IntegrationCategory.RELATIONAL_STORE:
        assert tmp_path is not None
    if entry.slug_a == "duckdb" or entry.slug_b == "duckdb":
        require_python_module("duckdb", qualification_context=entry.concern_id)
    if entry.slug_a == "chroma" or entry.slug_b == "chroma":
        require_python_module("chromadb", qualification_context=entry.concern_id)
    run_dual_slug_resolve_from_profile(
        category=entry.category,
        slug_a=entry.slug_a,
        slug_b=entry.slug_b,
        tmp_path=tmp_path,
    )


def run_package_c_v2_unregistered_for_entry(entry: IntegrationDualSlugCatalogEntry) -> None:
    assert_unregistered_fail_closed(category=entry.category)
