# © Artur Czarnecki. All rights reserved.

"""Governed Contractor Collaborative Work integration profile guard (DEBT-2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from governed_contractor_application.host.collaborative_work_integration_profile import (
    resolve_governed_contractor_collaborative_work_integration_profile,
)
from governed_contractor_application.manifest import build_governed_contractor_manifest
from intergrax.integrations.registry.catalog_manifests import POSTGRESQL, SQLITE

pytestmark = pytest.mark.unit


def test_collaborative_work_profile_unchanged_without_trace_db_path() -> None:
    manifest = build_governed_contractor_manifest()
    profile = manifest.integration_profile

    result = resolve_governed_contractor_collaborative_work_integration_profile(
        manifest,
        trace_db_path=None,
    )

    assert result is profile


def test_collaborative_work_profile_applies_sqlite_options_when_sqlite_active(
    tmp_path: Path,
) -> None:
    manifest = build_governed_contractor_manifest()
    trace_db_path = tmp_path / "trace.db"

    result = resolve_governed_contractor_collaborative_work_integration_profile(
        manifest,
        trace_db_path=trace_db_path,
    )

    sqlite_options = result.options[SQLITE.slug]
    assert sqlite_options["data_dir"] == str(tmp_path)
    assert sqlite_options["relational_db"] == str(tmp_path / "collaborative_work.db")


def test_collaborative_work_profile_preserves_existing_sqlite_options(tmp_path: Path) -> None:
    manifest = build_governed_contractor_manifest()
    custom_data_dir = str(tmp_path / "custom-data")
    custom_relational_db = str(tmp_path / "custom-data" / "existing.db")
    updated_profile = manifest.integration_profile.model_copy(
        update={
            "options": {
                **manifest.integration_profile.options,
                SQLITE.slug: {
                    "data_dir": custom_data_dir,
                    "relational_db": custom_relational_db,
                },
            },
        },
    )
    manifest = manifest.model_copy(update={"integration_profile": updated_profile})
    trace_db_path = tmp_path / "trace.db"

    result = resolve_governed_contractor_collaborative_work_integration_profile(
        manifest,
        trace_db_path=trace_db_path,
    )

    sqlite_options = result.options[SQLITE.slug]
    assert sqlite_options["data_dir"] == custom_data_dir
    assert sqlite_options["relational_db"] == custom_relational_db


def test_collaborative_work_profile_unchanged_for_non_sqlite_relational_provider(
    tmp_path: Path,
) -> None:
    manifest = build_governed_contractor_manifest()
    postgres_profile = manifest.integration_profile.model_copy(
        update={"relational_store": POSTGRESQL},
    )
    manifest = manifest.model_copy(update={"integration_profile": postgres_profile})
    profile = manifest.integration_profile
    trace_db_path = tmp_path / "trace.db"

    result = resolve_governed_contractor_collaborative_work_integration_profile(
        manifest,
        trace_db_path=trace_db_path,
    )

    assert result is profile
    assert SQLITE.slug not in result.options


def test_collaborative_work_profile_unchanged_for_custom_relational_slug(tmp_path: Path) -> None:
    manifest = build_governed_contractor_manifest()
    custom_profile = manifest.integration_profile.model_copy(
        update={"relational_store": "duckdb"},
    )
    manifest = manifest.model_copy(update={"integration_profile": custom_profile})
    profile = manifest.integration_profile
    trace_db_path = tmp_path / "trace.db"

    result = resolve_governed_contractor_collaborative_work_integration_profile(
        manifest,
        trace_db_path=trace_db_path,
    )

    assert result is profile
    assert SQLITE.slug not in result.options
