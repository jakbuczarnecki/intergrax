# © Artur Czarnecki. All rights reserved.

"""Skill bundle catalog adapter contract tests (Stage 2)."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog.adapters.skill import (
    SKILL_BUILTIN_CATALOG_SOURCE_ID,
    SkillBundleCatalogSource,
    project_skill_bundle_entry,
)
from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceKind
from intergrax.contracts.capability_catalog.skill_version_binding import (
    SkillVersionBindingDisposition,
)
from intergrax.skills.providers.harness.manifests import HARNESS_TOOL_SMOKE
from intergrax.skills.registry.bootstrap import register_default_skills
from intergrax.skills.registry.catalog import SkillBundleEntry, clear_skill_catalog, register_skill_bundle
from intergrax.skills.registry.runtime import SkillRegistry

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_catalog() -> None:
    clear_skill_catalog()
    yield
    clear_skill_catalog()


def _bundle() -> SkillBundleEntry:
    def _register(registry: SkillRegistry) -> None:
        del registry

    return SkillBundleEntry(
        bundle_id="rag",
        skill_ids=("skills.rag.search", "skills.rag.rank"),
        register=_register,
        description="RAG skills",
    )


def test_project_skill_bundle_entry_preserves_skill_identity_and_bundle_provenance() -> None:
    projected = project_skill_bundle_entry(_bundle(), "skills.rag.search")
    assert projected.identity.kind is CapabilityKind.SKILL
    assert projected.identity.logical.logical_id == "skills.rag.search"
    assert projected.identity.source.source_id == SKILL_BUILTIN_CATALOG_SOURCE_ID
    assert projected.identity.source.source_kind is CapabilitySourceKind.BUILTIN
    assert projected.provenance.package_reference == "rag"


def test_project_skill_bundle_entry_exposes_catalog_manifest_version_label() -> None:
    register_default_skills(bundle_ids=["harness"])
    projected = project_skill_bundle_entry(_bundle(), HARNESS_TOOL_SMOKE.skill_id)
    assert projected.provenance.version_label == HARNESS_TOOL_SMOKE.version
    assert (
        projected.provenance.version_binding_disposition
        is SkillVersionBindingDisposition.MATERIALIZED
    )


def test_skill_bundle_catalog_source_expands_bundle_to_capability_level_entries() -> None:
    register_skill_bundle(_bundle())
    entries = SkillBundleCatalogSource().read_entries()
    logical_ids = {entry.identity.logical.logical_id for entry in entries}
    assert logical_ids == {"skills.rag.search", "skills.rag.rank"}
    assert all(entry.provenance.package_reference == "rag" for entry in entries)
