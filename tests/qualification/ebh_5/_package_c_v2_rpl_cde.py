# © Artur Czarnecki. All rights reserved.

"""Package C v2 — RPL-C/D/E operational equivalence (independent of registry-only resolve)."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.integrations.contracts.base import IntegrationConfigurationError, UnknownIntegrationError
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.integrations.registry.profile import IntegrationProfile
from tests.qualification.ebh_5._dual_slug_rpl_ef_contract_evidence import assert_dual_slug_rpl_e_contract_evidence
from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import IntegrationDualSlugCatalogEntry
from tests.qualification.ebh_5._package_c_v2_primitives import (
    exercise_connect_close_if_supported,
    profile_for_slug,
    register_catalog_slug,
    run_dual_slug_resolve_from_profile,
)


def _assert_rpl_d_lifecycle_parity(inst_a: object, inst_b: object) -> None:
    """RPL-D — both backends expose symmetric lifecycle surface (connect/close or neither)."""
    for inst in (inst_a, inst_b):
        exercise_connect_close_if_supported(inst)
    connect_a = callable(getattr(inst_a, "connect", None))
    connect_b = callable(getattr(inst_b, "connect", None))
    close_a = callable(getattr(inst_a, "close", None))
    close_b = callable(getattr(inst_b, "close", None))
    assert connect_a == connect_b, "connect() symmetry required for dual-slug equivalence"
    assert close_a == close_b, "close() symmetry required for dual-slug equivalence"


def _assert_rpl_e_failure_isolation(entry: IntegrationDualSlugCatalogEntry, *, tmp_path: Path | None) -> None:
    """RPL-E — unregistered catalog fail-closed; registered slug still resolves after negative attempt."""
    clear_catalog()
    with pytest.raises((UnknownIntegrationError, IntegrationConfigurationError)):
        resolve_from_profile(IntegrationProfile(), entry.category)
    clear_catalog()
    register_catalog_slug(category=entry.category, slug=entry.slug_a)
    inst = resolve_from_profile(
        profile_for_slug(category=entry.category, slug=entry.slug_a, tmp_path=tmp_path),
        entry.category,
    )
    assert inst.provider_id.strip().lower() == entry.slug_a.strip().lower()


def run_package_c_v2_dual_slug_rpl_cde_equivalence(
    entry: IntegrationDualSlugCatalogEntry,
    *,
    tmp_path: Path | None = None,
) -> None:
    if entry.category.value == "relational_store":
        assert tmp_path is not None
    # RPL-C — profile-driven selection yields two distinct catalog providers @ resolve_from_profile.
    inst_a, inst_b = run_dual_slug_resolve_from_profile(
        category=entry.category,
        slug_a=entry.slug_a,
        slug_b=entry.slug_b,
        tmp_path=tmp_path,
    )
    assert inst_a.provider_id and inst_b.provider_id
    assert inst_a.provider_id != inst_b.provider_id
    _assert_rpl_d_lifecycle_parity(inst_a, inst_b)
    assert_dual_slug_rpl_e_contract_evidence(entry, inst_a, inst_b, tmp_path=tmp_path)
    _assert_rpl_e_failure_isolation(entry, tmp_path=tmp_path)
