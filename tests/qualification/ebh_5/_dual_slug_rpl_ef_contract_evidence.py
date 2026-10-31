# © Artur Czarnecki. All rights reserved.

"""Dual-slug RPL-E contract obligations + audit bindings (Package C v2 — not ID-only proof)."""

from __future__ import annotations

from typing import Final

import pytest
from tests.qualification.ebh_5._dual_slug_rpl_e_operational import assert_rpl_e_category_operational_surface
from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import IntegrationDualSlugCatalogEntry
from tests.qualification.ebh_5._proof_validation import pytest_test_function_exists

_PKG_C_V2: Final[str] = "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py"
_CDE_NODE: Final[str] = f"{_PKG_C_V2}::test_ebh5_pkgc_v2_dual_slug_rpl_cde_equivalence"
_UNREG_NODE: Final[str] = f"{_PKG_C_V2}::test_ebh5_pkgc_v2_dual_slug_unregistered"


def canonical_rpl_e_guarantees(entry: IntegrationDualSlugCatalogEntry) -> tuple[str, ...]:
    base: tuple[str, ...] = (
        "RPL-C: IntegrationProfile selects distinct registered catalog providers @ resolve_from_profile",
        "RPL-D: symmetric connect/close capability surface on both resolved instances",
        "RPL-E: resolved instances implement contract_for_category platform contract",
        "RPL-E: profile slug mismatch fail-closed when only the other slug is registered",
        "RPL-E: empty/unregistered catalog fail-closed @ resolve_from_profile",
    )
    if entry.category.value == "relational_store":
        return base + (
            "RPL-E: relational_store connect + execute('SELECT 1') + close on both backends (sanctioned local)",
        )
    if entry.operational_lifecycle:
        return base + ("RPL-E: lifecycle hooks exercised on both backends @ Package C v2",)
    return base + ("RPL-E: reachability + contract typing @ resolve — no live vendor credential path @ EBH-7",)


def dual_slug_rpl_ef_positive_negative_nodes(entry: IntegrationDualSlugCatalogEntry) -> tuple[str, str]:
    return (
        f"{_CDE_NODE}[{entry.concern_id}]",
        f"{_UNREG_NODE}[{entry.concern_id}]",
    )


def _assert_rpl_e_contract_typing(inst_a: object, inst_b: object, entry: IntegrationDualSlugCatalogEntry) -> None:
    from intergrax.runtime.integrations.contract_metadata import contract_for_category

    contract_cls = contract_for_category(entry.category.value)
    assert isinstance(inst_a, contract_cls), f"{entry.concern_id}: inst_a not {contract_cls.__name__}"
    assert isinstance(inst_b, contract_cls), f"{entry.concern_id}: inst_b not {contract_cls.__name__}"


def _assert_rpl_e_provider_identity(
    inst_a: object,
    inst_b: object,
    entry: IntegrationDualSlugCatalogEntry,
) -> None:
    assert getattr(inst_a, "provider_id", "").strip().lower() == entry.slug_a.strip().lower()
    assert getattr(inst_b, "provider_id", "").strip().lower() == entry.slug_b.strip().lower()


def _assert_rpl_e_wrong_slug_fail_closed(
    entry: IntegrationDualSlugCatalogEntry,
    *,
    tmp_path: object | None,
) -> None:
    from intergrax.integrations.contracts.base import IntegrationConfigurationError, UnknownIntegrationError
    from intergrax.integrations.registry.catalog import clear_catalog
    from intergrax.integrations.registry.factory import resolve_from_profile
    from tests.qualification.ebh_5._package_c_v2_primitives import profile_for_slug, register_catalog_slug

    clear_catalog()
    register_catalog_slug(category=entry.category, slug=entry.slug_a)
    with pytest.raises((UnknownIntegrationError, IntegrationConfigurationError)):
        resolve_from_profile(
            profile_for_slug(category=entry.category, slug=entry.slug_b, tmp_path=tmp_path),
            entry.category,
        )


def _assert_rpl_e_relational_operational(inst_a: object, inst_b: object) -> None:
    for inst in (inst_a, inst_b):
        connect = getattr(inst, "connect", None)
        execute = getattr(inst, "execute", None)
        close = getattr(inst, "close", None)
        assert callable(connect) and callable(execute) and callable(close)
        connect()
        execute("SELECT 1")
        close()


def assert_dual_slug_rpl_e_contract_evidence(
    entry: IntegrationDualSlugCatalogEntry,
    inst_a: object,
    inst_b: object,
    *,
    tmp_path: object | None = None,
) -> None:
    """RPL-E — contract-level replaceability evidence beyond provider_id / lifecycle symmetry."""
    _assert_rpl_e_contract_typing(inst_a, inst_b, entry)
    _assert_rpl_e_provider_identity(inst_a, inst_b, entry)
    if entry.category.value == "relational_store":
        _assert_rpl_e_relational_operational(inst_a, inst_b)
    assert_rpl_e_category_operational_surface(entry, inst_a, inst_b)
    _assert_rpl_e_wrong_slug_fail_closed(entry, tmp_path=tmp_path)


def dual_slug_rpl_ef_audit_errors(entry: IntegrationDualSlugCatalogEntry) -> tuple[str, ...]:
    """Static audit — maps concern to executed pytest proof nodes + canonical RPL-E obligations."""
    errors: list[str] = []
    pos, neg = dual_slug_rpl_ef_positive_negative_nodes(entry)
    if not pytest_test_function_exists(_CDE_NODE):
        errors.append(f"{entry.concern_id}: missing {_CDE_NODE}")
    if not pytest_test_function_exists(_UNREG_NODE):
        errors.append(f"{entry.concern_id}: missing {_UNREG_NODE}")
    if not canonical_rpl_e_guarantees(entry):
        errors.append(f"{entry.concern_id}: empty RPL-E guarantee list")
    if not pos.startswith(_CDE_NODE):
        errors.append(f"{entry.concern_id}: positive node binding drift")
    if not neg.startswith(_UNREG_NODE):
        errors.append(f"{entry.concern_id}: negative node binding drift")
    return tuple(errors)
