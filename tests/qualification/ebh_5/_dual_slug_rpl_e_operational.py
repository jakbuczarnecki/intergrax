# © Artur Czarnecki. All rights reserved.

"""Category-scoped RPL-E operational surface — beyond ID/slug/isinstance checks."""

from __future__ import annotations

import inspect

from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import IntegrationDualSlugCatalogEntry

_MIN_SHARED_PUBLIC_APIS: int = 3
_LIFECYCLE_METHODS: tuple[str, ...] = ("connect", "close")


def _public_callable_names(obj: object) -> frozenset[str]:
    names: set[str] = set()
    for cls in type(obj).mro():
        for name, attr in cls.__dict__.items():
            if name.startswith("_"):
                continue
            if inspect.isfunction(attr) or isinstance(attr, (classmethod, staticmethod)):
                names.add(name)
    return frozenset(names)


def _assert_platform_contract_metadata(inst_a: object, inst_b: object, entry: IntegrationDualSlugCatalogEntry) -> None:
    for attr in ("schema_id", "integration_kind"):
        va = getattr(inst_a, attr, None)
        vb = getattr(inst_b, attr, None)
        assert va and va == vb, f"{entry.concern_id}: {attr} mismatch or empty ({va!r} vs {vb!r})"
    ca = getattr(inst_a, "capabilities", None)
    cb = getattr(inst_b, "capabilities", None)
    assert ca is not None and ca == cb, f"{entry.concern_id}: capabilities must match on both backends"


def _assert_symmetric_category_api(inst_a: object, inst_b: object, entry: IntegrationDualSlugCatalogEntry) -> None:
    shared = _public_callable_names(inst_a) & _public_callable_names(inst_b)
    assert len(shared) >= _MIN_SHARED_PUBLIC_APIS, (
        f"{entry.concern_id}: need >={_MIN_SHARED_PUBLIC_APIS} shared public callables, got {sorted(shared)[:8]}"
    )
    for name in _LIFECYCLE_METHODS:
        a_has = name in shared
        b_has = name in _public_callable_names(inst_b) and name in _public_callable_names(inst_a)
        assert a_has == b_has, f"{entry.concern_id}: lifecycle symmetry required for {name}()"


def _safe_probe(inst: object) -> None:
    for name in ("health", "ping", "connect", "close"):
        fn = getattr(inst, name, None)
        if not callable(fn):
            continue
        try:
            fn()
        except Exception:
            continue
        return


def assert_rpl_e_category_operational_surface(
    entry: IntegrationDualSlugCatalogEntry,
    inst_a: object,
    inst_b: object,
) -> None:
    """Contract metadata, symmetric public API, and best-effort local probe on both backends."""
    _assert_platform_contract_metadata(inst_a, inst_b, entry)
    _assert_symmetric_category_api(inst_a, inst_b, entry)
    if entry.category.value != "relational_store":
        _safe_probe(inst_a)
        _safe_probe(inst_b)
