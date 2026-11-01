# © Artur Czarnecki. All rights reserved.

"""Category-scoped RPL-E operational surface — fail-closed; no swallowed probe exceptions."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import StrEnum

from intergrax.integrations.contracts.base import HealthStatus
from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import IntegrationDualSlugCatalogEntry

_LIFECYCLE_METHODS: tuple[str, ...] = ("connect", "close")

# Path B: presence checks only — no guessed method names; per-category ops added when contract SSOT pins them.
_REQUIRED_SYMMETRIC_OPERATIONS: dict[str, tuple[str, ...]] = {
    "message_bus": ("publish", "close"),
    "notification_channel": ("notify",),
    "object_storage": ("put", "get", "close"),
    "issue_tracker": ("create_issue", "get_issue"),
}


class OperationalEvidenceKind(StrEnum):
    RELATIONAL_EXECUTE = "relational_execute"
    STRUCTURAL_CONTRACT = "structural_contract"


@dataclass(frozen=True, slots=True)
class OperationalEvidencePolicy:
    kind: OperationalEvidenceKind
    required_symmetric_operations: tuple[str, ...]


def operational_evidence_policy(entry: IntegrationDualSlugCatalogEntry) -> OperationalEvidencePolicy:
    if entry.category.value == "relational_store":
        return OperationalEvidencePolicy(
            OperationalEvidenceKind.RELATIONAL_EXECUTE,
            ("connect", "execute", "close"),
        )
    required = _REQUIRED_SYMMETRIC_OPERATIONS.get(entry.category.value, ())
    return OperationalEvidencePolicy(OperationalEvidenceKind.STRUCTURAL_CONTRACT, required)


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


def _assert_symmetric_required_operations(
    inst_a: object,
    inst_b: object,
    entry: IntegrationDualSlugCatalogEntry,
    required: tuple[str, ...],
) -> None:
    names_a = _public_callable_names(inst_a)
    names_b = _public_callable_names(inst_b)
    for name in required:
        assert name in names_a and name in names_b, (
            f"{entry.concern_id}: required contract operation {name}() missing on both backends"
        )
    shared = names_a & names_b
    if not required:
        assert len(shared) >= 2, (
            f"{entry.concern_id}: structural Path B requires >=2 shared public contract callables, "
            f"got {sorted(shared)[:6]}"
        )
    for name in _LIFECYCLE_METHODS:
        assert (name in names_a) == (name in names_b), f"{entry.concern_id}: lifecycle symmetry required for {name}()"


def assert_strict_qualification_probe(
    inst: object,
    method_name: str,
    *,
    concern_id: str,
) -> None:
    """Invoke one sanctioned probe — any exception or unhealthy HealthStatus is FAIL."""
    fn = getattr(type(inst), method_name, None)
    if fn is None or not callable(getattr(inst, method_name, None)):
        msg = f"{concern_id}: operational probe {method_name}() required but not available"
        raise AssertionError(msg)
    result = getattr(inst, method_name)()
    if isinstance(result, HealthStatus):
        assert result.healthy, f"{concern_id}: {method_name}() reported unhealthy: {result.detail!r}"


def assert_rpl_e_category_operational_surface(
    entry: IntegrationDualSlugCatalogEntry,
    inst_a: object,
    inst_b: object,
) -> None:
    """Path B — structural contract surface; relational execute is owned by contract_evidence layer."""
    policy = operational_evidence_policy(entry)
    _assert_platform_contract_metadata(inst_a, inst_b, entry)
    _assert_symmetric_required_operations(inst_a, inst_b, entry, policy.required_symmetric_operations)
    if policy.kind is OperationalEvidenceKind.RELATIONAL_EXECUTE:
        return
