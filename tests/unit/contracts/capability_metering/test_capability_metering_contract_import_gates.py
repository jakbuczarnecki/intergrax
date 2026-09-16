# © Artur Czarnecki. All rights reserved.

"""ME-8 — capability metering contracts must stay billing-agnostic."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PACKAGE_MODULE = "intergrax.contracts.capability_metering"

_FORBIDDEN_BILLING_PREFIXES = (
    "intergrax.tools.providers.billing",
    "intergrax.skills.providers.billing",
    "intergrax.marketplace",
    "stripe",
    "adyen",
    "paddle",
    "chargebee",
    "zuora",
)


def _package_root() -> Path:
    package = importlib.import_module(_PACKAGE_MODULE)
    assert package.__path__ is not None
    return Path(package.__path__[0])


def _iter_package_py_files() -> list[Path]:
    return sorted(path for path in _package_root().rglob("*.py") if path.is_file())


def _collect_imports(tree: ast.AST) -> list[str]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def test_capability_metering_contracts_do_not_import_billing_or_marketplace() -> None:
    root = _package_root()
    for path in _iter_package_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for prefix in _FORBIDDEN_BILLING_PREFIXES:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(
                        f"{path.relative_to(root)} imports forbidden dependency: {imported}",
                    )


def test_usage_event_contract_has_no_billing_authority_fields() -> None:
    from intergrax.contracts.capability_metering import CapabilityUsageEvent

    forbidden = (
        "final_price",
        "amount_due",
        "invoice_id",
        "tax",
        "discount_applied",
        "settlement_status",
        "payment_status",
    )
    field_names = frozenset(CapabilityUsageEvent.model_fields)
    violations = [name for name in forbidden if name in field_names]
    assert not violations, f"usage event exposes billing authority fields: {violations}"
