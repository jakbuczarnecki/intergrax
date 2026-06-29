# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Contract-aware guards for provider shell generators (INTEGRATIONS-2B)."""

from __future__ import annotations

from pathlib import Path

# Hand-edited when a provider adopts the contract-based integration model.
HAND_EDITED_PROVIDER_FILES: frozenset[str] = frozenset(
    {
        "integration.py",
        "bundle.py",
        "__init__.py",
        "register.py",
        "manifest.py",
        "USAGE.md",
    }
)

CONTRACT_INTEGRATION_FILE = "integration.py"


def is_contract_aware_package(pkg: Path) -> bool:
    """True when the slug already hosts a contract-based integration module."""
    return (pkg / CONTRACT_INTEGRATION_FILE).is_file()


def should_skip_provider_file(pkg: Path, filename: str) -> bool:
    """
    Skip overwriting canonical/hand-edited files for contract-aware packages.

    Legacy shell generators must remain idempotent: they may (re)generate thin
    legacy shells for unmigrated providers, but must never destroy an existing
    contract-based integration package layout.
    """
    return is_contract_aware_package(pkg) and filename in HAND_EDITED_PROVIDER_FILES


def write_provider_file_if_allowed(pkg: Path, filename: str, content: str) -> bool:
    """Write a provider shell file unless contract-aware preservation applies."""
    if should_skip_provider_file(pkg, filename):
        return False
    (pkg / filename).write_text(content, encoding="utf-8")
    return True
