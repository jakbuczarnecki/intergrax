# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Namespaced diagnostic taxonomy extension SPI (R5)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DiagnosticTaxonomyContributor(Protocol):
    """Register namespaced failure/evidence kinds — not a global mega-enum."""

    @property
    def contributor_id(self) -> str:
        """Stable contributor identity."""

    @property
    def taxonomy_namespace(self) -> str:
        """Owning namespace prefix (e.g. company.payment)."""

    @property
    def priority(self) -> int:
        """Lower values merge earlier when building the catalog."""

    def registered_kinds(self) -> tuple[str, ...]:
        """Fully qualified kind tokens (namespace.kind)."""


def validate_taxonomy_kind_token(kind: str, *, expected_namespace: str) -> str:
    normalized = kind.strip()
    if not normalized or "." not in normalized:
        raise ValueError("taxonomy kind must be namespaced")
    prefix = expected_namespace.strip()
    if not normalized.startswith(f"{prefix}."):
        raise ValueError(
            f"taxonomy kind {normalized!r} must live under namespace {prefix!r}",
        )
    return normalized


__all__ = [
    "DiagnosticTaxonomyContributor",
    "validate_taxonomy_kind_token",
]
