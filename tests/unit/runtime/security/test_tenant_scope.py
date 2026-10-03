# © Artur Czarnecki. All rights reserved.

"""Unit tests for Security-owned middleware tenant scope semantics (CTRL-X-R3-R2)."""

from __future__ import annotations

import pytest

from intergrax.runtime.security.tenant_scope import (
    normalize_tenant_scope_id,
    tenant_scope_is_valid,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("\t", None),
        ("tenant-a", "tenant-a"),
        (" tenant-a ", "tenant-a"),
    ],
)
def test_normalize_tenant_scope_id(raw: str | None, expected: str | None) -> None:
    assert normalize_tenant_scope_id(raw) == expected


def test_tenant_scope_adversarial_matrix() -> None:
    cases: tuple[tuple[str | None, str | None, bool, bool], ...] = (
        (None, None, True, True),
        (None, None, False, False),
        ("", None, True, True),
        ("   ", None, True, True),
        ("A", None, False, True),
        ("A", "A", False, True),
        ("A", "B", False, False),
        (None, "B", True, False),
        ("", "B", True, False),
        ("   ", "B", True, False),
        (" A ", "A", False, True),
        ("A", " B ", False, False),
    )
    for request, resource, allow_unscoped, expected_allow in cases:
        assert (
            tenant_scope_is_valid(request, resource, allow_unscoped=allow_unscoped)
            is expected_allow
        ), (request, resource, allow_unscoped)
