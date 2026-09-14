# © Artur Czarnecki. All rights reserved.

"""EE-B3-A — authority escalation fail-closed gate."""

from __future__ import annotations

import pytest

from intergrax.contracts.delegation_authority import (
    DelegationAuthorityError,
    ParentExecutionAuthority,
    mint_effective_delegation_authority,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b3_a_parent_read_scope_child_write_scope_denied() -> None:
    parent = ParentExecutionAuthority.scoped(("read",))
    with pytest.raises(DelegationAuthorityError):
        mint_effective_delegation_authority(
            parent=parent,
            requested_permission_scopes=("read", "write"),
        )


def test_ee_b3_a_parent_unknown_cannot_delegate_scopes() -> None:
    parent = ParentExecutionAuthority.unknown()
    with pytest.raises(DelegationAuthorityError):
        mint_effective_delegation_authority(
            parent=parent,
            requested_permission_scopes=("write",),
        )


def test_ee_b3_a_parent_read_child_read_allowed() -> None:
    parent = ParentExecutionAuthority.scoped(("read",))
    effective = mint_effective_delegation_authority(
        parent=parent,
        requested_permission_scopes=("read",),
    )
    assert effective.effective_permission_scopes == ("read",)
