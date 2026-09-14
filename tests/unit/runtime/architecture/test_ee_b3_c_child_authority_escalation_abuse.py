# © Artur Czarnecki. All rights reserved.

"""EE-B3-C — AC-03 child authority escalation abuse."""

from __future__ import annotations

import pytest

from intergrax.contracts.delegation_authority import (
    DelegationAuthorityError,
    ParentExecutionAuthority,
    mint_effective_delegation_authority,
)
from intergrax.runtime.execution.authority.policy import (
    ChildAuthorityContext,
    DefaultStrictAuthorityPolicy,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b3_c_child_write_beyond_parent_read_denied() -> None:
    parent = ParentExecutionAuthority.scoped(("read",))
    policy = DefaultStrictAuthorityPolicy()
    with pytest.raises(DelegationAuthorityError):
        policy.resolve_child_authority(
            ChildAuthorityContext(
                parent_authority=parent,
                requested_permission_scopes=("read", "write"),
            ),
        )


def test_ee_b3_c_effective_child_authority_subset_of_parent() -> None:
    parent = ParentExecutionAuthority.scoped(("read", "audit"))
    effective = mint_effective_delegation_authority(
        parent=parent,
        requested_permission_scopes=("read",),
    )
    parent_scopes = set(parent.permission_scopes)
    assert set(effective.effective_permission_scopes) <= parent_scopes


def test_ee_b3_c_overreach_not_silently_narrowed_when_rejected() -> None:
    parent = ParentExecutionAuthority.scoped(("ops:read",))
    with pytest.raises(DelegationAuthorityError):
        mint_effective_delegation_authority(
            parent=parent,
            requested_permission_scopes=("ops:read", "ops:write"),
        )
