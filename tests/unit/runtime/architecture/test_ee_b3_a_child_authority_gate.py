# © Artur Czarnecki. All rights reserved.

"""EE-B3-A — child execution authority gate."""

from __future__ import annotations

import pytest

from intergrax.contracts.delegation_authority import (
    DelegationAuthorityError,
    ParentExecutionAuthority,
)
from intergrax.runtime.execution.authority.policy import (
    ChildAuthorityContext,
    DefaultStrictAuthorityPolicy,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b3_a_child_policy_denies_scope_beyond_parent() -> None:
    policy = DefaultStrictAuthorityPolicy()
    parent = ParentExecutionAuthority.scoped(("ops:read",))
    with pytest.raises(DelegationAuthorityError):
        policy.resolve_child_authority(
            ChildAuthorityContext(
                parent_authority=parent,
                requested_permission_scopes=("ops:read", "ops:write"),
            ),
        )


def test_ee_b3_a_child_inherits_parent_when_no_explicit_request() -> None:
    policy = DefaultStrictAuthorityPolicy()
    parent = ParentExecutionAuthority.scoped(("ops:read",))
    resolution = policy.resolve_child_authority(
        ChildAuthorityContext(
            parent_authority=parent,
            requested_permission_scopes=None,
        ),
    )
    assert resolution.authority == parent
    assert resolution.effective_delegation is None
