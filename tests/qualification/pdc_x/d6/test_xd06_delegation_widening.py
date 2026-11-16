# © Artur Czarnecki. All rights reserved.

"""PDC-D6-R004 (xd_06): a delegated child can never widen the authority of its parent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from intergrax.contracts.delegation_authority import (
    EFFECTIVE_DELEGATION_AUTHORITY_NODE_KEY,
    DelegationAuthorityError,
    ParentExecutionAuthority,
    effective_delegation_to_parent_authority,
    mint_effective_delegation_authority,
    resolve_parent_execution_authority_for_node,
    validate_effective_delegation_metadata_assertions,
    validate_execution_authority_metadata_assertions,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_xd06_child_requesting_a_scope_the_parent_lacks_is_denied() -> None:
    parent = ParentExecutionAuthority.scoped(("files.read", "tickets.read"))
    with pytest.raises(DelegationAuthorityError):
        mint_effective_delegation_authority(parent=parent, requested_permission_scopes=("files.read", "payments.write"))


def test_xd06_unknown_parent_authority_cannot_delegate_anything() -> None:
    with pytest.raises(DelegationAuthorityError):
        mint_effective_delegation_authority(
            parent=ParentExecutionAuthority.unknown(),
            requested_permission_scopes=("files.read",),
        )


def test_xd06_child_without_request_inherits_no_authority() -> None:
    effective = mint_effective_delegation_authority(
        parent=ParentExecutionAuthority.scoped(("files.read",)), requested_permission_scopes=()
    )
    assert effective.effective_permission_scopes == ()


def test_xd06_widening_is_denied_transitively_down_the_chain() -> None:
    root = ParentExecutionAuthority.scoped(("a.read", "b.read"))
    child = mint_effective_delegation_authority(parent=root, requested_permission_scopes=("a.read",))
    assert child.effective_permission_scopes == ("a.read",)
    grandchild_parent = effective_delegation_to_parent_authority(child)
    with pytest.raises(DelegationAuthorityError):
        # b.read was held by the root but not by the child: it cannot reappear
        mint_effective_delegation_authority(parent=grandchild_parent, requested_permission_scopes=("a.read", "b.read"))
    narrowed = mint_effective_delegation_authority(parent=grandchild_parent, requested_permission_scopes=("a.read",))
    assert narrowed.effective_permission_scopes == ("a.read",)


def test_xd06_child_with_empty_effective_authority_cannot_delegate_further() -> None:
    root = ParentExecutionAuthority.scoped(("a.read",))
    child = mint_effective_delegation_authority(parent=root, requested_permission_scopes=())
    with pytest.raises(DelegationAuthorityError):
        mint_effective_delegation_authority(
            parent=effective_delegation_to_parent_authority(child),
            requested_permission_scopes=("a.read",),
        )


class _Graph:
    def __init__(self, nodes: dict[str, SimpleNamespace]) -> None:
        self._nodes = nodes

    def node_by_id(self, node_id: str) -> SimpleNamespace:
        return self._nodes[node_id]


def _node(node_id: str, depends_on: tuple[str, ...], scopes: tuple[str, ...] | None) -> SimpleNamespace:
    metadata: dict[str, object] = {}
    if scopes is not None:
        metadata[EFFECTIVE_DELEGATION_AUTHORITY_NODE_KEY] = mint_effective_delegation_authority(
            parent=ParentExecutionAuthority.scoped(("a.read", "b.read", "c.read")),
            requested_permission_scopes=scopes,
        )
    return SimpleNamespace(id=node_id, depends_on=depends_on, metadata=metadata)


def test_xd06_join_node_only_holds_the_intersection_of_its_parents() -> None:
    graph = _Graph(
        {
            "left": _node("left", (), ("a.read", "b.read")),
            "right": _node("right", (), ("b.read", "c.read")),
        }
    )
    join = SimpleNamespace(id="join", depends_on=("left", "right"), metadata={})
    parent = resolve_parent_execution_authority_for_node(
        graph, join, root_authority=ParentExecutionAuthority.scoped(("a.read", "b.read", "c.read"))
    )
    assert parent.permission_scopes == ("b.read",)
    with pytest.raises(DelegationAuthorityError):
        mint_effective_delegation_authority(parent=parent, requested_permission_scopes=("a.read",))


def test_xd06_metadata_cannot_assert_authority_beyond_the_typed_root() -> None:
    trusted = ParentExecutionAuthority.scoped(("a.read",))
    assert validate_execution_authority_metadata_assertions({"execution_authority_unrestricted": True}, trusted)
    assert validate_execution_authority_metadata_assertions(
        {"execution_permission_scopes": ["a.read", "b.read"]}, trusted
    )
    assert (
        validate_execution_authority_metadata_assertions({"execution_permission_scopes": ["a.read"]}, trusted) is None
    )


def test_xd06_request_metadata_cannot_assert_wider_effective_scopes() -> None:
    effective = mint_effective_delegation_authority(
        parent=ParentExecutionAuthority.scoped(("a.read", "b.read")),
        requested_permission_scopes=("a.read",),
    )
    assert validate_effective_delegation_metadata_assertions(
        {"effective_permission_scopes": ["a.read", "b.read"]}, effective
    )
    assert (
        validate_effective_delegation_metadata_assertions({"effective_permission_scopes": ["a.read"]}, effective)
        is None
    )
