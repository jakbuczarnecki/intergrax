# © Artur Czarnecki. All rights reserved.

"""Static role/scope → tool and agent allowlists (Phase H-APP.2.3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ApplicationScopePolicy(Protocol):
    """Map authenticated principal scopes to allowed Tier-2/Tier-0 ids."""

    def allowed_tool_ids(self, *, scopes: frozenset[str]) -> frozenset[str]: ...

    def allowed_agent_ids(self, *, scopes: frozenset[str]) -> frozenset[str]: ...


class StaticApplicationScopePolicy:
    """Default scope policy backed by explicit role → id maps."""

    __slots__ = ("_role_tools", "_role_agents")

    def __init__(
        self,
        *,
        role_tools: dict[str, frozenset[str]] | None = None,
        role_agents: dict[str, frozenset[str]] | None = None,
    ) -> None:
        self._role_tools = role_tools or {}
        self._role_agents = role_agents or {}

    def allowed_tool_ids(self, *, scopes: frozenset[str]) -> frozenset[str]:
        allowed: set[str] = set()
        for scope in scopes:
            allowed.update(self._role_tools.get(scope, ()))
        return frozenset(allowed)

    def allowed_agent_ids(self, *, scopes: frozenset[str]) -> frozenset[str]:
        allowed: set[str] = set()
        for scope in scopes:
            allowed.update(self._role_agents.get(scope, ()))
        return frozenset(allowed)
