# © Artur Czarnecki. All rights reserved.

"""Trusted runtime carrier for active Execution authority during governed work."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from intergrax.contracts.delegation_authority import (
    EffectiveDelegationAuthority,
    ParentExecutionAuthority,
)


@dataclass(frozen=True, slots=True)
class ActiveExecutionAuthorityState:
    authority: ParentExecutionAuthority
    effective_delegation: EffectiveDelegationAuthority | None = None


_active_execution_authority: ContextVar[ActiveExecutionAuthorityState | None] = ContextVar(
    "active_execution_authority",
    default=None,
)


def bind_active_execution_authority(
    authority: ParentExecutionAuthority,
    *,
    effective_delegation: EffectiveDelegationAuthority | None = None,
) -> Token:
    return _active_execution_authority.set(
        ActiveExecutionAuthorityState(
            authority=authority,
            effective_delegation=effective_delegation,
        )
    )


def reset_active_execution_authority(token: Token) -> None:
    _active_execution_authority.reset(token)


def peek_active_execution_authority() -> ParentExecutionAuthority | None:
    state = _active_execution_authority.get()
    if state is None:
        return None
    return state.authority


def peek_active_effective_delegation() -> EffectiveDelegationAuthority | None:
    state = _active_execution_authority.get()
    if state is None:
        return None
    return state.effective_delegation


def require_active_execution_authority() -> ParentExecutionAuthority:
    authority = peek_active_execution_authority()
    if authority is None:
        raise RuntimeError("active execution authority required")
    return authority


class ActiveExecutionAuthority:
    """Stateless facade; canonical active authority lives in ContextVar."""

    __slots__ = ()

    def bind(
        self,
        authority: ParentExecutionAuthority,
        *,
        effective_delegation: EffectiveDelegationAuthority | None = None,
    ) -> Token:
        return bind_active_execution_authority(
            authority,
            effective_delegation=effective_delegation,
        )

    def reset(self, token: Token) -> None:
        reset_active_execution_authority(token)

    @property
    def current(self) -> ParentExecutionAuthority | None:
        return peek_active_execution_authority()

    def require(self) -> ParentExecutionAuthority:
        return require_active_execution_authority()
