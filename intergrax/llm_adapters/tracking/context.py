# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Request-scoped LLM observability context (tenant for billing aggregates)."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

_llm_tenant_id: ContextVar[str] = ContextVar("intergrax_llm_tenant_id", default="")


def set_llm_tenant_id(tenant_id: Optional[str]) -> None:
    """Set tenant for subsequent LLM metric records in this async/task context."""
    _llm_tenant_id.set((tenant_id or "").strip())


def get_llm_tenant_id() -> str:
    return _llm_tenant_id.get()


def clear_llm_tenant_id() -> None:
    _llm_tenant_id.set("")


@contextmanager
def llm_tenant_scope(tenant_id: Optional[str]) -> Iterator[None]:
    """Bind tenant for LLM metrics for the duration of a Nexus task/run."""
    previous = get_llm_tenant_id()
    set_llm_tenant_id(tenant_id)
    try:
        yield
    finally:
        if previous:
            set_llm_tenant_id(previous)
        else:
            clear_llm_tenant_id()
