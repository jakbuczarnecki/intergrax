# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Request-scoped LLM observability context (tenant for billing aggregates)."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

_llm_tenant_id: ContextVar[str] = ContextVar("intergrax_llm_tenant_id", default="")


def set_llm_tenant_id(tenant_id: Optional[str]) -> None:
    """Set tenant for subsequent LLM metric records in this async/task context."""
    _llm_tenant_id.set((tenant_id or "").strip())


def get_llm_tenant_id() -> str:
    return _llm_tenant_id.get()


def clear_llm_tenant_id() -> None:
    _llm_tenant_id.set("")
