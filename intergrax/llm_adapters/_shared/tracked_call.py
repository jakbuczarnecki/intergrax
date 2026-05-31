# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter, LLMCallStats


@contextmanager
def tracked_llm_call(
    adapter: LLMAdapter,
    *,
    run_id: Optional[str] = None,
) -> Generator[LLMCallStats, None, None]:
    """
    Context manager wrapping ``usage.begin_call`` / ``usage.end_call``.

    Usage::

        with tracked_llm_call(adapter, run_id=run_id) as call:
            ...
            adapter.usage.end_call(call, input_tokens=..., output_tokens=..., success=True)
    """
    call = adapter.usage.begin_call(run_id=run_id)
    try:
        yield call
    finally:
        pass
