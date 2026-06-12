# © Artur Czarnecki. All rights reserved.

"""ContextCompiler hot-path helpers (CE-3.9)."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler
from intergrax.runtime.nexus.context.context_compiler_models import ContextCompileResult
from intergrax.runtime.nexus.context.context_preflight import verify_context_preflight

if TYPE_CHECKING:
    from intergrax.runtime.nexus.config import RuntimeConfig


def compile_chat_messages(
    messages: List[ChatMessage],
    config: "RuntimeConfig",
    *,
    compiler: ContextCompiler | None = None,
    max_output_tokens: Optional[int] = None,
    run_preflight: bool = True,
) -> ContextCompileResult:
    """Apply global budget compiler + optional preflight on a message list."""
    active_compiler = compiler or ContextCompiler()
    result = active_compiler.compile(
        list(messages),
        config,
        max_output_tokens=max_output_tokens,
    )
    if run_preflight:
        verify_context_preflight(
            result.messages,
            config.llm_adapter,
            max_output_tokens=max_output_tokens,
        )
    return result


def compile_prompt_text(
    prompt: str,
    config: "RuntimeConfig",
    *,
    compiler: ContextCompiler | None = None,
    max_output_tokens: Optional[int] = None,
    run_preflight: bool = True,
) -> str:
    """Compile a single-turn prompt through the ContextCompiler spine."""
    messages = [ChatMessage(role="user", content=prompt)]
    result = compile_chat_messages(
        messages,
        config,
        compiler=compiler,
        max_output_tokens=max_output_tokens,
        run_preflight=run_preflight,
    )
    if not result.messages:
        return ""
    return result.messages[-1].content or ""
