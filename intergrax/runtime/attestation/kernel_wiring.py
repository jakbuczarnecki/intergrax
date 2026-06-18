# © Artur Czarnecki. All rights reserved.

"""Wire execution boundary export settings into HarnessKernel session context."""

from __future__ import annotations

from typing import Any

from intergrax.runtime.nexus.config import RuntimeConfig


def apply_boundary_export_to_kernel(kernel_ctx: Any, config: RuntimeConfig) -> None:
    """Copy EBE runtime settings and shared buffer from agent runtime config."""
    if config.execution_boundary_export is not None:
        kernel_ctx.execution_boundary_export = config.execution_boundary_export
    if config.boundary_event_buffer is not None:
        kernel_ctx.boundary_event_buffer = config.boundary_event_buffer
