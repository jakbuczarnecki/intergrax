# © Artur Czarnecki. All rights reserved.

"""Re-export EE host orchestration spec builder (path stability for architecture gates)."""

from intergrax.runtime.execution.host_orchestration_environment_spec_builder import (
    build_host_orchestration_loop_init_spec_from_environment,
)

__all__ = ["build_host_orchestration_loop_init_spec_from_environment"]
