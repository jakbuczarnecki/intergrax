# © Artur Czarnecki. All rights reserved.

"""CE-owned typed assembly runtime dependency contract (CE-01-R1)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

from intergrax.contracts.runtime_event_recording import RuntimeEventRecorderPort
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.context_lifecycle.contracts import ContextOptimizationPolicy
from intergrax.runtime.context_lifecycle.repository import OptimizationArtifactRepository
from intergrax.runtime.context_lifecycle.message_sequence_execution_port import (
    MessageSequenceArtifactExecutionPort,
)


@runtime_checkable
class ContextEngineRuntimeConfig(Protocol):
    """Minimal runtime config surface required by canonical ContextEngine assembly."""

    @property
    def llm_adapter(self) -> LLMAdapter | None: ...

    @property
    def production_mode(self) -> bool: ...


@runtime_checkable
class ContextAssemblyUCLRuntime(Protocol):
    """UCL execution dependencies for assembly (concrete type is Execution-owned)."""

    @property
    def repository(self) -> OptimizationArtifactRepository: ...

    @property
    def message_sequence_executor(self) -> MessageSequenceArtifactExecutionPort: ...

    @property
    def strategy_versions(self) -> Mapping[str, str]: ...

    @property
    def artifact_id_factory(self) -> Callable[[], str]: ...

    @property
    def wait_timeout_seconds(self) -> float: ...


@runtime_checkable
class ContextAssemblyRuntime(Protocol):
    """Typed assembly runtime seam for ``ContextEngine.assemble`` (contract ≠ implementation)."""

    @property
    def runtime_config(self) -> ContextEngineRuntimeConfig: ...

    @property
    def base_messages(self) -> tuple[ChatMessage, ...]: ...

    @property
    def max_output_tokens(self) -> int | None: ...

    @property
    def optimization_policy(self) -> ContextOptimizationPolicy | None: ...

    @property
    def ucl_runtime(self) -> ContextAssemblyUCLRuntime | None: ...

    @property
    def event_bus(self) -> RuntimeEventRecorderPort | None: ...

    @property
    def node_id(self) -> str | None: ...

    @property
    def agent_id(self) -> str | None: ...


def validate_context_assembly_ucl_runtime(ucl_runtime: ContextAssemblyUCLRuntime) -> None:
    """Fail-closed semantic validation for UCL runtime contract values (not concrete Nexus type)."""
    if not isinstance(ucl_runtime, ContextAssemblyUCLRuntime):
        raise ValueError("ucl_runtime must satisfy ContextAssemblyUCLRuntime")
    if not isinstance(ucl_runtime.repository, OptimizationArtifactRepository):
        raise ValueError("repository must satisfy OptimizationArtifactRepository")
    executor = ucl_runtime.message_sequence_executor
    if not isinstance(executor, MessageSequenceArtifactExecutionPort):
        raise ValueError("message_sequence_executor must satisfy MessageSequenceArtifactExecutionPort")
    strategy_versions = ucl_runtime.strategy_versions
    if not isinstance(strategy_versions, Mapping):
        raise ValueError("strategy_versions must be a Mapping")
    if not strategy_versions:
        raise ValueError("strategy_versions must contain at least one entry")
    for key, value in strategy_versions.items():
        if not isinstance(key, str) or not key:
            raise ValueError("strategy_versions keys must be non-empty strings")
        if not isinstance(value, str) or not value:
            raise ValueError("strategy_versions values must be non-empty strings")
    if not callable(ucl_runtime.artifact_id_factory):
        raise TypeError("artifact_id_factory must be callable")
    timeout = ucl_runtime.wait_timeout_seconds
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("wait_timeout_seconds must be int or float")
    timeout_value = float(timeout)
    if not math.isfinite(timeout_value) or timeout_value < 0 or timeout_value > 5.0:
        raise ValueError("wait_timeout_seconds must be finite and in [0, 5.0]")


def validate_context_assembly_event_recorder(
    recorder: RuntimeEventRecorderPort | None,
) -> RuntimeEventRecorderPort | None:
    if recorder is None:
        return None
    if not isinstance(recorder, RuntimeEventRecorderPort):
        raise ValueError("event_bus must satisfy RuntimeEventRecorderPort")
    return recorder
