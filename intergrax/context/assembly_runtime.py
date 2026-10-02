# © Artur Czarnecki. All rights reserved.

"""CE-owned typed assembly runtime dependency contract (CE-01-R1)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from intergrax.llm.messages import ChatMessage

if TYPE_CHECKING:
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
    from intergrax.runtime.context_lifecycle.contracts import ContextOptimizationPolicy
    from intergrax.runtime.context_lifecycle.repository import OptimizationArtifactRepository
    from intergrax.runtime.events.event_bus import RuntimeEventBus
    from intergrax.runtime.token_optimization.message_sequence_artifact import (
        MessageSequenceArtifactExecutor,
    )


@runtime_checkable
class ContextEngineRuntimeConfig(Protocol):
    """Minimal runtime config surface required by canonical ContextEngine assembly."""

    @property
    def llm_adapter(self) -> LLMAdapter | None: ...

    @property
    def production_mode(self) -> bool: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


@runtime_checkable
class ContextAssemblyUCLRuntime(Protocol):
    """UCL execution dependencies for assembly (concrete type is Execution-owned)."""

    @property
    def repository(self) -> OptimizationArtifactRepository: ...

    @property
    def message_sequence_executor(self) -> MessageSequenceArtifactExecutor: ...

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
    def event_bus(self) -> RuntimeEventBus | None: ...

    @property
    def node_id(self) -> str | None: ...

    @property
    def agent_id(self) -> str | None: ...
