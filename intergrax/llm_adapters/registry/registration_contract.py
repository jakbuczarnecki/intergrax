# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


class LLMAdapterDependencyError(RuntimeError):
    """Raised when a selected LLM provider SDK is not installed."""


class LLMAdapterRegistrationError(RuntimeError):
    """Raised when an LLM provider registration is invalid."""


@runtime_checkable
class LLMAdapterFactory(Protocol):
    def __call__(self, **kwargs: object) -> LLMAdapter: ...


@dataclass(frozen=True, slots=True)
class LLMAdapterRegistrationSpec:
    provider_id: str
    factory: LLMAdapterFactory


@dataclass(frozen=True, slots=True)
class OptionalDependencyRequirement:
    import_names: tuple[str, ...]
    distribution_name: str
    extra_name: str

    def ensure_available(self, *, provider_id: str) -> None:
        last_exc: ModuleNotFoundError | None = None
        for name in self.import_names:
            try:
                importlib.import_module(name)
                return
            except ModuleNotFoundError as exc:
                if exc.name not in self.import_names:
                    raise
                last_exc = exc

        if last_exc is not None:
            raise LLMAdapterDependencyError(
                f"LLM provider '{provider_id}' requires dependency "
                f"'{self.distribution_name}'. Install it with "
                f"'Intergrax-ai[{self.extra_name}]' before selecting this provider."
            ) from last_exc
