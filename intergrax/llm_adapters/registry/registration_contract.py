# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

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


@runtime_checkable
class LLMAdapterRegistrationTarget(Protocol):
    @classmethod
    def register_from_spec(
        cls,
        spec: LLMAdapterRegistrationSpec,
        *,
        override: bool = False,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class OptionalDependencyRequirement:
    import_names: tuple[str, ...]
    distribution_name: str
    extra_name: str

    def matches_missing_module(self, exc: ModuleNotFoundError) -> bool:
        missing_name = exc.name
        if missing_name is None:
            return False
        for import_name in self.import_names:
            if missing_name == import_name or missing_name.startswith(f"{import_name}."):
                return True
        return False

    def to_dependency_error(
        self,
        *,
        provider_id: str,
        exc: ModuleNotFoundError,
    ) -> LLMAdapterDependencyError:
        return LLMAdapterDependencyError(
            f"LLM provider '{provider_id}' requires dependency "
            f"'{self.distribution_name}'. Install it with "
            f"'Intergrax-ai[{self.extra_name}]' before selecting this provider."
        )
