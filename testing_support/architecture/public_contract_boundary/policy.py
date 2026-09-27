# © Artur Czarnecki. All rights reserved.

"""Dependency classification policy for public contract surfaces."""

from __future__ import annotations

from testing_support.architecture.public_contract_boundary.import_extraction import (
    is_public_contract_module,
)
from testing_support.architecture.public_contract_boundary.models import DependencyRuleId

_SHARED_SEMANTIC_CONTRACT_PREFIXES: tuple[str, ...] = (
    "intergrax.llm.messages",
    "intergrax.websearch.schemas.search_hit",
)

_PUBLIC_CONTRACT_COMPAT_REEXPORTS: dict[str, frozenset[str]] = {
    "intergrax.llm_adapters.contracts.runtime_lifecycle_binding": frozenset(
        {"intergrax.llm_adapters.base.lifecycle_binding"},
    ),
}


def is_shared_semantic_contract_module(imported_module: str) -> bool:
    return any(
        imported_module == prefix or imported_module.startswith(f"{prefix}.")
        for prefix in _SHARED_SEMANTIC_CONTRACT_PREFIXES
    )


def is_allowed_compat_reexport(source_module: str, imported_module: str) -> bool:
    allowed = _PUBLIC_CONTRACT_COMPAT_REEXPORTS.get(source_module, frozenset())
    return imported_module in allowed or any(
        imported_module.startswith(f"{allowed_prefix}.") for allowed_prefix in allowed
    )


def classify_intergrax_dependency(imported_module: str) -> DependencyRuleId | None:
    if is_shared_semantic_contract_module(imported_module):
        return None
    if imported_module.startswith("intergrax.runtime."):
        return DependencyRuleId.FORBIDDEN_RUNTIME_NAMESPACE
    body = imported_module.removeprefix("intergrax.")
    if (
        not is_public_contract_module(imported_module)
        and (".registry." in f".{body}." or body.startswith("registry."))
    ):
        return DependencyRuleId.FORBIDDEN_REGISTRY_NAMESPACE
    if ".bootstrap." in f".{body}." or body.startswith("bootstrap."):
        return DependencyRuleId.FORBIDDEN_BOOTSTRAP_NAMESPACE
    if imported_module.startswith("intergrax.") and not is_public_contract_module(
        imported_module,
    ):
        return DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION
    return None
