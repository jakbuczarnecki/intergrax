# © Artur Czarnecki. All rights reserved.

"""Dependency classification policy for public contract surfaces."""

from __future__ import annotations

from testing_support.architecture.public_contract_boundary.import_extraction import (
    is_public_contract_module,
)
from testing_support.architecture.public_contract_boundary.models import DependencyRuleId


def classify_intergrax_dependency(imported_module: str) -> DependencyRuleId | None:
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
