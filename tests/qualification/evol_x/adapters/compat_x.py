# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.migrations.registry import CONTRACT_SCHEMA_REGISTRY
from testing_support.change_safety.assess.contract_diff import diff_json_field_schemas
from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult


class CompatXRegistryAdapter:
    """Read-only COMPAT-X surface: contract schema registry entries."""

    def list_contract_names(self) -> frozenset[str]:
        return frozenset(entry.contract_name for entry in CONTRACT_SCHEMA_REGISTRY)

    def diff_schemas(
        self,
        before: dict,
        after: dict,
    ) -> CompatibilityAtChangeResult:
        return diff_json_field_schemas(before, after)
