# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib
from typing import Any

from intergrax.contracts.migrations.registry import CONTRACT_SCHEMA_REGISTRY
from testing_support.change_safety.assess.contract_diff import SchemaDiffRecord, diff_schema_record
from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult


class CompatXRegistryAdapter:
    """Read-only COMPAT-X surface: contract schema registry entries."""

    def list_contract_names(self) -> frozenset[str]:
        return frozenset(entry.contract_name for entry in CONTRACT_SCHEMA_REGISTRY)

    def contract_module_ids(self) -> frozenset[str]:
        return frozenset(f"module:{entry.module_path}" for entry in CONTRACT_SCHEMA_REGISTRY)

    def schema_for(self, contract_name: str) -> dict[str, Any]:
        for entry in CONTRACT_SCHEMA_REGISTRY:
            if entry.contract_name == contract_name:
                module = importlib.import_module(entry.module_path)
                model = getattr(module, entry.contract_name)
                return dict(model.model_json_schema())
        raise KeyError(contract_name)

    def diff_schemas(self, before: dict, after: dict) -> CompatibilityAtChangeResult:
        return diff_schema_record(before, after).result

    def diff_records(self, before: dict | None, after: dict | None) -> SchemaDiffRecord:
        return diff_schema_record(before, after)
