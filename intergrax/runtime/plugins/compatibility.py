# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime plugin compatibility evaluation (§42.22)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.plugins.contract import RuntimePlugin
from intergrax.runtime.schema.registry import RuntimeVersionInfo


@dataclass(frozen=True, slots=True)
class RuntimePluginCompatibilityResult:
    """Immutable compatibility evaluation for a single RuntimePlugin."""

    plugin_id: str
    plugin_contract_bundle: str
    runtime_contract_bundle: str
    missing_schemas: frozenset[str]

    @property
    def compatible(self) -> bool:
        return (
            self.plugin_contract_bundle == self.runtime_contract_bundle
            and not self.missing_schemas
        )


class RuntimePluginCompatibilityError(Exception):
    """Raised when a RuntimePlugin is incompatible with the composed runtime."""

    def __init__(self, result: RuntimePluginCompatibilityResult) -> None:
        self.plugin_id = result.plugin_id
        self.plugin_contract_bundle = result.plugin_contract_bundle
        self.runtime_contract_bundle = result.runtime_contract_bundle
        self.missing_schemas = result.missing_schemas
        details: list[str] = []
        if result.plugin_contract_bundle != result.runtime_contract_bundle:
            details.append(
                "contract_bundle mismatch: "
                f"plugin requires {result.plugin_contract_bundle!r}, "
                f"runtime has {result.runtime_contract_bundle!r}"
            )
        if result.missing_schemas:
            details.append(
                f"missing schemas: {sorted(result.missing_schemas)!r}"
            )
        super().__init__(
            f"runtime plugin {result.plugin_id!r} incompatible: {'; '.join(details)}"
        )


def evaluate_runtime_plugin_compatibility(
    plugin: RuntimePlugin,
    runtime: RuntimeVersionInfo,
) -> RuntimePluginCompatibilityResult:
    """Evaluate whether *plugin* can register against *runtime*."""
    missing = plugin.compatible_runtime.supported_schemas - runtime.supported_schemas
    return RuntimePluginCompatibilityResult(
        plugin_id=plugin.plugin_id,
        plugin_contract_bundle=plugin.compatible_runtime.contract_bundle,
        runtime_contract_bundle=runtime.contract_bundle,
        missing_schemas=frozenset(missing),
    )
