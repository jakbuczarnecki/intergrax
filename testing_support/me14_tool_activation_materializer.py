# © Artur Czarnecki. All rights reserved.

"""ME-14 fixture materializer — registers resolved package into host registry."""

from __future__ import annotations

from intergrax.tools.catalog import ToolPackageResolution
from intergrax.tools.dynamic_acquisition import ToolHostActivationMaterializer
from intergrax.tools.registry.provenance import ToolRuntimeActivationMetadata
from intergrax.tools.registry.runtime import ToolRegistry
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V2,
    ME14_VERSION_V2,
    me14_echo_contract,
    Me14EchoToolHandler,
    ME14_OUTPUT_V1,
    ME14_OUTPUT_V2,
)


class Me14ToolHostActivationMaterializer(ToolHostActivationMaterializer):
    def __init__(self, registry: ToolRegistry, *, catalog_source_id: str) -> None:
        self._registry = registry
        self._catalog_source_id = catalog_source_id

    def materialize(
        self,
        resolution: ToolPackageResolution,
    ) -> tuple[str, ToolRuntimeActivationMetadata]:
        candidate = resolution.package_candidate
        if candidate.package_digest is None:
            raise ValueError("package digest required")
        if (
            candidate.package_version == ME14_VERSION_V2
            or candidate.package_digest == ME14_DIGEST_V2
        ):
            output = ME14_OUTPUT_V2
        else:
            output = ME14_OUTPUT_V1
        activation = ToolRuntimeActivationMetadata(
            catalog_source_id=self._catalog_source_id,
            logical_tool_id=candidate.logical_tool_id,
            package_reference=candidate.package_reference,
            version_label=candidate.package_version,
            content_digest=candidate.package_digest,
        )
        self._registry.register(
            me14_echo_contract(),
            Me14EchoToolHandler(output=output),
            activation=activation,
        )
        return candidate.logical_tool_id, activation


__all__ = ["Me14ToolHostActivationMaterializer"]
