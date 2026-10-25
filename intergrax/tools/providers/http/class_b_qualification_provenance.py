# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Trusted registration provenance for TOOL-LIFE-X HTTP Class B qualification tools."""

from __future__ import annotations

import hashlib
from typing import Final

from intergrax.tools.registry.provenance import ToolRuntimeActivationMetadata
from intergrax.tools.registry.runtime import RegisteredTool

TOOL_LIFE_HTTP_CLASS_B_QUALIFICATION_CATALOG_SOURCE_ID: Final = (
    "tool_life.http.class_b_qualification_composition.v1"
)
TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_PACKAGE_REFERENCE: Final = (
    "intergrax.tools.providers.http.idempotent_post_qual"
)
TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_VERSION_LABEL: Final = "tool_life.r4.qual.v1"

_HANDLER_SANCTION_MARKER: Final = (
    "intergrax.tools.providers.http.idempotent_post_qual."
    "HttpIdempotentPostQualHandler.v1"
)
TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_HANDLER_CONTENT_DIGEST: Final = hashlib.sha256(
    _HANDLER_SANCTION_MARKER.encode("utf-8"),
).hexdigest()


def build_sanctioned_http_idempotent_post_qual_activation(
    *,
    logical_tool_id: str,
) -> ToolRuntimeActivationMetadata:
    return ToolRuntimeActivationMetadata(
        catalog_source_id=TOOL_LIFE_HTTP_CLASS_B_QUALIFICATION_CATALOG_SOURCE_ID,
        logical_tool_id=logical_tool_id,
        package_reference=TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_PACKAGE_REFERENCE,
        version_label=TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_VERSION_LABEL,
        content_digest=TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_HANDLER_CONTENT_DIGEST,
    )


def registered_tool_matches_sanctioned_http_idempotent_post_qual(
    registered: RegisteredTool,
) -> bool:
    from intergrax.tools.providers.http.idempotent_post_qual import (
        HttpIdempotentPostQualHandler,
    )

    if type(registered.handler) is not HttpIdempotentPostQualHandler:
        return False
    activation = registered.activation
    if activation is None:
        return False
    expected = build_sanctioned_http_idempotent_post_qual_activation(
        logical_tool_id=registered.contract.tool_id,
    )
    return (
        activation.catalog_source_id == expected.catalog_source_id
        and activation.logical_tool_id == expected.logical_tool_id
        and activation.package_reference == expected.package_reference
        and activation.version_label == expected.version_label
        and activation.content_digest == expected.content_digest
    )


__all__ = [
    "TOOL_LIFE_HTTP_CLASS_B_QUALIFICATION_CATALOG_SOURCE_ID",
    "TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_HANDLER_CONTENT_DIGEST",
    "TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_PACKAGE_REFERENCE",
    "TOOL_LIFE_HTTP_IDEMPOTENT_POST_QUAL_VERSION_LABEL",
    "build_sanctioned_http_idempotent_post_qual_activation",
    "registered_tool_matches_sanctioned_http_idempotent_post_qual",
]
