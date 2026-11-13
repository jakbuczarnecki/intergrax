# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Durable references from context (and tool results) to artifacts (PDC-D8-R012, R013).

A reference names an artifact by tenant, class, id, version and content digest. It lives in the
context as a short marker, so it survives fragment compaction (reference fragments are mandatory
and compaction never touches the artifact store). Resolving always goes through the governed
access port and re-checks the digest.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.context.contracts import (
    ContextAuthorityClass,
    ContextFragment,
    ContextFragmentScopeRef,
    ContextFragmentSource,
)
from intergrax.contracts.artifact_ref import (
    ArtifactClass,
    ArtifactDigestMismatchError,
    is_valid_artifact_digest,
)
from intergrax.runtime.nexus.artifacts.governed_access import (
    ArtifactAccessContext,
    ArtifactReadRequest,
    GovernedArtifactReadPort,
)

_SAFE_PART: Final = re.compile(r"^[^/@#\s]+$")
_MARKER_RE: Final = re.compile(
    r"artifact://(?P<tenant>[^/@#\s]+)/(?P<cls>[a-z_]+)/(?P<id>[^/@#\s]+)@(?P<ver>\d+)#(?P<digest>sha256:[0-9a-f]{64})"
)
REFERENCE_FRAGMENT_PREFIX: Final = "[artifact reference] "
REFERENCE_TOOL_NAME: Final = "artifact.reference"


class ArtifactReferenceToken(BaseModel):
    """Self-verifying reference to one artifact version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    artifact_class: ArtifactClass
    artifact_id: str
    version: int = Field(ge=1)
    content_digest: str

    @field_validator("tenant_id", "artifact_id")
    @classmethod
    def _safe(cls, value: str) -> str:
        if not _SAFE_PART.match(value):
            raise ValueError("must be non-empty and free of '/', '@', '#' and whitespace")
        return value

    @field_validator("content_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not is_valid_artifact_digest(value):
            raise ValueError("content_digest must match sha256:<64 lowercase hex>")
        return value

    def to_marker(self) -> str:
        return (
            f"artifact://{self.tenant_id}/{self.artifact_class.value}/{self.artifact_id}"
            f"@{self.version}#{self.content_digest}"
        )

    @classmethod
    def from_marker(cls, marker: str) -> ArtifactReferenceToken:
        found = _MARKER_RE.fullmatch(marker.strip())
        if found is None:
            raise ValueError("not an artifact reference marker")
        return cls._from_match(found)

    @classmethod
    def find_in_text(cls, text: str) -> ArtifactReferenceToken:
        found = _MARKER_RE.search(text)
        if found is None:
            raise ValueError("no artifact reference marker in text")
        return cls._from_match(found)

    @classmethod
    def _from_match(cls, found: re.Match[str]) -> ArtifactReferenceToken:
        return cls(
            tenant_id=found.group("tenant"),
            artifact_class=ArtifactClass(found.group("cls")),
            artifact_id=found.group("id"),
            version=int(found.group("ver")),
            content_digest=found.group("digest"),
        )


def reference_fragment(
    token: ArtifactReferenceToken,
    *,
    fragment_id: str,
    tool_call_id: str | None = None,
    tool_name: str = REFERENCE_TOOL_NAME,
) -> ContextFragment:
    """A mandatory context fragment that carries the reference (never compacted away).

    The fragment is a tool observation, so it names the originating call (``tool_call_id`` and
    ``tool_name`` metadata): provider canonicalization rejects an unlinked tool output. Without an
    explicit call id the link is the stable per-artifact id ``artifact-ref:<artifact_id>``.
    """
    content = f"{REFERENCE_FRAGMENT_PREFIX}{token.to_marker()}"
    return ContextFragment(
        fragment_id=fragment_id,
        source=ContextFragmentSource.TOOL_OUTPUT,
        source_id=f"artifact:{token.artifact_id}",
        content=content,
        token_estimate=max(1, len(content) // 4),
        relevance_score=1.0,
        freshness_score=1.0,
        confidence_score=1.0,
        mandatory=True,
        metadata={
            "artifact_ref": token.model_dump(mode="json"),
            "tool_call_id": tool_call_id or f"artifact-ref:{token.artifact_id}",
            "tool_name": tool_name,
        },
        authority_class=ContextAuthorityClass.TOOL_OBSERVATION,
        scope_ref=ContextFragmentScopeRef(tenant_id=token.tenant_id),
    )


def token_from_fragment(fragment: ContextFragment) -> ArtifactReferenceToken:
    """Recover the reference from a fragment's content (the metadata is not relied on)."""
    return ArtifactReferenceToken.find_in_text(fragment.content)


class ArtifactReferenceResolver:
    """Resolves references through the governed access port; verifies the digest."""

    def __init__(self, port: GovernedArtifactReadPort) -> None:
        self._port = port

    def resolve(self, context: ArtifactAccessContext, token: ArtifactReferenceToken) -> bytes:
        result = self._port.read(
            context,
            ArtifactReadRequest(
                artifact_class=token.artifact_class,
                tenant_id=token.tenant_id,
                artifact_id=token.artifact_id,
                version=token.version,
            ),
        )
        if result.ref.content_digest != token.content_digest:
            raise ArtifactDigestMismatchError("resolved artifact does not match the reference digest")
        return result.content

    def resolve_marker(self, context: ArtifactAccessContext, marker: str) -> bytes:
        return self.resolve(context, ArtifactReferenceToken.from_marker(marker))
