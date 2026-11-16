# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Artifact taxonomy: four classes, four owners, no global storage engine (ADR-PDC-08).

This module only *declares* ownership. It stores nothing and enforces nothing beyond the
consistency gate; every class keeps its own owner and its own persistence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from intergrax.contracts.artifact_ref import ArtifactClass


@dataclass(frozen=True)
class ArtifactClassOwner:
    """The single owner of one artifact class."""

    artifact_class: ArtifactClass
    owner_id: str
    owner_path: str
    persistence: str


ARTIFACT_CLASS_OWNERS: Mapping[ArtifactClass, ArtifactClassOwner] = MappingProxyType(
    {
        ArtifactClass.BUSINESS: ArtifactClassOwner(
            artifact_class=ArtifactClass.BUSINESS,
            owner_id="collaborative_work",
            owner_path="intergrax/collaborative_work/artifact_service.py",
            persistence="Collaborative Work repositories and ArtifactContentStore",
        ),
        ArtifactClass.EXECUTION_EVIDENCE: ArtifactClassOwner(
            artifact_class=ArtifactClass.EXECUTION_EVIDENCE,
            owner_id="evidence_plane",
            owner_path="intergrax/runtime/nexus/artifacts/store_base.py",
            persistence="execution ArtifactStore (in-memory, SQLite)",
        ),
        ArtifactClass.CONTEXT: ArtifactClassOwner(
            artifact_class=ArtifactClass.CONTEXT,
            owner_id="ucl",
            owner_path="intergrax/runtime/context_lifecycle/contracts.py",
            persistence="UCL repositories (context_lifecycle)",
        ),
        ArtifactClass.TOOL_OUTPUT: ArtifactClassOwner(
            artifact_class=ArtifactClass.TOOL_OUTPUT,
            owner_id="tool_runtime",
            owner_path="intergrax/runtime/nexus/tools/tool_runtime.py",
            persistence="spill-to-reference into the execution ArtifactStore, ephemeral retention",
        ),
    }
)

#: Classes the execution ``ArtifactStore`` accepts. Business artifacts stay with Collaborative
#: Work and context artifacts with UCL; the store refuses them instead of becoming a global engine.
STORE_ACCEPTED_CLASSES: frozenset[ArtifactClass] = frozenset(
    {ArtifactClass.EXECUTION_EVIDENCE, ArtifactClass.TOOL_OUTPUT}
)


def owner_of(artifact_class: ArtifactClass) -> ArtifactClassOwner:
    return ARTIFACT_CLASS_OWNERS[artifact_class]


def assert_taxonomy_consistent() -> None:
    """Fail unless every class has exactly one distinct owner."""
    if set(ARTIFACT_CLASS_OWNERS) != set(ArtifactClass):
        raise AssertionError("every artifact class needs exactly one owner")
    owner_ids = [owner.owner_id for owner in ARTIFACT_CLASS_OWNERS.values()]
    if len(set(owner_ids)) != len(owner_ids):
        raise AssertionError("one owner may not own two artifact classes")
    for artifact_class, owner in ARTIFACT_CLASS_OWNERS.items():
        if owner.artifact_class is not artifact_class:
            raise AssertionError(f"owner table entry mismatch for {artifact_class.value}")
    foreign = STORE_ACCEPTED_CLASSES & {ArtifactClass.BUSINESS, ArtifactClass.CONTEXT}
    if foreign:
        raise AssertionError("the execution store must not own business or context artifacts")
