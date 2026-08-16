# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Generic reader of package-owned agent capability metadata — non-executable."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from intergrax.agent_distribution.agent_capability_metadata import (
    AgentCapabilityDescriptor,
    merge_agent_capability_descriptors,
)
from intergrax.agent_distribution.agent_project_metadata import (
    AgentProjectMetadata,
    parse_agent_project_pyproject,
    project_agent_capability_descriptors,
)


class PackageAgentCapabilityMetadataProvider:
    """Aggregate ``AgentCapabilityDescriptor`` rows from package pyproject metadata.

    Callers supply package roots or already-parsed ``AgentProjectMetadata``.
    This type does not author contract identities, versions, or capabilities.
    """

    def __init__(
        self,
        *,
        package_roots: Sequence[str | Path] = (),
        project_metadata: Sequence[AgentProjectMetadata] = (),
    ) -> None:
        self._package_roots = tuple(Path(root) for root in package_roots)
        self._project_metadata = tuple(project_metadata)

    def list_agent_capability_descriptors(self) -> Sequence[AgentCapabilityDescriptor]:
        collected: list[AgentCapabilityDescriptor] = []
        for metadata in self._project_metadata:
            collected.extend(project_agent_capability_descriptors(metadata))
        for root in self._package_roots:
            pyproject_path = root / "pyproject.toml" if root.is_dir() else root
            text = pyproject_path.read_text(encoding="utf-8")
            collected.extend(
                project_agent_capability_descriptors(parse_agent_project_pyproject(text))
            )
        return merge_agent_capability_descriptors(collected)


BuiltinAgentCapabilityMetadataProvider = PackageAgentCapabilityMetadataProvider
