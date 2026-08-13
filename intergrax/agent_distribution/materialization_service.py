# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime materialization coordinator (AP-8 §19)."""

from __future__ import annotations

import re
from collections.abc import Mapping

from intergrax.agent_distribution.errors import (
    MaterializationError,
    MaterializationInputConflict,
    MaterializationUnsupportedTopology,
)
from intergrax.agent_distribution.materialization import MaterializationInput, MaterializationOutput
from intergrax.agent_distribution.materialization_adapters import RuntimeMaterializationAdapter
from intergrax.agent_distribution.runtime_revision import MaterializationTopology, RuntimeRevisionState

_ARTIFACT_DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")


class RuntimeMaterializationService:
    """Validate inputs, select adapter, and return canonical materialization output."""

    def __init__(
        self,
        adapters: Mapping[MaterializationTopology, RuntimeMaterializationAdapter],
    ) -> None:
        self._adapters = dict(adapters)

    def materialize(
        self,
        materialization_input: MaterializationInput,
        *,
        topology: MaterializationTopology | None = None,
    ) -> MaterializationOutput:
        """Materialize one candidate artifact without activation or registry mutation."""
        self._validate_input_consistency(materialization_input)
        selected_topology = topology or materialization_input.runtime_revision.materialization_topology
        if selected_topology is None:
            raise MaterializationError("materialization topology is required")
        adapter = self._adapters.get(selected_topology)
        if adapter is None:
            raise MaterializationUnsupportedTopology(
                f"no adapter registered for topology {selected_topology.value}"
            )
        if adapter.topology != selected_topology:
            raise MaterializationError("adapter topology mismatch")

        try:
            raw_output = adapter.materialize(materialization_input)
        except MaterializationUnsupportedTopology:
            raise
        except MaterializationError:
            raise
        except Exception as exc:
            raise MaterializationError(str(exc)) from exc

        return self._validate_output(raw_output, expected_topology=selected_topology)

    @staticmethod
    def _validate_input_consistency(materialization_input: MaterializationInput) -> None:
        revision = materialization_input.runtime_revision
        lock = materialization_input.materialized_runtime_lock
        graph = materialization_input.candidate_runtime_graph
        roster = materialization_input.effective_roster
        build_context = materialization_input.application_build_context

        if revision.revision_state not in {
            RuntimeRevisionState.CANDIDATE,
            RuntimeRevisionState.VALIDATED,
        }:
            raise MaterializationInputConflict(
                "materialization requires candidate or validated runtime revision"
            )

        if lock.lock_id is None or lock.lock_digest is None:
            raise MaterializationInputConflict("materialized runtime lock requires content identity")
        if graph.runtime_graph_digest is None:
            raise MaterializationInputConflict("candidate runtime graph requires content identity")
        if graph.materialized_runtime_lock_id != lock.lock_id:
            raise MaterializationInputConflict(
                "graph materialized_runtime_lock_id does not match lock.lock_id"
            )
        if roster.effective_roster_revision_id is None:
            raise MaterializationInputConflict("effective roster requires revision identity")

        if revision.application_environment_id != build_context.application_environment_id:
            raise MaterializationInputConflict("runtime revision environment mismatch")
        if revision.application_release_id != build_context.application_release_id:
            raise MaterializationInputConflict("runtime revision release mismatch")
        if revision.platform_version != build_context.platform_version:
            raise MaterializationInputConflict("runtime revision platform version mismatch")
        if revision.effective_roster_revision_id != roster.effective_roster_revision_id:
            raise MaterializationInputConflict("runtime revision roster revision mismatch")
        if graph.application_id != build_context.application_id:
            raise MaterializationInputConflict("runtime graph application_id mismatch")

        if revision.materialized_runtime_lock_id is not None:
            if revision.materialized_runtime_lock_id != lock.lock_id:
                raise MaterializationInputConflict("runtime revision lock id mismatch")
        if revision.materialized_runtime_lock_digest is not None:
            if revision.materialized_runtime_lock_digest != lock.lock_digest:
                raise MaterializationInputConflict("runtime revision lock digest mismatch")
        if revision.runtime_graph_digest is not None:
            if revision.runtime_graph_digest != graph.runtime_graph_digest:
                raise MaterializationInputConflict("runtime revision graph digest mismatch")

        enabled_digests = sorted(
            entry.package_digest
            for entry in roster.entries
            if entry.effective_enablement
        )
        if revision.installed_agent_package_digests:
            if sorted(revision.installed_agent_package_digests) != enabled_digests:
                raise MaterializationInputConflict(
                    "runtime revision installed digests do not match enabled roster"
                )
        closure_digests = {entry.package_digest for entry in lock.agent_closure}
        for digest in enabled_digests:
            if digest not in closure_digests:
                raise MaterializationInputConflict(
                    "enabled roster digest missing from lock agent closure"
                )

    @staticmethod
    def _validate_output(
        output: MaterializationOutput,
        *,
        expected_topology: MaterializationTopology,
    ) -> MaterializationOutput:
        if output.topology != expected_topology:
            raise MaterializationError("adapter returned unexpected topology")
        if not output.materialization_artifact_digest.strip():
            raise MaterializationError("materialization artifact digest is required")
        if not output.artifact_locator.strip():
            raise MaterializationError("artifact locator is required")
        if not output.runtime_graph_manifest_path.strip():
            raise MaterializationError("runtime graph manifest path is required")
        if not _ARTIFACT_DIGEST_RE.match(output.materialization_artifact_digest.strip().lower()):
            raise MaterializationError("materialization artifact digest must be sha256:<64 hex>")
        return output
