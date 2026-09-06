# © Artur Czarnecki. All rights reserved.

"""Reusable Agent Platform admin test doubles and authorization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.profile_resolution.activation_store import (
    InMemoryActiveEffectiveProfileRevisionStore,
)
from intergrax.applications._shared.profile_resolution.execution_pinning import (
    InMemoryEffectiveProfileExecutionPinningStore,
)
from intergrax.applications._shared.profile_resolution.store import (
    InMemoryEffectiveProfileRevisionStore,
)
from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadata
from intergrax.agent_distribution.catalog import AgentCatalogEntry
from intergrax.agent_distribution.materialization import MaterializationOutput
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

ADMIN_TEST_MATERIALIZATION_ARTIFACT_DIGEST = "sha256:" + ("d" * 64)

_TEST_PRINCIPAL = RequestIdentity(
    tenant_id="tenant-test",
    user_id="admin-1",
    principal_type=PrincipalType.USER,
    auth_subject="admin-1",
)


@dataclass
class _AllowEvaluator:
    def evaluate(self, request: object) -> PolicyDecision:
        del request
        return PolicyDecision(action=PolicyAction.ALLOW, reason="test_allow")


def admin_test_principal() -> RequestIdentity:
    return _TEST_PRINCIPAL


def allow_mutation_boundary() -> ControlPlaneMutationAuthorizationBoundary:
    return ControlPlaneMutationAuthorizationBoundary(evaluator=_AllowEvaluator())


class _LifecycleProofDurableRevisionStore(InMemoryEffectiveProfileRevisionStore):
    @property
    def is_durable(self) -> bool:
        return True


class _LifecycleProofDurablePinningStore(InMemoryEffectiveProfileExecutionPinningStore):
    @property
    def is_durable(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class LifecycleProofDurableProfileStores:
    revision_store: _LifecycleProofDurableRevisionStore
    pinning_store: _LifecycleProofDurablePinningStore
    active_store: InMemoryActiveEffectiveProfileRevisionStore


def lifecycle_proof_durable_profile_stores(
    runtime_root: Path,
) -> LifecycleProofDurableProfileStores:
    """Process-local durable-marked profile stores for strict lifecycle execution proofs."""
    del runtime_root
    return LifecycleProofDurableProfileStores(
        revision_store=_LifecycleProofDurableRevisionStore(),
        pinning_store=_LifecycleProofDurablePinningStore(),
        active_store=InMemoryActiveEffectiveProfileRevisionStore(),
    )


class AgentProjectMetadataTestProvider:
    def __init__(self, records: dict[str, AgentProjectMetadata]) -> None:
        self._records = records

    def get_metadata(self, metadata_ref: str) -> AgentProjectMetadata | None:
        return self._records.get(metadata_ref)


class DeterministicAgentDistributionAdapter:
    topology = MaterializationTopology.OCI_IMAGE
    materializer_id = "intergrax.admin-test"
    materializer_version = "1.0.0"

    def materialize(self, materialization_input: object) -> MaterializationOutput:
        del materialization_input
        return MaterializationOutput(
            materialization_artifact_digest=ADMIN_TEST_MATERIALIZATION_ARTIFACT_DIGEST,
            artifact_locator="test://artifact",
            health_check_evidence_ref="test://health",
            runtime_graph_manifest_path=".intergrax-runtime-graph.json",
            topology=self.topology,
        )


class FakeAgentCatalog:
    def __init__(self, entries: list[AgentCatalogEntry]) -> None:
        self._entries = entries

    @property
    def catalog_source_id(self) -> str:
        return "builtin-1"

    def list_entries(self, filters: object | None = None) -> list[AgentCatalogEntry]:
        del filters
        return list(self._entries)

    def resolve_package(
        self, entry: AgentCatalogEntry, *, version_selector: str
    ) -> object:
        del entry, version_selector
        raise NotImplementedError

    def health(self) -> None:
        return None
