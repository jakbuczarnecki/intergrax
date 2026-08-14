# © Artur Czarnecki. All rights reserved.

"""AP-8 runtime materialization coordinator tests."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadata
from intergrax.agent_distribution.dependency import (
    CandidateDependencySpecification,
    DependencyResolverInput,
    InstalledAgentPackageRequirement,
    MaterializedLockPackage,
    RepositoryDependencyDeclaration,
)
from intergrax.agent_distribution.errors import (
    MaterializationError,
    MaterializationInputConflict,
    MaterializationUnsupportedTopology,
)
from intergrax.agent_distribution.in_memory_stores import InMemoryAgentArtifactMetadataStore
from intergrax.agent_distribution.materialization import (
    ApplicationBuildContext,
    MaterializationInput,
    MaterializationOutput,
)
from intergrax.agent_distribution.materialization_adapters import (
    FakeRuntimeMaterializationAdapter,
    UnsupportedVenvBundleMaterializationAdapter,
)
from intergrax.agent_distribution.materialization_service import RuntimeMaterializationService
from intergrax.agent_distribution.package_artifact_provider import (
    FilesystemArtifactStoreRefResolver,
    MetadataBackedPackageArtifactProvider,
    PackageArtifactProvider,
    sha256_file_digest,
)
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_graph_service import (
    CandidateRuntimeGraphBuilder,
    CandidateRuntimeGraphValidator,
)
from intergrax.agent_distribution.runtime_lock import MaterializedRuntimeLockProducer
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.resolver import ResolvedDependencyClosure
from intergrax.agent_distribution.stores import AgentArtifactMetadata

_DIGEST_B = "sha256:" + ("b" * 64)
_REQUESTS_DIGEST = "sha256:" + ("c" * 64)
_AGENT_A = "intergrax-local-search-agent"
_APP_ID = "local_workspace_application"
_RELEASE = "rel-ap8"
_ENV = "env-prod"
_PLATFORM = "0.1.0"


class _InMemoryMetadataProvider:
    def __init__(self, records: dict[str, AgentProjectMetadata]) -> None:
        self._records = records

    def get_metadata(self, metadata_ref: str) -> AgentProjectMetadata | None:
        return self._records.get(metadata_ref)


@dataclass
class _RecordingAdapter:
    topology: MaterializationTopology = MaterializationTopology.OCI_IMAGE
    materializer_id: str = "intergrax.recording"
    materializer_version: str = "1.0.0"
    output: MaterializationOutput | None = None
    invoked: bool = False

    def materialize(self, materialization_input: MaterializationInput) -> MaterializationOutput:
        self.invoked = True
        assert self.output is not None
        return self.output


def _write_agent_wheel(tmp_path: Path) -> tuple[Path, str]:
    artifact_root = tmp_path / "artifact-store"
    wheel = artifact_root / "intergrax_local_search_agent-1.0.0-py3-none-any.whl"
    wheel.parent.mkdir(parents=True, exist_ok=True)
    wheel.write_bytes(b"PK\x03\x04ap8-agent-wheel-bytes")
    return wheel, sha256_file_digest(wheel)


def _artifact_provider_for_wheel(
    tmp_path: Path,
    wheel: Path,
    package_digest: str,
) -> PackageArtifactProvider:
    artifact_root = wheel.parent
    rel = wheel.relative_to(artifact_root).as_posix()
    store = InMemoryAgentArtifactMetadataStore()
    store.persist_metadata(
        AgentArtifactMetadata(
            package_digest=package_digest,
            artifact_store_ref=f"file://{rel}",
            distribution_package_id=_AGENT_A,
            agent_project_metadata_ref="meta://search",
        )
    )
    return MetadataBackedPackageArtifactProvider(
        metadata_store=store,
        ref_resolver=FilesystemArtifactStoreRefResolver(root=artifact_root),
    )


def _lock_and_graph(
    agent_digest: str,
    *,
    requests_digest: str | None = _REQUESTS_DIGEST,
) -> tuple[object, object, EffectiveRoster]:
    spec = CandidateDependencySpecification(
        application_release_id=_RELEASE,
        platform_version=_PLATFORM,
        repository_declaration=RepositoryDependencyDeclaration(
            application_release_id=_RELEASE,
            direct_dependencies=("requests>=2.32", "Intergrax-ai"),
        ),
        agent_packages=(
            InstalledAgentPackageRequirement(
                distribution_package_id=_AGENT_A,
                package_digest=agent_digest,
                agent_project_metadata_ref="meta://search",
            ),
        ),
    )
    resolver_input = DependencyResolverInput(
        specification=spec,
        resolver_algorithm_id="intergrax.test-resolver",
        resolver_algorithm_version="1.0.0",
    )
    resolved = ResolvedDependencyClosure(
        resolver_algorithm_id="intergrax.test-resolver",
        resolver_algorithm_version="1.0.0",
        python_version="3.12",
        packages=(
            MaterializedLockPackage(
                distribution_name=_AGENT_A,
                version="1.0.0",
                package_digest=agent_digest,
            ),
            MaterializedLockPackage(
                distribution_name="requests",
                version="2.32.0",
                package_digest=requests_digest,
            ),
        ),
    )
    lock = MaterializedRuntimeLockProducer().produce(resolver_input, resolved)
    roster = EffectiveRoster(
        application_id=_APP_ID,
        application_environment_id=_ENV,
        manifest_release_id=_RELEASE,
        binding_revisions=(1,),
        entries=(
            EffectiveRosterEntry(
                logical_agent_id="search",
                installation_slot_id="slot-search",
                package_digest=agent_digest,
                distribution_package_id=_AGENT_A,
                effective_enablement=True,
            ),
        ),
    ).with_revision_id()
    provider = _InMemoryMetadataProvider(
        {"meta://search": AgentProjectMetadata(distribution_package_id=_AGENT_A, dependencies=())}
    )
    graph = CandidateRuntimeGraphBuilder(provider).build(
        lock=lock,
        effective_roster=roster,
        repository_declaration=spec.repository_declaration,
        agent_metadata_refs={_AGENT_A: "meta://search"},
    )
    graph = CandidateRuntimeGraphValidator().validate(
        lock=lock,
        effective_roster=roster,
        graph=graph,
    )
    return lock, graph, roster


def _build_fixture(
    tmp_path: Path,
) -> tuple[MaterializationInput, _RecordingAdapter, PackageArtifactProvider, str]:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    app_root = source_root / "applications" / _APP_ID
    agent_wheel, agent_digest = _write_agent_wheel(tmp_path)
    artifact_provider = _artifact_provider_for_wheel(tmp_path, agent_wheel, agent_digest)
    for path, content in (
        (source_root / "pyproject.toml", "[project]\nname='Intergrax-ai'\n"),
        (source_root / "uv.lock", "# lock\n"),
        (app_root / "pyproject.toml", "[project]\nname='app'\n"),
        (source_root / "intergrax" / "marker.txt", "platform\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    lock, graph, roster = _lock_and_graph(agent_digest)
    revision = RuntimeRevision(
        runtime_revision_id="rev-ap8-1",
        application_id=_APP_ID,
        application_environment_id=_ENV,
        application_release_id=_RELEASE,
        platform_version=_PLATFORM,
        effective_roster_revision_id=roster.effective_roster_revision_id,
        installed_agent_package_digests=(agent_digest,),
        materialized_runtime_lock_id=lock.lock_id,
        materialized_runtime_lock_digest=lock.lock_digest,
        runtime_graph_digest=graph.runtime_graph_digest,
        materialization_topology=MaterializationTopology.OCI_IMAGE,
        revision_state=RuntimeRevisionState.CANDIDATE,
    )
    build_context = ApplicationBuildContext(
        application_id=_APP_ID,
        application_release_id=_RELEASE,
        application_environment_id=_ENV,
        source_context_root=str(source_root),
        platform_version=_PLATFORM,
        python_version="3.12",
        output_root=str(output_root),
        application_source_root=f"applications/{_APP_ID}",
        agent_source_roots=((_AGENT_A, "agents/local_search_agent"),),
    )
    materialization_input = MaterializationInput(
        runtime_revision=revision,
        materialized_runtime_lock=lock,
        candidate_runtime_graph=graph,
        effective_roster=roster,
        application_build_context=build_context,
    )
    adapter = _RecordingAdapter(
        output=MaterializationOutput(
            materialization_artifact_digest=agent_digest,
            artifact_locator="test://artifact",
            health_check_evidence_ref="test://health",
            runtime_graph_manifest_path=".intergrax-runtime-graph.json",
            topology=MaterializationTopology.OCI_IMAGE,
        )
    )
    return materialization_input, adapter, artifact_provider, agent_digest


def test_valid_inputs_invoke_adapter(tmp_path: Path) -> None:
    materialization_input, adapter, _provider, _digest = _build_fixture(tmp_path)
    service = RuntimeMaterializationService({MaterializationTopology.OCI_IMAGE: adapter})
    output = service.materialize(materialization_input)
    assert adapter.invoked is True
    assert output.artifact_locator == "test://artifact"


def test_lock_graph_mismatch_fails_closed(tmp_path: Path) -> None:
    materialization_input, adapter, _provider, _digest = _build_fixture(tmp_path)
    tampered_graph = materialization_input.candidate_runtime_graph.model_copy(
        update={"materialized_runtime_lock_id": "sha256:" + ("f" * 64)}
    )
    service = RuntimeMaterializationService({MaterializationTopology.OCI_IMAGE: adapter})
    with pytest.raises(MaterializationInputConflict):
        service.materialize(
            materialization_input.model_copy(
                update={"candidate_runtime_graph": tampered_graph}
            )
        )
    assert adapter.invoked is False


def test_roster_revision_mismatch_fails_closed(tmp_path: Path) -> None:
    materialization_input, adapter, _provider, _digest = _build_fixture(tmp_path)
    bad_revision = materialization_input.runtime_revision.model_copy(
        update={"effective_roster_revision_id": "sha256:" + ("c" * 64)}
    )
    service = RuntimeMaterializationService({MaterializationTopology.OCI_IMAGE: adapter})
    with pytest.raises(MaterializationInputConflict):
        service.materialize(
            materialization_input.model_copy(update={"runtime_revision": bad_revision})
        )


def test_release_mismatch_fails_closed(tmp_path: Path) -> None:
    materialization_input, adapter, _provider, _digest = _build_fixture(tmp_path)
    bad_context = materialization_input.application_build_context.model_copy(
        update={"application_release_id": "rel-other"}
    )
    service = RuntimeMaterializationService({MaterializationTopology.OCI_IMAGE: adapter})
    with pytest.raises(MaterializationInputConflict):
        service.materialize(
            materialization_input.model_copy(
                update={"application_build_context": bad_context}
            )
        )


def test_environment_mismatch_fails_closed(tmp_path: Path) -> None:
    materialization_input, adapter, _provider, _digest = _build_fixture(tmp_path)
    bad_context = materialization_input.application_build_context.model_copy(
        update={"application_environment_id": "env-other"}
    )
    service = RuntimeMaterializationService({MaterializationTopology.OCI_IMAGE: adapter})
    with pytest.raises(MaterializationInputConflict):
        service.materialize(
            materialization_input.model_copy(
                update={"application_build_context": bad_context}
            )
        )


def test_unsupported_topology_raises_typed_failure(tmp_path: Path) -> None:
    materialization_input, _adapter, artifact_provider, _digest = _build_fixture(tmp_path)
    service = RuntimeMaterializationService(
        {
            MaterializationTopology.OCI_IMAGE: FakeRuntimeMaterializationAdapter(
                package_artifact_provider=artifact_provider
            )
        }
    )
    with pytest.raises(MaterializationUnsupportedTopology):
        service.materialize(
            materialization_input,
            topology=MaterializationTopology.SANDBOX_SIDECAR,
        )


def test_adapter_wrong_topology_fails_closed(tmp_path: Path) -> None:
    materialization_input, adapter, _provider, agent_digest = _build_fixture(tmp_path)
    adapter.output = MaterializationOutput(
        materialization_artifact_digest=agent_digest,
        artifact_locator="test://artifact",
        runtime_graph_manifest_path=".intergrax-runtime-graph.json",
        topology=MaterializationTopology.VENV_BUNDLE,
    )
    service = RuntimeMaterializationService({MaterializationTopology.OCI_IMAGE: adapter})
    with pytest.raises(MaterializationError):
        service.materialize(materialization_input)


def test_empty_artifact_digest_fails_closed(tmp_path: Path) -> None:
    materialization_input, adapter, _provider, _digest = _build_fixture(tmp_path)
    adapter.output = MaterializationOutput(
        materialization_artifact_digest="not-a-digest",
        artifact_locator="test://artifact",
        runtime_graph_manifest_path=".intergrax-runtime-graph.json",
        topology=MaterializationTopology.OCI_IMAGE,
    )
    service = RuntimeMaterializationService({MaterializationTopology.OCI_IMAGE: adapter})
    with pytest.raises(MaterializationError):
        service.materialize(materialization_input)


def test_runtime_graph_manifest_path_required(tmp_path: Path) -> None:
    materialization_input, adapter, _provider, agent_digest = _build_fixture(tmp_path)
    adapter.output = MaterializationOutput.model_construct(
        materialization_artifact_digest=agent_digest,
        artifact_locator="test://artifact",
        runtime_graph_manifest_path="   ",
        topology=MaterializationTopology.OCI_IMAGE,
    )
    service = RuntimeMaterializationService({MaterializationTopology.OCI_IMAGE: adapter})
    with pytest.raises(MaterializationError):
        service.materialize(materialization_input)


def test_materialization_does_not_activate_revision(tmp_path: Path) -> None:
    materialization_input, _adapter, artifact_provider, _digest = _build_fixture(tmp_path)
    service = RuntimeMaterializationService(
        {
            MaterializationTopology.OCI_IMAGE: FakeRuntimeMaterializationAdapter(
                package_artifact_provider=artifact_provider
            )
        }
    )
    before = materialization_input.runtime_revision.revision_state
    service.materialize(materialization_input)
    assert materialization_input.runtime_revision.revision_state is before


def test_fake_adapter_digest_is_deterministic(tmp_path: Path) -> None:
    materialization_input, _adapter, artifact_provider, _digest = _build_fixture(tmp_path)
    service = RuntimeMaterializationService(
        {
            MaterializationTopology.OCI_IMAGE: FakeRuntimeMaterializationAdapter(
                package_artifact_provider=artifact_provider
            )
        }
    )
    first = service.materialize(materialization_input)
    second = service.materialize(materialization_input)
    assert first.materialization_artifact_digest == second.materialization_artifact_digest


def test_different_graph_changes_fake_digest(tmp_path: Path) -> None:
    materialization_input, _adapter, artifact_provider, _digest = _build_fixture(tmp_path)
    service = RuntimeMaterializationService(
        {
            MaterializationTopology.OCI_IMAGE: FakeRuntimeMaterializationAdapter(
                package_artifact_provider=artifact_provider
            )
        }
    )
    baseline = service.materialize(materialization_input)
    other_graph = materialization_input.candidate_runtime_graph.model_copy(
        update={"direct_third_party_distributions": ()}
    ).with_content_identity()
    other_revision = materialization_input.runtime_revision.model_copy(
        update={"runtime_graph_digest": other_graph.runtime_graph_digest}
    )
    other_input = materialization_input.model_copy(
        update={
            "candidate_runtime_graph": other_graph,
            "runtime_revision": other_revision,
        }
    )
    changed = service.materialize(other_input)
    assert changed.materialization_artifact_digest != baseline.materialization_artifact_digest


@pytest.mark.parametrize(
    ("field", "update", "match"),
    [
        ("runtime_revision", {"application_id": "app-other"}, "application_id"),
        ("effective_roster", {"application_id": "app-other"}, "application_id"),
        ("candidate_runtime_graph", {"application_id": "app-other"}, "application_id"),
        (
            "application_build_context",
            {"application_id": "app-other"},
            "application_id",
        ),
        (
            "runtime_revision",
            {"application_environment_id": "env-other"},
            "environment",
        ),
        (
            "effective_roster",
            {"application_environment_id": "env-other"},
            "environment",
        ),
        (
            "application_build_context",
            {"application_environment_id": "env-other"},
            "environment",
        ),
    ],
)
def test_materialization_rejects_application_identity_mismatch(
    tmp_path: Path,
    field: str,
    update: dict[str, str],
    match: str,
) -> None:
    materialization_input, _, _, _ = _build_fixture(tmp_path)
    original = getattr(materialization_input, field)
    mutated = original.model_copy(update=update)
    tampered = materialization_input.model_copy(update={field: mutated})
    with pytest.raises(MaterializationInputConflict, match=match):
        RuntimeMaterializationService._validate_input_consistency(tampered)


def test_manifest_references_graph_digest_and_lock_id(tmp_path: Path) -> None:
    materialization_input, _adapter, artifact_provider, agent_digest = _build_fixture(tmp_path)
    service = RuntimeMaterializationService(
        {
            MaterializationTopology.OCI_IMAGE: FakeRuntimeMaterializationAdapter(
                package_artifact_provider=artifact_provider
            )
        }
    )
    service.materialize(materialization_input)
    candidate_dir = (
        Path(materialization_input.application_build_context.output_root)
        / f"candidate-{materialization_input.runtime_revision.runtime_revision_id}"
    )
    manifest = json.loads(
        (candidate_dir / ".intergrax-runtime-graph.json").read_text(encoding="utf-8")
    )
    assert manifest["runtime_graph_digest"] == materialization_input.candidate_runtime_graph.runtime_graph_digest
    assert manifest["materialized_runtime_lock_id"] == materialization_input.materialized_runtime_lock.lock_id
    assert manifest["enabled_roster_agents"][0]["package_digest"] == agent_digest
    blob = json.dumps(manifest)
    assert "secret" not in blob.lower()
    assert "password" not in blob.lower()


def test_venv_topology_is_explicitly_unsupported(tmp_path: Path) -> None:
    materialization_input, _adapter, _provider, _digest = _build_fixture(tmp_path)
    revision = materialization_input.runtime_revision.model_copy(
        update={"materialization_topology": MaterializationTopology.VENV_BUNDLE}
    )
    service = RuntimeMaterializationService(
        {
            MaterializationTopology.VENV_BUNDLE: UnsupportedVenvBundleMaterializationAdapter(),
        }
    )
    with pytest.raises(MaterializationUnsupportedTopology):
        service.materialize(
            materialization_input.model_copy(update={"runtime_revision": revision}),
            topology=MaterializationTopology.VENV_BUNDLE,
        )


def test_agent_distribution_import_boundary_for_ap8_modules() -> None:
    repo = Path(__file__).resolve().parents[3]
    package_root = repo / "intergrax" / "agent_distribution"
    modules = (
        "materialization_service.py",
        "materialization_adapters.py",
        "runtime_context_staging.py",
    )
    agent_dirs = {
        p.name
        for p in (repo / "agents").iterdir()
        if p.is_dir() and (p / "__init__.py").is_file() and not p.name.startswith("_")
    }
    app_dirs = {
        p.name
        for p in (repo / "applications").iterdir()
        if p.is_dir() and (p / "pyproject.toml").is_file()
    }
    violations: list[str] = []
    for module in modules:
        path = package_root / module
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            modules_imported: list[str] = []
            if isinstance(node, ast.Import):
                modules_imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules_imported.append(node.module)
            for imported in modules_imported:
                top = imported.split(".", 1)[0]
                if top == "agents" or top in agent_dirs:
                    violations.append(f"{module} imports {imported}")
                if top == "applications" or top in app_dirs:
                    violations.append(f"{module} imports {imported}")
    assert not violations, "\n".join(violations)
