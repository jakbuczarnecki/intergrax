# © Artur Czarnecki. All rights reserved.

"""AC-5 Phase 3 — revision-bound canonical factory E2E through AP-10 projection."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_context_staging import (
    RUNTIME_LOCK_MANIFEST_FILENAME,
    directory_content_digest,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agents.agent_contract import Agent
from intergrax.applications._shared.registry_projection import (
    MaterializedRegistryProjection,
    RegistryProjectionError,
    RegistryProjectionInputBundle,
    build_registry_projection,
)
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    InMemoryRuntimeAgentFactoryResolver,
    RuntimeAgentFactoryResolutionError,
)
from intergrax.applications._shared.venv_bundle_runtime_agent_factory_resolver import (
    build_production_runtime_agent_factory_resolver,
)
from intergrax.applications._shared.wiring import invoke_canonical_agent_factory
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.errors import AgentImportError
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP = "app_ac5_p3"
_ENV = "env-ac5-p3"
_RELEASE = "rel-ac5-p3"
_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_LOCK_ID = "lock-ac5-p3"
_LOCK_DIGEST = "sha256:" + ("c" * 64)
_GRAPH_DIGEST = "sha256:" + ("d" * 64)
_ARTIFACT = "sha256:" + ("e" * 64)
_ROSTER = "sha256:" + ("f" * 64)
_REF_IMMUTABLE = AgentBindingFactoryReference(builder_key="immutable")
_REF_OTHER = AgentBindingFactoryReference(builder_key="other")
_LOCAL_FACTORY_PATH = "local_stub.factory.build_local"
_IMMUTABLE_MARKER = "IMMUTABLE"
_LOCAL_MARKER = "LOCAL"


@dataclass
class _FactoryProbe:
    marker: str
    calls: int = 0
    seen_bindings: list[AgentBinding] = field(default_factory=list)
    seen_contexts: list[ApplicationBuildContext] = field(default_factory=list)

    def factory(
        self,
        ctx: ApplicationBuildContext,
        binding: AgentBinding,
    ) -> EchoAgent:
        self.calls += 1
        self.seen_bindings.append(binding)
        self.seen_contexts.append(ctx)
        return EchoAgent()


class _OtherAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="other-agent",
            name="Other",
            description="stub",
            capabilities=["other"],
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )


def _immutable_factory(probe: _FactoryProbe) -> Callable[..., EchoAgent]:
    def _factory(ctx: ApplicationBuildContext, binding: AgentBinding) -> EchoAgent:
        return probe.factory(ctx, binding)

    return _factory


def _local_manifest_factory(
    _ctx: ApplicationBuildContext,
    _binding: AgentBinding,
) -> EchoAgent:
    raise AssertionError(
        "LOCAL manifest factory must not run in revision-bound production"
    )


def _local_manifest_binding(contract_id: str) -> AgentBinding:
    return AgentBinding.mount(
        EchoAgent,
        contract_id=contract_id,
        factory=_local_manifest_factory,
    )


def _manifest(
    *,
    agents: list[AgentBinding] | None = None,
) -> ApplicationManifest:
    roster_agents = agents or [
        _local_manifest_binding("search"),
        _local_manifest_binding("indexer"),
    ]
    return ApplicationManifest.lab(
        app_id=_APP, name="AC5 Phase 3", agents=roster_agents
    )


def _entry(
    logical_agent_id: str,
    *,
    package_digest: str,
    factory_reference: AgentBindingFactoryReference,
    merged_config: dict[str, object] | None = None,
    effective_enablement: bool = True,
) -> EffectiveRosterEntry:
    return EffectiveRosterEntry(
        logical_agent_id=logical_agent_id,
        installation_slot_id=f"slot-{logical_agent_id}",
        package_digest=package_digest,
        distribution_package_id=f"pkg-{logical_agent_id}",
        effective_enablement=effective_enablement,
        factory_reference=factory_reference,
        manifest_origin_ref=f"manifest:agents/{logical_agent_id}",
        merged_config=merged_config or {},
    )


def _roster(
    entries: tuple[EffectiveRosterEntry, ...],
    *,
    revision_id: str = _ROSTER,
) -> EffectiveRoster:
    return EffectiveRoster(
        application_id=_APP,
        application_environment_id=_ENV,
        manifest_release_id=_RELEASE,
        entries=entries,
        effective_roster_revision_id=revision_id,
    )


def _revision(
    *,
    revision_id: str = "rev-ac5-p3",
    package_digests: tuple[str, ...] = (_DIGEST_A,),
    roster_revision_id: str = _ROSTER,
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_id=_APP,
        application_environment_id=_ENV,
        application_release_id=_RELEASE,
        platform_version="0.1.0",
        effective_roster_revision_id=roster_revision_id,
        installed_agent_package_digests=package_digests,
        materialized_runtime_lock_id=_LOCK_ID,
        materialized_runtime_lock_digest=_LOCK_DIGEST,
        runtime_graph_digest=_GRAPH_DIGEST,
        materialization_artifact_digest=_ARTIFACT,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=RuntimeRevisionState.VALIDATED,
        activated_at=datetime.now(UTC),
    )


def _resolver(
    *registrations: tuple[str, AgentBindingFactoryReference, object],
) -> InMemoryRuntimeAgentFactoryResolver:
    resolver = InMemoryRuntimeAgentFactoryResolver()
    for package_digest, factory_reference, factory in registrations:
        resolver.register(
            package_digest=package_digest,
            factory_reference=factory_reference,
            factory=factory,
        )
    return resolver


def _bundle(
    *,
    roster: EffectiveRoster,
    revision: RuntimeRevision,
    manifest: ApplicationManifest | None = None,
    factory_resolver: InMemoryRuntimeAgentFactoryResolver | None,
    ctx: ApplicationBuildContext | None = None,
) -> RegistryProjectionInputBundle:
    manifest_obj = manifest or _manifest()
    build_ctx = ctx or ApplicationBuildContext.for_manifest(manifest_obj)
    return RegistryProjectionInputBundle(
        runtime_revision=revision,
        effective_roster=roster,
        manifest=manifest_obj,
        build_context=build_ctx,
        factory_resolver=factory_resolver,
        builders=None,
        materialization_artifact_digest=_ARTIFACT,
    )


def _project(bundle: RegistryProjectionInputBundle) -> MaterializedRegistryProjection:
    return build_registry_projection(bundle)


def test_happy_path_revision_bound_canonical_factory_projection() -> None:
    probe = _FactoryProbe(marker=_IMMUTABLE_MARKER)
    roster = _roster(
        (
            _entry(
                "search",
                package_digest=_DIGEST_A,
                factory_reference=_REF_IMMUTABLE,
                merged_config={"marker": "frozen-config"},
            ),
        )
    )
    revision = _revision(package_digests=(_DIGEST_A,))
    manifest = _manifest()
    ctx = ApplicationBuildContext.for_manifest(manifest)
    resolver = _resolver((_DIGEST_A, _REF_IMMUTABLE, _immutable_factory(probe)))

    projection = _project(
        _bundle(
            roster=roster,
            revision=revision,
            manifest=manifest,
            factory_resolver=resolver,
            ctx=ctx,
        )
    )

    assert projection.agent_registry.list_agent_ids() == ["search"]
    assert probe.calls == 1
    assert probe.seen_contexts[0] is ctx
    assert probe.seen_bindings[0].contract_id == "search"
    assert probe.seen_bindings[0].config.get("marker") == "frozen-config"
    assert projection.evidence.runtime_revision_id == revision.runtime_revision_id


def test_multi_agent_exact_factory_identity_per_entry() -> None:
    probe_a = _FactoryProbe(marker="FACTORY-A")
    probe_b = _FactoryProbe(marker="FACTORY-B")
    roster = _roster(
        (
            _entry(
                "search", package_digest=_DIGEST_A, factory_reference=_REF_IMMUTABLE
            ),
            _entry("indexer", package_digest=_DIGEST_B, factory_reference=_REF_OTHER),
        )
    )
    revision = _revision(package_digests=(_DIGEST_A, _DIGEST_B))
    resolver = _resolver(
        (_DIGEST_A, _REF_IMMUTABLE, _immutable_factory(probe_a)),
        (_DIGEST_B, _REF_OTHER, _immutable_factory(probe_b)),
    )

    projection = _project(
        _bundle(roster=roster, revision=revision, factory_resolver=resolver)
    )

    assert projection.agent_registry.list_agent_ids() == ["indexer", "search"]
    assert probe_a.calls == 1
    assert probe_b.calls == 1
    assert probe_a.seen_bindings[0].contract_id == "search"
    assert probe_b.seen_bindings[0].contract_id == "indexer"


def test_wrong_package_digest_fails_before_invocation() -> None:
    probe = _FactoryProbe(marker=_IMMUTABLE_MARKER)
    roster = _roster(
        (
            _entry(
                "search",
                package_digest=_DIGEST_B,
                factory_reference=_REF_IMMUTABLE,
            ),
        )
    )
    revision = _revision(package_digests=(_DIGEST_A,))
    resolver = _resolver((_DIGEST_B, _REF_IMMUTABLE, _immutable_factory(probe)))

    with pytest.raises(
        RegistryProjectionError, match="is not trusted by runtime revision"
    ):
        _project(_bundle(roster=roster, revision=revision, factory_resolver=resolver))
    assert probe.calls == 0


def test_wrong_factory_reference_fails_before_invocation() -> None:
    probe = _FactoryProbe(marker=_IMMUTABLE_MARKER)
    roster = _roster(
        (
            _entry(
                "search",
                package_digest=_DIGEST_A,
                factory_reference=_REF_OTHER,
            ),
        )
    )
    revision = _revision(package_digests=(_DIGEST_A,))
    resolver = _resolver((_DIGEST_A, _REF_IMMUTABLE, _immutable_factory(probe)))

    with pytest.raises(RegistryProjectionError, match="cannot resolve factory"):
        _project(_bundle(roster=roster, revision=revision, factory_resolver=resolver))
    assert probe.calls == 0


def test_internal_typeerror_invoked_once_no_fallback() -> None:
    calls = 0

    def _broken_factory(
        _ctx: ApplicationBuildContext,
        _binding: AgentBinding,
    ) -> EchoAgent:
        nonlocal calls
        calls += 1
        raise TypeError("internal factory failure")

    roster = _roster(
        (_entry("search", package_digest=_DIGEST_A, factory_reference=_REF_IMMUTABLE),)
    )
    revision = _revision(package_digests=(_DIGEST_A,))
    resolver = _resolver((_DIGEST_A, _REF_IMMUTABLE, _broken_factory))

    with pytest.raises(TypeError, match="internal factory failure"):
        _project(_bundle(roster=roster, revision=revision, factory_resolver=resolver))
    assert calls == 1


def test_invalid_factory_result_fails_before_registry_projection() -> None:
    def _bad_factory(
        _ctx: ApplicationBuildContext,
        _binding: AgentBinding,
    ) -> object:
        return object()

    roster = _roster(
        (_entry("search", package_digest=_DIGEST_A, factory_reference=_REF_IMMUTABLE),)
    )
    revision = _revision(package_digests=(_DIGEST_A,))
    resolver = _resolver((_DIGEST_A, _REF_IMMUTABLE, _bad_factory))

    with pytest.raises(AgentImportError, match="must return Agent"):
        _project(_bundle(roster=roster, revision=revision, factory_resolver=resolver))


def test_host_builders_present_cannot_override_resolver() -> None:
    probe = _FactoryProbe(marker=_IMMUTABLE_MARKER)
    conflicting = _FactoryProbe(marker="BUILDERS")
    roster = _roster(
        (_entry("search", package_digest=_DIGEST_A, factory_reference=_REF_IMMUTABLE),)
    )
    revision = _revision(package_digests=(_DIGEST_A,))
    resolver = _resolver((_DIGEST_A, _REF_IMMUTABLE, _immutable_factory(probe)))
    bundle = _bundle(roster=roster, revision=revision, factory_resolver=resolver)
    bundle = RegistryProjectionInputBundle(
        runtime_revision=bundle.runtime_revision,
        effective_roster=bundle.effective_roster,
        manifest=bundle.manifest,
        build_context=bundle.build_context,
        factory_resolver=bundle.factory_resolver,
        builders={EchoAgent: _immutable_factory(conflicting)},
        materialization_artifact_digest=bundle.materialization_artifact_digest,
    )

    projection = _project(bundle)
    assert projection.agent_registry.has("search")
    assert probe.calls == 1
    assert conflicting.calls == 0


def test_manifest_factory_present_cannot_override_resolver() -> None:
    probe = _FactoryProbe(marker=_IMMUTABLE_MARKER)
    roster = _roster(
        (_entry("search", package_digest=_DIGEST_A, factory_reference=_REF_IMMUTABLE),)
    )
    revision = _revision(package_digests=(_DIGEST_A,))
    resolver = _resolver((_DIGEST_A, _REF_IMMUTABLE, _immutable_factory(probe)))

    projection = _project(
        _bundle(roster=roster, revision=revision, factory_resolver=resolver)
    )

    assert projection.agent_registry.has("search")
    assert probe.calls == 1


def test_factory_path_on_manifest_cannot_bypass_resolver() -> None:
    probe = _FactoryProbe(marker=_IMMUTABLE_MARKER)
    roster = _roster(
        (
            _entry(
                "search",
                package_digest=_DIGEST_A,
                factory_reference=_REF_IMMUTABLE,
            ),
        )
    )
    revision = _revision(package_digests=(_DIGEST_A,))
    manifest = _manifest(
        agents=[
            AgentBinding.mount(EchoAgent, contract_id="search").model_copy(
                update={"factory": None, "factory_path": _LOCAL_FACTORY_PATH}
            ),
        ]
    )
    resolver = _resolver((_DIGEST_A, _REF_IMMUTABLE, _immutable_factory(probe)))

    projection = _project(
        _bundle(
            roster=roster,
            revision=revision,
            manifest=manifest,
            factory_resolver=resolver,
        )
    )

    assert projection.agent_registry.has("search")
    assert probe.calls == 1


def test_wrong_agent_type_from_factory_fails_closed() -> None:
    def _wrong_type_factory(
        _ctx: ApplicationBuildContext,
        _binding: AgentBinding,
    ) -> _OtherAgent:
        return _OtherAgent()

    roster = _roster(
        (_entry("search", package_digest=_DIGEST_A, factory_reference=_REF_IMMUTABLE),)
    )
    revision = _revision(package_digests=(_DIGEST_A,))
    resolver = _resolver((_DIGEST_A, _REF_IMMUTABLE, _wrong_type_factory))

    with pytest.raises(AgentImportError, match="expected instance of EchoAgent"):
        _project(_bundle(roster=roster, revision=revision, factory_resolver=resolver))


def test_disabled_roster_entry_not_materialized() -> None:
    probe = _FactoryProbe(marker=_IMMUTABLE_MARKER)
    roster = _roster(
        (
            _entry(
                "search", package_digest=_DIGEST_A, factory_reference=_REF_IMMUTABLE
            ),
            _entry(
                "indexer",
                package_digest=_DIGEST_B,
                factory_reference=_REF_OTHER,
                effective_enablement=False,
            ),
        )
    )
    revision = _revision(package_digests=(_DIGEST_A, _DIGEST_B))
    resolver = _resolver(
        (_DIGEST_A, _REF_IMMUTABLE, _immutable_factory(probe)),
        (_DIGEST_B, _REF_OTHER, _immutable_factory(_FactoryProbe(marker="SKIP"))),
    )

    projection = _project(
        _bundle(roster=roster, revision=revision, factory_resolver=resolver)
    )

    assert projection.agent_registry.list_agent_ids() == ["search"]
    assert probe.calls == 1


def test_resolver_catalog_knowledge_not_revision_authority() -> None:
    probe = _FactoryProbe(marker=_IMMUTABLE_MARKER)
    resolver = _resolver((_DIGEST_A, _REF_IMMUTABLE, _immutable_factory(probe)))
    revision = _revision(revision_id="rev-unauthorized", package_digests=())

    with pytest.raises(
        RuntimeAgentFactoryResolutionError, match="is not part of runtime revision"
    ):
        resolver.resolve_factory(
            runtime_revision=revision,
            package_digest=_DIGEST_A,
            factory_reference=_REF_IMMUTABLE,
        )
    assert probe.calls == 0


def test_strict_invocation_contract_direct() -> None:
    probe = _FactoryProbe(marker=_IMMUTABLE_MARKER)
    manifest = _manifest()
    binding = manifest.agents[0]
    ctx = ApplicationBuildContext.for_manifest(manifest)
    agent = invoke_canonical_agent_factory(_immutable_factory(probe), ctx, binding)
    assert isinstance(agent, EchoAgent)
    assert probe.calls == 1
    assert probe.seen_contexts[0] is ctx
    assert probe.seen_bindings[0] is binding


def test_venv_bundle_production_projection_e2e(tmp_path: Path) -> None:
    from intergrax.agent_distribution.dependency import (
        MaterializedAgentClosureEntry,
        MaterializedLockPackage,
        MaterializedRuntimeLock,
    )

    artifact_root = tmp_path / "artifact"
    site_packages = artifact_root / "site-packages"
    package_dir = site_packages / "example_agent"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "factory.py").write_text(
        textwrap.dedent(
            """
            from echo.echo_agent import EchoAgent

            MARKER = "VENV_IMMUTABLE"

            def build_agent(ctx, binding):
                return EchoAgent()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    lock = MaterializedRuntimeLock(
        resolver_algorithm_id="intergrax.test",
        resolver_algorithm_version="1",
        inputs_digest="inputs-1",
        intergrax_version="0.1.0",
        python_version="3.12",
        packages=(
            MaterializedLockPackage(
                distribution_name="pkg-search",
                version="1.0.0",
                package_digest=_DIGEST_A,
            ),
        ),
        agent_closure=(
            MaterializedAgentClosureEntry(
                distribution_package_id="pkg-search",
                package_digest=_DIGEST_A,
                role="direct",
            ),
        ),
    ).with_content_identity()
    (artifact_root / RUNTIME_LOCK_MANIFEST_FILENAME).write_text(
        json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    digest = directory_content_digest(artifact_root)
    roster = _roster(
        (
            _entry(
                "search",
                package_digest=_DIGEST_A,
                factory_reference=AgentBindingFactoryReference(
                    factory_path="example_agent.factory.build_agent"
                ),
            ),
        )
    )
    revision = _revision(package_digests=(_DIGEST_A,)).model_copy(
        update={
            "materialization_artifact_digest": digest,
            "materialized_runtime_lock_id": lock.lock_id,
            "materialized_runtime_lock_digest": lock.lock_digest,
        }
    )
    manifest = _manifest(agents=[AgentBinding.mount(EchoAgent, contract_id="search")])
    resolver = build_production_runtime_agent_factory_resolver(
        runtime_revision=revision,
        artifact_root=artifact_root,
        expected_artifact_digest=digest,
    )
    projection = _project(
        RegistryProjectionInputBundle(
            runtime_revision=revision,
            effective_roster=roster,
            manifest=manifest,
            build_context=ApplicationBuildContext.for_manifest(manifest),
            factory_resolver=resolver,
            builders=None,
            materialization_artifact_digest=digest,
        )
    )
    assert projection.agent_registry.list_agent_ids() == ["search"]
    assert projection.evidence.materialization_artifact_digest == digest


def test_venv_factory_cache_key_is_revision_bound_not_authority() -> None:
    """Cache keys include artifact digest + package digest + factory reference."""
    from intergrax.applications._shared import (
        venv_bundle_runtime_agent_factory_resolver as module,
    )

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "_FactoryCache" in source
    assert "cache_key = (" in source
    assert "artifact_digest" in source
    assert "digest" in source
    assert "factory_reference.factory_path" in source
