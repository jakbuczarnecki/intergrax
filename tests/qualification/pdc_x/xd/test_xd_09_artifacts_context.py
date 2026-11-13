# © Artur Czarnecki. All rights reserved.

"""XD-09 (Artifacts <-> Context/Tools): spill to a ref, the ref survives compaction, evidence is undeletable.

PDC-D2-R018 and PDC-D8-R019 (evidence artifacts survive compaction), PDC-D8-R012 (spill round trip)
and PDC-D8-R013 (the context reference resolves after compaction).

Integrated version of ``d8/test_xd09_spill_and_context_reference.py``: the reference travels through
the real ``DefaultNexusContextEngine`` assembly under budget pressure (not only the fragment
compaction helper), the store is the durable SQLite artifact store, the read path is the governed
port over the real governance pipeline, and TTL deletion runs as real ``WorkerRuntime`` background
work. Fakes: the LLM adapter (external provider) only.

The canonical composition-root checks at the end are strict xfails that pin the open D8 wiring
findings (see the FINDING reasons).
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import json
from dataclasses import fields
from datetime import timedelta
from pathlib import Path

import pytest
from intergrax.background_tasks.definition import TaskDefinition
from intergrax.background_tasks.registry import TaskRegistry
from intergrax.background_tasks.state_store import TaskResultStore, TaskStateStore
from intergrax.background_tasks.worker_runtime import WorkerRuntime
from intergrax.context.budget.compaction import DeterministicTailCompactionStrategy
from intergrax.context.budget.compaction_apply import apply_fragment_compaction
from intergrax.context.contracts import ContextAuthorityClass, ContextFragmentSource
from intergrax.context.provider_descriptor import build_provider_descriptor
from intergrax.context.registry import ContextPluginRegistry
from intergrax.contracts.artifact_ref import (
    ArtifactClass,
    ArtifactRetentionClass,
    compute_artifact_digest,
)
from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.admission_wiring import (
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.nexus.artifacts.context_reference import (
    ArtifactReferenceResolver,
    ArtifactReferenceToken,
    reference_fragment,
)
from intergrax.runtime.nexus.artifacts.governed_access import ArtifactNotFound
from intergrax.runtime.nexus.artifacts.retention import (
    ARTIFACT_RETENTION_SWEEP_TASK_NAME,
    ArtifactRetentionSchedule,
    ArtifactRetentionSweeper,
    make_retention_sweep_handler,
)
from intergrax.runtime.nexus.artifacts.spill import ToolOutputSpiller
from intergrax.runtime.nexus.artifacts.store_base import ArtifactDeletionRefused, ArtifactStore
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.persistence.sqlite_composition import create_sqlite_runtime_persistence
from tests.qualification.pdc_x.d2._support import (
    FakeAdapter,
    FragmentProvider,
    assembly_request,
    make_fragment,
    message_text,
    provider_context,
)
from tests.qualification.pdc_x.d8._d8_support import (
    T0,
    TENANT_A,
    MutableClock,
    make_artifact,
    make_context,
    make_governance,
    make_port,
    reader_grants,
    sqlite_store,
)
from tests.unit.runtime.background_execution.reentry_admission_doubles import admission_kwargs

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[4]
BIG = ("row,value,notes\n" + "alpha,1,some tabular tool output\n" * 4000).encode()
EVIDENCE = b"run evidence: decision record 42 (audit)"
THRESHOLD = 4096


def _spiller(store: ArtifactStore, clock: MutableClock) -> ToolOutputSpiller:
    return ToolOutputSpiller(
        store, threshold_bytes=THRESHOLD, schedule=ArtifactRetentionSchedule.default(), clock=clock
    )


def _spill(spiller: ToolOutputSpiller) -> ArtifactReferenceToken:
    result = spiller.spill_if_large(
        tenant_id=TENANT_A,
        run_id="run-1",
        task_id="task-1",
        tool_id="sql.query",
        agent_id="agent-1",
        content=BIG,
    )
    assert result.inline is None and result.reference is not None
    return result.reference


def _evidence_token(store: ArtifactStore) -> ArtifactReferenceToken:
    stored = store.put(
        make_artifact(
            artifact_id="evidence-1",
            data=EVIDENCE,
            retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE,
            expires_at_utc=T0 + timedelta(minutes=1),  # even an explicit expiry must not delete evidence
        )
    )
    return ArtifactReferenceToken(
        tenant_id=TENANT_A,
        artifact_class=ArtifactClass.EXECUTION_EVIDENCE,
        artifact_id=stored.artifact_id,
        version=stored.version,
        content_digest=stored.content_digest,
    )


class _ReferenceProvider(FragmentProvider):
    """A context provider declared for tool observations, as the artifact-reference provider would be."""

    @property
    def descriptor(self):  # type: ignore[no-untyped-def]
        return build_provider_descriptor(
            self.provider_id,
            provider_version="1.0.0",
            supported_sources=self._sources,
            origin="test",
            trusted_authority_class=ContextAuthorityClass.TOOL_OBSERVATION,
            allowed_authority_classes=frozenset({ContextAuthorityClass.TOOL_OBSERVATION}),
        )


def _ref_fragment(token: ArtifactReferenceToken, index: int, *, linked: bool):  # type: ignore[no-untyped-def]
    fragment = reference_fragment(token, fragment_id=f"ref-{index}")
    if not linked:
        return fragment  # exactly what the product builds
    # The shape a repair would take: the reference observation is linked to its tool call.
    metadata = {**fragment.metadata, "tool_call_id": f"call-{index}", "tool_name": "sql.query"}
    return dataclasses.replace(fragment, metadata=metadata)


async def _assemble_under_pressure(*tokens: ArtifactReferenceToken, linked: bool, budget: int = 300) -> str:
    """Run the real context engine with the reference fragments plus bulky optional context."""
    registry = ContextPluginRegistry()
    registry.add_provider(
        _ReferenceProvider(
            "xd09.refs",
            {ContextFragmentSource.TOOL_OUTPUT},
            lambda: [_ref_fragment(token, i, linked=linked) for i, token in enumerate(tokens)],
        )
    )
    registry.add_provider(
        FragmentProvider(
            "xd09.noise",
            {ContextFragmentSource.RAG},
            lambda: [
                make_fragment(
                    f"noise-{i}",
                    ContextFragmentSource.RAG,
                    "RAG-FILLER " + ("lorem ipsum dolor " * 400),
                    relevance=0.2 + 0.1 * i,
                    token_estimate=1500,
                )
                for i in range(3)
            ],
        )
    )
    assembled = await DefaultNexusContextEngine(registry=registry).assemble(
        assembly_request(budget_tokens=budget),
        provider_ctx=provider_context(FakeAdapter(window=4096), "what did the query return?"),
    )
    assert assembled.total_tokens <= assembled.budget_tokens
    return message_text(assembled.messages)


# ----------------------------------------------------------------------- R012 / R013 through the engine


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING-4 (D2 x D8): the artifact reference fragment never reaches the model-facing context through the "
        "real engine. reference_fragment() (intergrax/runtime/nexus/artifacts/context_reference.py) builds a "
        "TOOL_OUTPUT fragment without a tool_call_id; canonicalize_fragment() "
        "(intergrax/context/provider_lifecycle.py, D2 R009 hardening) rejects it with provider.contract_violation "
        "and the whole providing source fails, so the mandatory reference is lost before the model call. "
        "Smallest repair: give reference_fragment a tool_call_id/tool_name of the originating call (the linked shape "
        "is proven to survive compaction in the next test) and compose an authority-declared reference provider."
    ),
)
async def test_xd09_reference_as_shipped_reaches_the_engine_output(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    token = _spill(_spiller(store, MutableClock()))
    text = await _assemble_under_pressure(token, linked=False)
    assert token.to_marker() in text


async def test_xd09_spilled_output_reference_survives_engine_compaction_and_resolves(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    clock = MutableClock()
    token = _spill(_spiller(store, clock))
    assert token.content_digest == compute_artifact_digest(BIG)

    text = await _assemble_under_pressure(token, linked=True)
    assert token.to_marker() in text, "the reference is carried into the model-facing context"
    assert "RAG-FILLER" not in text, "optional bulk was compacted away under the budget"
    assert "alpha,1,some tabular tool output" not in text, "the spilled output itself never enters the context"

    parsed = ArtifactReferenceToken.find_in_text(text)
    assert parsed == token
    del store  # restart: the durable store still resolves the reference
    port, _ = make_port(sqlite_store(tmp_path), clock=clock)
    assert ArtifactReferenceResolver(port).resolve(make_context(), parsed) == BIG


async def test_xd09_reference_fragment_survives_compaction_helper_untouched(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    token = _spill(_spiller(store, MutableClock()))
    ref = reference_fragment(token, fragment_id="ref")
    noisy = [
        make_fragment(f"n-{i}", ContextFragmentSource.RAG, "lorem " * 800, relevance=0.1 * (i + 1), token_estimate=2000)
        for i in range(3)
    ]
    compacted, provenance = apply_fragment_compaction(
        [ref, *noisy],
        strategy=DeterministicTailCompactionStrategy(),
        count_text=lambda text: max(1, len(text) // 4),
        fragment_budget_tokens=200,
    )
    assert provenance
    survivor = next(f for f in compacted if f.fragment_id == "ref")
    assert survivor.content == ref.content
    assert store.get(TENANT_A, token.artifact_id).data == BIG  # compaction never touches the store


# ----------------------------------------------------------------------- D2-R018 / D8-R019


async def test_xd09_evidence_artifact_survives_compaction_and_cannot_be_deleted(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    clock = MutableClock()
    evidence = _evidence_token(store)
    spill = _spill(_spiller(store, clock))
    harness = make_governance(granted=reader_grants())
    port, _ = make_port(store, harness=harness, clock=clock)
    digest_before = compute_artifact_digest(EVIDENCE)

    # Compaction pressure: the evidence reference is mandatory context and is kept verbatim.
    text = await _assemble_under_pressure(evidence, spill, linked=True)
    assert evidence.to_marker() in text and spill.to_marker() in text
    assert "RAG-FILLER" not in text

    # The evidence artifact itself is untouched and still verifies through the governed port.
    assert store.get(TENANT_A, "evidence-1").content_digest == digest_before
    assert ArtifactReferenceResolver(port).resolve(make_context(), evidence) == EVIDENCE

    # No delete path removes it: direct delete, run delete and the TTL sweep all refuse.
    with pytest.raises(ArtifactDeletionRefused):
        store.delete_artifact(TENANT_A, "evidence-1")
    with pytest.raises(ArtifactDeletionRefused):
        store.delete_for_run(TENANT_A, "run-1")
    clock.advance(hours=2)  # past both the evidence expiry and the ephemeral spill TTL
    sweep_ctx = make_context(agent_id="artifact_retention_sweep")
    report = ArtifactRetentionSweeper(store, clock=clock, audit=harness.recorder).sweep(TENANT_A, context=sweep_ctx)
    assert "evidence-1" in report.protected_ids and "evidence-1" not in report.deleted_ids
    assert spill.artifact_id in report.deleted_ids  # ephemeral output is the only thing the sweep removes
    assert store.get(TENANT_A, "evidence-1").data == EVIDENCE

    # Restart: evidence still there and still resolvable; the expired spill is a clean not-found.
    reopened = sqlite_store(tmp_path)
    port2, _ = make_port(reopened, clock=clock)
    assert ArtifactReferenceResolver(port2).resolve(make_context(), evidence) == EVIDENCE
    with pytest.raises(ArtifactNotFound):
        ArtifactReferenceResolver(port2).resolve(make_context(), spill)


async def test_xd09_session_compaction_keeps_source_evidence_and_leaves_evidence_artifacts_alone(
    tmp_path: Path,
) -> None:
    """Durable (UCL) compaction is additive: sources stay reconstructable, evidence is unaffected."""
    from intergrax.runtime.context_lifecycle import SQLiteOptimizationArtifactRepository
    from intergrax.runtime.token_optimization.durable_compaction_validation import (
        DurableCompactionValidationCompiler,
        DurableCompactionValidationStatus,
    )
    from tests.unit.runtime.token_optimization import test_durable_compaction_closeout as fx

    store = sqlite_store(tmp_path)
    evidence = _evidence_token(store)
    repository = SQLiteOptimizationArtifactRepository(
        str(tmp_path / "ucl.sqlite"),
        clock=lambda: fx._NOW,
        reservation_id_factory=lambda: "reservation-xd09",
    )
    try:
        request = fx._request()
        built = fx._builder(repository, [0], []).build(request)
        assert built.candidate is not None
        outcome = DurableCompactionValidationCompiler(
            repository=repository,
            clock=lambda: fx._NOW,
            receipt_id_factory=lambda: "durable-receipt-xd09",
        ).compile(fx._validation_request(request, built))
        assert outcome.status is DurableCompactionValidationStatus.PASSED

        artifact = repository.resolve(outcome.candidate.artifact_reference)
        assert artifact is not None
        payload = json.loads(artifact.payload.decode("utf-8"))
        # every original message stays reconstructable by id from the compacted artifact
        assert tuple(payload["source_refs"]) == tuple(m.message_id for m in fx._messages())
        assert payload["source_content_hash"] == request.snapshot.source_identity.source_content_hash
    finally:
        repository.close()
    # ... and the evidence plane (artifact store) did not change as a side effect of compaction
    assert store.get(TENANT_A, evidence.artifact_id).content_digest == evidence.content_digest
    assert [a.artifact_id for a in store.list_for_run(TENANT_A, "run-1")] == ["evidence-1"]


# ----------------------------------------------------------------------- TTL as background work


async def test_xd09_expired_spill_is_removed_by_the_sweep_running_as_real_worker_runtime_work(
    tmp_path: Path,
) -> None:
    store = sqlite_store(tmp_path)
    clock = MutableClock()
    spill = _spill(_spiller(store, clock))
    evidence = _evidence_token(store)
    harness = make_governance(granted=reader_grants())

    sweeper = ArtifactRetentionSweeper(store, clock=clock, audit=harness.recorder)
    kv = SqliteDistributedKVStore(tmp_path / "worker_kv.sqlite3")
    registry = TaskRegistry()
    registry.register(
        TaskDefinition(
            task_name=ARTIFACT_RETENTION_SWEEP_TASK_NAME,
            payload_schema=dict,
            handler=make_retention_sweep_handler(sweeper),
        )
    )
    execution_registry = TaskExecutionRegistry()
    registry.bind_execution_registry(execution_registry)
    worker = WorkerRuntime(
        registry=registry,
        state_store=TaskStateStore(kv_store=kv, provider="sqlite"),
        result_store=TaskResultStore(kv_store=kv),
        execution_registry=execution_registry,
        provider="sqlite",
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
        **admission_kwargs(wire_background_execution_admission_dependencies(kv_store=kv)),  # pyright: ignore[reportArgumentType]
    )

    def run(run_id: str):  # type: ignore[no-untyped-def]
        request = TaskRequest(
            tenant_id=TENANT_A,
            run_id=run_id,
            task_name=ARTIFACT_RETENTION_SWEEP_TASK_NAME,
            payload=json.dumps({"limit": 100}).encode("utf-8"),
        )
        return worker.process_request(request, task_id=run_id)

    # Before expiry: nothing is deleted.
    assert run("sweep-1").status is TaskStatus.SUCCEEDED
    assert store.get(TENANT_A, spill.artifact_id).data == BIG

    clock.advance(hours=2)
    assert run("sweep-2").status is TaskStatus.SUCCEEDED
    with pytest.raises(KeyError):
        store.get(TENANT_A, spill.artifact_id)
    assert store.get(TENANT_A, evidence.artifact_id).data == EVIDENCE
    deletes = [e for e in harness.sink.events if e.capability == "artifact.delete.outcome"]
    assert any(e.agent_id == "artifact_retention_sweep" for e in deletes)


# ----------------------------------------------------------------------- canonical composition root findings


@functools.cache
def _production_sources() -> tuple[Path, ...]:
    files: list[Path] = []
    for root in ("intergrax", "agents", "applications"):
        files.extend(
            path
            for path in (REPO_ROOT / root).rglob("*.py")
            if "nexus/artifacts" not in path.as_posix() and "__pycache__" not in path.as_posix()
        )
    return tuple(files)


_PROBED = frozenset(
    {
        "ToolOutputSpiller",
        "GovernedArtifactReadPort",
        "ArtifactRetentionSweeper",
        "register_retention_sweep",
        "make_retention_sweep_handler",
    }
)


@functools.cache
def _all_references() -> dict[str, tuple[str, ...]]:
    """One pass over production sources outside the artifacts package: probed name -> referencing files."""
    found: dict[str, list[str]] = {name: [] for name in _PROBED}
    for path in _production_sources():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not any(name in text for name in _PROBED):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            hit: str | None = None
            if isinstance(node, ast.ImportFrom):
                hit = next((a.name for a in node.names if a.name in _PROBED), None)
            elif isinstance(node, ast.Name) and node.id in _PROBED:
                hit = node.id
            elif isinstance(node, ast.Attribute) and node.attr in _PROBED:
                hit = node.attr
            if hit is not None:
                found[hit].append(path.relative_to(REPO_ROOT).as_posix())
    return {name: tuple(sites) for name, sites in found.items()}


def _referenced_names(names: set[str]) -> dict[str, tuple[str, ...]]:
    references = _all_references()
    return {name: references[name] for name in names}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING-1 (D8): ToolOutputSpiller is not wired into the tool runtime. Nothing outside "
        "intergrax/runtime/nexus/artifacts references it, so tool results are never spilled in a real run. "
        "Smallest repair: call ToolOutputSpiller.spill_if_large from the tool-result path in "
        "intergrax/runtime/nexus/tools/ (tool_runtime / plan_context_invocation) and inject the store at the root."
    ),
)
def test_xd09_canonical_tool_runtime_spills_large_output() -> None:
    used = _referenced_names({"ToolOutputSpiller"})
    assert used["ToolOutputSpiller"], "no production module outside the artifacts package uses ToolOutputSpiller"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING-2 (D8): neither GovernedArtifactReadPort nor the retention sweep "
        "(register_retention_sweep / make_retention_sweep_handler / ArtifactRetentionSweeper) is registered by any "
        "composition root; TTL deletion and governed reads exist only in tests. Smallest repair: construct the port "
        "and register the sweep handler on the TaskExecutionRegistry in the platform worker bootstrap "
        "(intergrax/runtime/task/worker_bootstrap.py) behind the runtime config."
    ),
)
def test_xd09_governed_port_and_retention_sweep_are_composed_by_a_root() -> None:
    used = _referenced_names(
        {
            "GovernedArtifactReadPort",
            "ArtifactRetentionSweeper",
            "register_retention_sweep",
            "make_retention_sweep_handler",
        }
    )
    missing = sorted(name for name, sites in used.items() if not sites)
    assert missing == [], f"not composed by any production module: {missing}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING-3 (D8): the canonical SQLite persistence root has no artifact store. "
        "SQLiteRuntimePersistenceBundle (intergrax/runtime/persistence/sqlite_composition.py) exposes no "
        "ArtifactStore and sqlite_opens.py has no open_artifact_store_at, so SQLiteArtifactStore is only built "
        "directly. Smallest repair: add open_artifact_store_at to sqlite_opens.py, an artifact path to "
        "SqliteStorePaths and an artifact_store field to the bundle."
    ),
)
def test_xd09_sqlite_persistence_root_composes_the_artifact_store(tmp_path: Path) -> None:
    bundle = create_sqlite_runtime_persistence(data_dir=tmp_path / "sqlite")
    stores = [f.name for f in fields(bundle) if isinstance(getattr(bundle, f.name), ArtifactStore)]
    assert stores, "no ArtifactStore on the canonical SQLite persistence bundle"
