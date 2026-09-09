"""Durable resume / checkpoint tests for provider-neutral storage bootstrap."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.batching import (
    compute_bootstrap_plan,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.codec import (
    decode_checkpoint,
    encode_checkpoint,
    run_identity_digest,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.compatibility import (
    build_run_identity,
    compute_resume_decision,
    initial_checkpoint_state,
    utc_now_iso,
    validate_checkpoint_compatibility,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.contracts import (
    VPI_STORAGE_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
    BootstrapCheckpointState,
    BootstrapRunIdentity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.errors import (
    CheckpointAlreadyExists,
    CheckpointConcurrentModification,
    CheckpointCorrupt,
    CheckpointPersistenceError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.filesystem_store import (
    FilesystemBootstrapCheckpointStore,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapBatchPhase,
    BootstrapBatchSize,
    BootstrapFinalStatus,
    BootstrapProgress,
    BootstrapRequest,
    RelationalTargetId,
    ResumeMode,
    VectorTargetId,
    VerificationMode,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.service import (
    StorageBootstrapDependencies,
    StorageBootstrapService,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    BootstrapFailureCategory,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_storage_bootstrap_data_pack_load import (
    FakeDataPackReader,
    FakeRelationalAdapter,
    FakeVectorAdapter,
    InMemoryBootstrapCheckpointStore,
    _build_pairs,
    _ready_manifest,
    _request,
    _service,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_CHECKPOINT_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/storage_bootstrap/data_pack_load/checkpoint"
)
_FORBIDDEN_PROVIDER_IMPORTS = frozenset(
    {"psycopg", "qdrant_client", "pgvector", "torch", "transformers", "sentence_transformers"}
)


@dataclass
class FailingCheckpointStore(InMemoryBootstrapCheckpointStore):
    fail_commits_remaining: int = 0

    def commit_batch(
        self,
        *,
        run_identity: object,
        expected_revision: int,
        checkpoint: object,
    ) -> object:
        if self.fail_commits_remaining > 0:
            self.fail_commits_remaining -= 1
            raise CheckpointPersistenceError("injected checkpoint persistence failure")
        return super().commit_batch(
            run_identity=run_identity,
            expected_revision=expected_revision,
            checkpoint=checkpoint,
        )


@dataclass
class CountingRelationalAdapter(FakeRelationalAdapter):
    write_calls: int = 0

    def write_batch(self, batch: object) -> object:
        self.write_calls += 1
        return super().write_batch(batch)  # type: ignore[arg-type]


@dataclass
class CountingVectorAdapter(FakeVectorAdapter):
    write_calls: int = 0

    def write_batch(self, batch: object) -> object:
        self.write_calls += 1
        return super().write_batch(batch)  # type: ignore[arg-type]


def _run_identity_for(count: int, *, batch_size: int = 2) -> BootstrapRunIdentity:
    manifest = _ready_manifest(count)
    request = _request(batch_size=batch_size)
    return build_run_identity(manifest=manifest, request=request)


def _plan_for(count: int, *, batch_size: int = 2) -> object:
    return compute_bootstrap_plan(
        record_count=count,
        batch_size=batch_size,
        relational_target=RelationalTargetId("vpi-products"),
        vector_target=VectorTargetId("vpi-product-embeddings"),
    )


def _filesystem_store(tmp_path: Path) -> FilesystemBootstrapCheckpointStore:
    return FilesystemBootstrapCheckpointStore(checkpoint_root=tmp_path / "bootstrap-state")


def _fresh_then_resume(
    count: int,
    *,
    batch_size: int = 2,
    checkpoint_store: InMemoryBootstrapCheckpointStore | FilesystemBootstrapCheckpointStore,
    relational: FakeRelationalAdapter | None = None,
    vector: FakeVectorAdapter | None = None,
    stop_after_batches: int | None = None,
) -> tuple[object, object]:
    relational = relational or FakeRelationalAdapter()
    vector = vector or FakeVectorAdapter()
    if stop_after_batches is not None:
        vector.fail_on_batch = stop_after_batches
    first = _service(
        count,
        relational=relational,
        vector=vector,
        checkpoint_store=checkpoint_store,  # type: ignore[arg-type]
    ).run(_request(batch_size=batch_size))
    vector.fail_on_batch = None
    second = _service(
        count,
        relational=relational,
        vector=vector,
        checkpoint_store=checkpoint_store,  # type: ignore[arg-type]
    ).run(_request(batch_size=batch_size, resume_mode=ResumeMode.RESUME))
    return first, second


# --- FRESH ---


def test_fresh_initializes_checkpoint_state(tmp_path: Path) -> None:
    store = _filesystem_store(tmp_path)
    result = _service(4, checkpoint_store=store).run(_request(batch_size=2))  # type: ignore[arg-type]
    identity = _run_identity_for(4, batch_size=2)
    loaded = store.load(identity)
    assert result.status is BootstrapFinalStatus.SUCCESS
    assert loaded is not None
    assert loaded.committed_record_count == 4
    assert loaded.last_committed_batch_number == 1


def test_fresh_batch0_commits_checkpoint(tmp_path: Path) -> None:
    store = InMemoryBootstrapCheckpointStore()
    _service(2, checkpoint_store=store).run(_request(batch_size=2))
    loaded = store.load(_run_identity_for(2, batch_size=2))
    assert isinstance(loaded, BootstrapCheckpointState)
    assert loaded.last_committed_batch_number == 0
    assert loaded.committed_record_count == 2


def test_fresh_all_batches_complete(tmp_path: Path) -> None:
    result = _service(5, checkpoint_store=InMemoryBootstrapCheckpointStore()).run(_request(batch_size=2))
    assert result.status is BootstrapFinalStatus.SUCCESS
    assert result.committed_batches == 3


def test_fresh_rejects_existing_checkpoint() -> None:
    store = InMemoryBootstrapCheckpointStore()
    _service(2, checkpoint_store=store).run(_request(batch_size=2))
    retry = _service(2, checkpoint_store=store).run(_request(batch_size=2))
    assert retry.failure is not None
    assert retry.failure.category is BootstrapFailureCategory.CHECKPOINT_FAILED


# --- RESUME ---


def test_resume_skips_committed_batch0() -> None:
    store = InMemoryBootstrapCheckpointStore()
    relational = CountingRelationalAdapter()
    vector = CountingVectorAdapter(fail_on_batch=1)
    _service(4, relational=relational, vector=vector, checkpoint_store=store).run(_request(batch_size=2))
    relational.write_calls = 0
    vector.write_calls = 0
    vector.fail_on_batch = None
    result = _service(
        4,
        relational=relational,
        vector=vector,
        checkpoint_store=store,
    ).run(_request(batch_size=2, resume_mode=ResumeMode.RESUME))
    assert result.resumed_from_batch == 1
    assert result.previously_committed_batches == 1
    assert relational.write_calls == 1
    assert vector.write_calls == 1


def test_resume_continues_at_batch1() -> None:
    store = InMemoryBootstrapCheckpointStore()
    first, second = _fresh_then_resume(4, batch_size=2, checkpoint_store=store, stop_after_batches=0)
    assert first.committed_batches == 0
    assert second.committed_batches == 2
    assert second.status is BootstrapFinalStatus.SUCCESS


def test_resume_does_not_rewrite_committed_providers() -> None:
    store = InMemoryBootstrapCheckpointStore()
    relational = CountingRelationalAdapter()
    vector = CountingVectorAdapter()
    _service(4, relational=relational, vector=vector, checkpoint_store=store).run(_request(batch_size=2))
    first_calls = (relational.write_calls, vector.write_calls)
    _service(
        4,
        relational=relational,
        vector=vector,
        checkpoint_store=store,
    ).run(_request(batch_size=2, resume_mode=ResumeMode.RESUME))
    assert relational.write_calls == first_calls[0]
    assert vector.write_calls == first_calls[1]


# --- RELATIONAL FAILURE ---


def test_relational_failure_leaves_checkpoint_unchanged() -> None:
    store = InMemoryBootstrapCheckpointStore()
    result = _service(
        4,
        relational=FakeRelationalAdapter(fail_on_batch=0),
        checkpoint_store=store,
    ).run(_request(batch_size=2))
    loaded = store.load(_run_identity_for(4, batch_size=2))
    assert isinstance(loaded, BootstrapCheckpointState)
    assert loaded.committed_record_count == 0
    assert result.committed_batches == 0


# --- VECTOR FAILURE ---


def test_vector_failure_leaves_checkpoint_unchanged() -> None:
    store = InMemoryBootstrapCheckpointStore()
    _service(
        4,
        vector=FakeVectorAdapter(fail_on_batch=1),
        checkpoint_store=store,
    ).run(_request(batch_size=2))
    loaded = store.load(_run_identity_for(4, batch_size=2))
    assert isinstance(loaded, BootstrapCheckpointState)
    assert loaded.last_committed_batch_number == 0


def test_vector_failure_resume_retries_same_batch() -> None:
    store = InMemoryBootstrapCheckpointStore()
    relational = CountingRelationalAdapter()
    first, second = _fresh_then_resume(
        4,
        batch_size=2,
        checkpoint_store=store,
        relational=relational,
        stop_after_batches=1,
    )
    assert first.committed_batches == 1
    assert second.committed_batches == 1
    assert second.status is BootstrapFinalStatus.SUCCESS


def test_vector_failure_resume_relational_skipped() -> None:
    store = InMemoryBootstrapCheckpointStore()
    relational = FakeRelationalAdapter()
    _service(4, relational=relational, vector=FakeVectorAdapter(fail_on_batch=1), checkpoint_store=store).run(
        _request(batch_size=2)
    )
    before = len(relational.storage)
    _service(
        4,
        relational=relational,
        vector=FakeVectorAdapter(),
        checkpoint_store=store,
    ).run(_request(batch_size=2, resume_mode=ResumeMode.RESUME))
    assert len(relational.storage) == before


# --- VERIFICATION FAILURE ---


def test_verification_failure_leaves_checkpoint_unchanged() -> None:
    store = InMemoryBootstrapCheckpointStore()
    _service(
        2,
        vector=FakeVectorAdapter(verify_mismatch_on_batch=0),
        checkpoint_store=store,
    ).run(_request(batch_size=2))
    loaded = store.load(_run_identity_for(2, batch_size=2))
    assert isinstance(loaded, BootstrapCheckpointState)
    assert loaded.committed_record_count == 0


def test_verification_failure_resume_retries_batch() -> None:
    store = InMemoryBootstrapCheckpointStore()
    vector = FakeVectorAdapter(verify_mismatch_on_batch=0)
    first = _service(2, vector=vector, checkpoint_store=store).run(_request(batch_size=2))
    vector.verify_mismatch_on_batch = None
    second = _service(2, vector=vector, checkpoint_store=store).run(
        _request(batch_size=2, resume_mode=ResumeMode.RESUME)
    )
    assert first.committed_batches == 0
    assert second.committed_batches == 1


# --- CHECKPOINT FAILURE ---


def test_checkpoint_failure_does_not_mark_batch_committed() -> None:
    store = FailingCheckpointStore(fail_commits_remaining=1)
    result = _service(2, checkpoint_store=store).run(_request(batch_size=2))
    assert result.failure is not None
    assert result.failure.category is BootstrapFailureCategory.CHECKPOINT_FAILED
    assert result.committed_batches == 0


def test_checkpoint_failure_resume_retries_batch() -> None:
    store = FailingCheckpointStore(fail_commits_remaining=1)
    first = _service(2, checkpoint_store=store).run(_request(batch_size=2))
    second = _service(2, checkpoint_store=store).run(_request(batch_size=2, resume_mode=ResumeMode.RESUME))
    assert first.committed_batches == 0
    assert second.committed_batches == 1


def test_checkpoint_failure_resume_adapters_skipped() -> None:
    store = FailingCheckpointStore(fail_commits_remaining=1)
    relational = CountingRelationalAdapter()
    _service(2, relational=relational, checkpoint_store=store).run(_request(batch_size=2))
    calls_after_failed_commit = relational.write_calls
    _service(2, relational=relational, checkpoint_store=store).run(
        _request(batch_size=2, resume_mode=ResumeMode.RESUME)
    )
    assert relational.write_calls == calls_after_failed_commit + 1


def test_checkpoint_failure_second_commit_succeeds() -> None:
    store = FailingCheckpointStore(fail_commits_remaining=1)
    _service(2, checkpoint_store=store).run(_request(batch_size=2))
    result = _service(2, checkpoint_store=store).run(_request(batch_size=2, resume_mode=ResumeMode.RESUME))
    loaded = store.load(_run_identity_for(2, batch_size=2))
    assert isinstance(loaded, BootstrapCheckpointState)
    assert loaded.committed_record_count == 2
    assert result.status is BootstrapFinalStatus.SUCCESS


# --- COMPATIBILITY ---


@pytest.mark.parametrize(
    ("mutator_name", "expected_fragment"),
    [
        ("content_identity", "data_pack_content_identity"),
        ("record_count", "record_count"),
        ("batch_size", "batch_size"),
        ("relational_target", "relational_target"),
        ("vector_target", "vector_target"),
        ("verification_mode", "verification_mode"),
    ],
)
def test_resume_incompatible_request(mutator_name: str, expected_fragment: str) -> None:
    store = InMemoryBootstrapCheckpointStore()
    _service(4, checkpoint_store=store).run(_request(batch_size=2))
    mutated = _request(batch_size=2, resume_mode=ResumeMode.RESUME)
    if mutator_name == "content_identity":
        loaded = store.load(_run_identity_for(4, batch_size=2))
        assert isinstance(loaded, BootstrapCheckpointState)
        tampered = replace(
            loaded,
            run_identity=replace(
                loaded.run_identity,
                data_pack_content_identity="changed-in-checkpoint",
            ),
        )
        store.states[run_identity_digest(_run_identity_for(4, batch_size=2))] = tampered
        result = _service(4, checkpoint_store=store).run(
            _request(batch_size=2, resume_mode=ResumeMode.RESUME)
        )
    elif mutator_name == "record_count":
        loaded = store.load(_run_identity_for(4, batch_size=2))
        assert isinstance(loaded, BootstrapCheckpointState)
        tampered = replace(
            loaded,
            total_records=999,
            run_identity=replace(loaded.run_identity, record_count=999),
        )
        store.states[run_identity_digest(_run_identity_for(4, batch_size=2))] = tampered
        result = _service(4, checkpoint_store=store).run(
            _request(batch_size=2, resume_mode=ResumeMode.RESUME)
        )
    elif mutator_name == "batch_size":
        result = _service(4, checkpoint_store=store).run(_request(batch_size=3, resume_mode=ResumeMode.RESUME))
        expected_fragment = "batch_size"
        assert result.failure is not None
        assert expected_fragment in (result.failure.detail or "")
        return
    elif mutator_name == "relational_target":
        mutated = replace(mutated, relational_target=RelationalTargetId("other-relational"))
        result = _service(4, checkpoint_store=store).run(mutated)
    elif mutator_name == "vector_target":
        mutated = replace(mutated, vector_target=VectorTargetId("other-vector"))
        result = _service(4, checkpoint_store=store).run(mutated)
    elif mutator_name == "verification_mode":
        mutated = replace(mutated, verification_mode=VerificationMode.SKIP)
        result = _service(4, checkpoint_store=store).run(mutated)
    else:
        raise AssertionError(mutator_name)
    assert result.failure is not None
    assert result.failure.category is BootstrapFailureCategory.CHECKPOINT_FAILED
    assert expected_fragment in (result.failure.detail or "")


def test_resume_missing_checkpoint_fails() -> None:
    result = _service(2, checkpoint_store=InMemoryBootstrapCheckpointStore()).run(
        _request(batch_size=2, resume_mode=ResumeMode.RESUME)
    )
    assert result.failure is not None
    assert "not found" in (result.failure.detail or "")


# --- CORRUPTION ---


def test_malformed_checkpoint_fails_closed(tmp_path: Path) -> None:
    store = _filesystem_store(tmp_path)
    identity = _run_identity_for(2, batch_size=2)
    path = store.checkpoint_root / run_identity_digest(identity) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(CheckpointCorrupt):
        store.load(identity)


def test_checksum_mismatch_fails_closed() -> None:
    state = initial_checkpoint_state(
        run_identity=_run_identity_for(2, batch_size=2),
        plan=_plan_for(2, batch_size=2),
        updated_at_utc=utc_now_iso(),
    )
    encoded = encode_checkpoint(state)
    payload = json.loads(encoded.text)
    payload["state_checksum_sha256"] = "0" * 64
    with pytest.raises(CheckpointCorrupt):
        decode_checkpoint(json.dumps(payload))


def test_non_contiguous_committed_prefix_rejected() -> None:
    identity = _run_identity_for(6, batch_size=2)
    plan = _plan_for(6, batch_size=2)
    state = initial_checkpoint_state(run_identity=identity, plan=plan, updated_at_utc=utc_now_iso())
    corrupt = BootstrapCheckpointState(
        schema_version=state.schema_version,
        run_identity=state.run_identity,
        batch_size=state.batch_size,
        total_records=state.total_records,
        batch_count=state.batch_count,
        last_committed_batch_number=2,
        last_committed_global_row_index=5,
        last_committed_identity="offer:0",
        committed_record_count=6,
        batch_phases=(
            BootstrapBatchPhase.COMMITTED,
            BootstrapBatchPhase.PENDING,
            BootstrapBatchPhase.COMMITTED,
        ),
        state_revision=state.state_revision,
        updated_at_utc=state.updated_at_utc,
    )
    with pytest.raises(CheckpointCorrupt):
        compute_resume_decision(corrupt)


def test_unsupported_schema_version_rejected() -> None:
    state = initial_checkpoint_state(
        run_identity=_run_identity_for(2, batch_size=2),
        plan=_plan_for(2, batch_size=2),
        updated_at_utc=utc_now_iso(),
    )
    encoded = encode_checkpoint(state).text.replace(
        VPI_STORAGE_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION,
        "vpi-storage-bootstrap-checkpoint-v0",
    )
    with pytest.raises(CheckpointCorrupt):
        decode_checkpoint(encoded)


# --- CONCURRENCY ---


def test_stale_revision_commit_rejected() -> None:
    store = InMemoryBootstrapCheckpointStore()
    identity = _run_identity_for(2, batch_size=2)
    plan = _plan_for(2, batch_size=2)
    initial = initial_checkpoint_state(run_identity=identity, plan=plan, updated_at_utc=utc_now_iso())
    store.initialize(identity, initial)
    advanced = replace(initial, state_revision=2, committed_record_count=2, last_committed_batch_number=0)
    store.commit_batch(run_identity=identity, expected_revision=1, checkpoint=advanced)
    with pytest.raises(CheckpointConcurrentModification):
        store.commit_batch(run_identity=identity, expected_revision=1, checkpoint=advanced)


def test_two_writers_no_last_write_wins() -> None:
    store = InMemoryBootstrapCheckpointStore()
    identity = _run_identity_for(2, batch_size=2)
    plan = _plan_for(2, batch_size=2)
    initial = initial_checkpoint_state(run_identity=identity, plan=plan, updated_at_utc=utc_now_iso())
    store.initialize(identity, initial)
    first = replace(initial, state_revision=2, committed_record_count=1, last_committed_batch_number=0)
    store.commit_batch(run_identity=identity, expected_revision=1, checkpoint=first)
    stale = replace(first, state_revision=2, committed_record_count=999)
    with pytest.raises(CheckpointConcurrentModification):
        store.commit_batch(run_identity=identity, expected_revision=1, checkpoint=stale)


# --- COMPLETED RESUME ---


def test_completed_resume_zero_provider_writes() -> None:
    store = InMemoryBootstrapCheckpointStore()
    relational = CountingRelationalAdapter()
    vector = CountingVectorAdapter()
    _service(2, relational=relational, vector=vector, checkpoint_store=store).run(_request(batch_size=2))
    relational.write_calls = 0
    vector.write_calls = 0
    result = _service(
        2,
        relational=relational,
        vector=vector,
        checkpoint_store=store,
    ).run(_request(batch_size=2, resume_mode=ResumeMode.RESUME))
    assert result.status is BootstrapFinalStatus.SUCCESS
    assert relational.write_calls == 0
    assert vector.write_calls == 0
    assert result.committed_batches == 0


def test_repeated_completed_resume_is_deterministic() -> None:
    store = InMemoryBootstrapCheckpointStore()
    _service(2, checkpoint_store=store).run(_request(batch_size=2))
    first = _service(2, checkpoint_store=store).run(_request(batch_size=2, resume_mode=ResumeMode.RESUME))
    second = _service(2, checkpoint_store=store).run(_request(batch_size=2, resume_mode=ResumeMode.RESUME))
    assert first.status is BootstrapFinalStatus.SUCCESS
    assert second.status is BootstrapFinalStatus.SUCCESS


# --- PLAN ONLY ---


def test_plan_only_does_not_create_checkpoint(tmp_path: Path) -> None:
    store = _filesystem_store(tmp_path)
    _service(4, checkpoint_store=store).plan(_request(batch_size=2))  # type: ignore[arg-type]
    assert list((tmp_path / "bootstrap-state").glob("**/*")) == []


# --- FILESYSTEM DURABILITY ---


def test_filesystem_atomic_create_and_read(tmp_path: Path) -> None:
    store = _filesystem_store(tmp_path)
    identity = _run_identity_for(2, batch_size=2)
    plan = _plan_for(2, batch_size=2)
    state = initial_checkpoint_state(run_identity=identity, plan=plan, updated_at_utc=utc_now_iso())
    store.initialize(identity, state)
    loaded = store.load(identity)
    assert loaded == state


def test_filesystem_replace_keeps_old_state_on_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _filesystem_store(tmp_path)
    identity = _run_identity_for(2, batch_size=2)
    plan = _plan_for(2, batch_size=2)
    initial = initial_checkpoint_state(run_identity=identity, plan=plan, updated_at_utc=utc_now_iso())
    store.initialize(identity, initial)

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", _boom)
    advanced = replace(initial, state_revision=2, committed_record_count=2, last_committed_batch_number=0)
    with pytest.raises(CheckpointPersistenceError):
        store.commit_batch(run_identity=identity, expected_revision=1, checkpoint=advanced)
    assert store.load(identity) == initial


def test_filesystem_initialize_rejects_duplicate(tmp_path: Path) -> None:
    store = _filesystem_store(tmp_path)
    identity = _run_identity_for(2, batch_size=2)
    plan = _plan_for(2, batch_size=2)
    state = initial_checkpoint_state(run_identity=identity, plan=plan, updated_at_utc=utc_now_iso())
    store.initialize(identity, state)
    with pytest.raises(CheckpointAlreadyExists):
        store.initialize(identity, state)


# --- SECURITY ---


def test_encoded_checkpoint_has_no_forbidden_keys() -> None:
    state = initial_checkpoint_state(
        run_identity=_run_identity_for(2, batch_size=2),
        plan=_plan_for(2, batch_size=2),
        updated_at_utc=utc_now_iso(),
    )
    encoded = encode_checkpoint(state).text.lower()
    for forbidden in ("password", "dsn", "api_key", "record_json", "semantic_text", "dense_embedding"):
        assert forbidden not in encoded


# --- PROGRESS ---


def test_resume_emits_resumed_progress() -> None:
    store = InMemoryBootstrapCheckpointStore()
    _service(4, checkpoint_store=store).run(_request(batch_size=2))
    events: list[BootstrapProgress] = []

    class _Sink:
        def emit(self, progress: BootstrapProgress) -> None:
            events.append(progress)

    _service(4, checkpoint_store=store).run(_request(batch_size=2, resume_mode=ResumeMode.RESUME), progress_sink=_Sink())
    assert any(event.phase.value == "RESUMED" for event in events)


# --- PORTABILITY ---


@pytest.mark.parametrize("vector_label", ("qdrant-like", "pgvector-like"))
def test_same_checkpoint_with_vector_compositions(vector_label: str) -> None:
    store = InMemoryBootstrapCheckpointStore()
    result = _service(
        4,
        vector=FakeVectorAdapter(adapter_label=vector_label),
        checkpoint_store=store,
    ).run(_request(batch_size=2))
    assert result.status is BootstrapFinalStatus.SUCCESS


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.split(".")[0])
    return imports


@pytest.mark.parametrize("provider", sorted(_FORBIDDEN_PROVIDER_IMPORTS))
def test_checkpoint_core_has_no_provider_imports(provider: str) -> None:
    violations: list[str] = []
    for module_path in sorted(_CHECKPOINT_ROOT.rglob("*.py")):
        if provider in _module_imports(module_path):
            violations.append(str(module_path.relative_to(_REPO_ROOT)))
    assert violations == []


def test_validate_compatibility_accepts_matching_manifest() -> None:
    manifest = _ready_manifest(4)
    request = _request(batch_size=2)
    identity = build_run_identity(manifest=manifest, request=request)
    plan = _plan_for(4, batch_size=2)
    state = initial_checkpoint_state(run_identity=identity, plan=plan, updated_at_utc=utc_now_iso())
    compatibility = validate_checkpoint_compatibility(
        run_identity=identity,
        checkpoint=state,
        manifest=manifest,
    )
    assert compatibility.is_compatible
