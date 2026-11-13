# © Artur Czarnecki. All rights reserved.

"""XD-09 slice owned by B-D8: PDC-D8-R012 (spill round trip) and R013 (context ref after compaction).

Re-pointed by PDC-P6-01 to the cross-domain integration suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.context.budget.compaction import DeterministicTailCompactionStrategy
from intergrax.context.budget.compaction_apply import apply_fragment_compaction
from intergrax.context.contracts import ContextFragment, ContextFragmentSource
from intergrax.contracts.artifact_ref import (
    ArtifactClass,
    ArtifactDigestMismatchError,
    ArtifactRetentionClass,
    compute_artifact_digest,
)
from intergrax.runtime.nexus.artifacts.context_reference import (
    ArtifactReferenceResolver,
    ArtifactReferenceToken,
    reference_fragment,
    token_from_fragment,
)
from intergrax.runtime.nexus.artifacts.governed_access import ArtifactAccessDenied
from intergrax.runtime.nexus.artifacts.retention import ArtifactRetentionSchedule
from intergrax.runtime.nexus.artifacts.spill import ToolOutputSpiller

from ._d8_support import (
    T0,
    TENANT_A,
    TENANT_B,
    MutableClock,
    make_context,
    make_governance,
    make_port,
    reader_grants,
    sqlite_store,
)

pytestmark = pytest.mark.unit

BIG = ("row,value,notes\n" + "alpha,1,some tabular tool output\n" * 4000).encode()
THRESHOLD = 4096


def _spiller(store, clock) -> ToolOutputSpiller:
    return ToolOutputSpiller(
        store,
        threshold_bytes=THRESHOLD,
        schedule=ArtifactRetentionSchedule.default(),
        clock=clock,
    )


def _spill(spiller: ToolOutputSpiller, content: bytes | str = BIG, **kw):
    return spiller.spill_if_large(
        tenant_id=TENANT_A,
        run_id="run-1",
        task_id="task-1",
        tool_id="sql.query",
        agent_id="agent-1",
        content=content,
        **kw,
    )


# ------------------------------------------------------------------ R012


def test_small_tool_output_stays_inline(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    result = _spill(_spiller(store, MutableClock()), content="tiny output")
    assert result.inline == "tiny output" and result.reference is None
    assert list(store.list_for_run(TENANT_A, "run-1")) == []


def test_large_tool_output_spills_to_an_artifact_reference_and_round_trips(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    clock = MutableClock()
    result = _spill(_spiller(store, clock))
    assert result.inline is None and result.reference is not None
    token = result.reference
    assert token.content_digest == compute_artifact_digest(BIG)
    assert token.artifact_class is ArtifactClass.TOOL_OUTPUT
    assert len(result.preview) < len(BIG)  # the model sees a short preview + the reference

    stored = store.get(TENANT_A, token.artifact_id)
    assert stored.retention_class is ArtifactRetentionClass.EPHEMERAL
    assert stored.expires_at_utc is not None and stored.expires_at_utc > T0
    assert stored.producer is not None and stored.producer.created_by_tool_id == "sql.query"
    assert stored.lineage is not None and stored.lineage.run_id == "run-1"

    port, _ = make_port(store, clock=clock)
    assert ArtifactReferenceResolver(port).resolve(make_context(), token) == BIG  # round trip


def test_text_and_bytes_inputs_round_trip_identically(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    spiller = _spiller(store, MutableClock())
    port, _ = make_port(store)
    resolver = ArtifactReferenceResolver(port)
    as_text = _spill(spiller, content=BIG.decode()).reference
    as_bytes = _spill(spiller, content=BIG).reference
    assert as_text is not None and as_bytes is not None
    assert resolver.resolve(make_context(), as_text) == BIG
    assert resolver.resolve(make_context(), as_bytes) == BIG


def test_spilled_output_survives_restart(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    token = _spill(_spiller(store, MutableClock())).reference
    assert token is not None
    del store
    reopened = sqlite_store(tmp_path)
    port, _ = make_port(reopened)
    assert ArtifactReferenceResolver(port).resolve(make_context(), token) == BIG


def test_a_reference_with_the_wrong_digest_is_rejected_on_resolve(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    token = _spill(_spiller(store, MutableClock())).reference
    assert token is not None
    forged = token.model_copy(update={"content_digest": compute_artifact_digest(b"other")})
    port, _ = make_port(store)
    with pytest.raises(ArtifactDigestMismatchError):
        ArtifactReferenceResolver(port).resolve(make_context(), forged)


def test_resolving_a_reference_goes_through_the_governed_port(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    token = _spill(_spiller(store, MutableClock())).reference
    assert token is not None
    harness = make_governance(granted={})
    port, _ = make_port(store, harness=harness)
    with pytest.raises(ArtifactAccessDenied):
        ArtifactReferenceResolver(port).resolve(make_context(), token)
    assert harness.sink.events  # the denial is in evidence


# ------------------------------------------------------------------ R013


def test_context_reference_resolves_after_real_fragment_compaction(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    token = _spill(_spiller(store, MutableClock())).reference
    assert token is not None
    ref_fragment = reference_fragment(token, fragment_id="frag-ref")
    assert ref_fragment.mandatory
    noisy = [
        ContextFragment(
            fragment_id=f"noise-{i}",
            source=ContextFragmentSource.RAG,
            source_id="rag",
            content=("lorem ipsum dolor sit amet " * 400) + f" tail-{i}",
            token_estimate=2000,
            relevance_score=0.1 * (i + 1),
            freshness_score=0.5,
            confidence_score=0.5,
            mandatory=False,
        )
        for i in range(3)
    ]
    compacted, provenance = apply_fragment_compaction(
        [ref_fragment, *noisy],
        strategy=DeterministicTailCompactionStrategy(),
        count_text=lambda text: max(1, len(text) // 4),
        fragment_budget_tokens=200,
    )
    assert provenance, "the real compaction phase must have shrunk optional fragments"
    survivor = next(f for f in compacted if f.fragment_id == "frag-ref")
    assert survivor.content == ref_fragment.content  # reference fragment untouched
    parsed = token_from_fragment(survivor)
    assert parsed == token

    port, _ = make_port(store)
    assert ArtifactReferenceResolver(port).resolve(make_context(), parsed) == BIG


def test_compaction_never_removes_the_referenced_artifact(tmp_path: Path) -> None:
    """Compaction operates on fragments only; the artifact store is untouched by it."""
    store = sqlite_store(tmp_path)
    token = _spill(_spiller(store, MutableClock())).reference
    assert token is not None
    apply_fragment_compaction(
        [reference_fragment(token, fragment_id="r")],
        strategy=DeterministicTailCompactionStrategy(),
        count_text=lambda text: len(text),
        fragment_budget_tokens=1,
    )
    assert store.get(TENANT_A, token.artifact_id).data == BIG


def test_marker_round_trips() -> None:
    token = ArtifactReferenceToken(
        tenant_id=TENANT_A,
        artifact_class=ArtifactClass.TOOL_OUTPUT,
        artifact_id="art-9",
        version=2,
        content_digest=compute_artifact_digest(b"x"),
    )
    assert ArtifactReferenceToken.from_marker(token.to_marker()) == token
    with pytest.raises(ValueError):
        ArtifactReferenceToken.from_marker("not a marker")


def test_context_reference_of_another_tenant_is_not_resolvable(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    token = _spill(_spiller(store, MutableClock())).reference
    assert token is not None
    harness = make_governance(granted=reader_grants(tenant_id=TENANT_B))
    port, _ = make_port(store, harness=harness)
    with pytest.raises(ArtifactAccessDenied) as exc:
        ArtifactReferenceResolver(port).resolve(make_context(tenant_id=TENANT_B), token)
    assert exc.value.reason_code == "tenant_mismatch"
