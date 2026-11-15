# © Artur Czarnecki. All rights reserved.
# pyright: reportArgumentType=false, reportCallIssue=false

"""T-IGX-041-B pinned tests: F-D2-11 a raising reindex hook cannot break retrieval or ingest."""

from __future__ import annotations

from typing import Any

import pytest
from intergrax.rag.governance.embedding_version_policy import (
    ReindexQueueRequest,
    _notify_reindex_needed,
    clear_reindex_queue_hooks,
    evaluate_ingest_embedding_version,
    register_reindex_queue_hook,
)
from intergrax.rag.governance.source_freshness import SourceFreshnessObservation, SourceFreshnessStatus
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from tests.qualification.pdc_x.d2.test_d2_rag_freshness_wiring import _hit, _service

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _isolated_reindex_hooks() -> Any:
    clear_reindex_queue_hooks()
    yield
    clear_reindex_queue_hooks()


def _raising(request: ReindexQueueRequest) -> None:
    raise RuntimeError("operator hook exploded")


def _stale_service(**kwargs: Any):  # type: ignore[no-untyped-def]
    hits = [_hit("a", "docs/a.md", revision="rev-1")]
    current = {"docs/a.md": SourceFreshnessObservation(source_id="docs/a.md", revision="rev-2")}
    return _service(hits, source_observer=current.get, **kwargs)


def test_f_d2_11_a_raising_hook_does_not_break_a_successful_retrieval() -> None:
    register_reindex_queue_hook(_raising)

    result = _stale_service().retrieve(RetrievalRequest(query="policy"))

    assert result.used is True
    assert [chunk.id for chunk in result.chunks] == ["a"]
    assert result.freshness is not None
    assert result.freshness.verdicts[0].status is SourceFreshnessStatus.STALE


def test_f_d2_11_the_other_hooks_still_receive_the_request() -> None:
    received: list[ReindexQueueRequest] = []
    register_reindex_queue_hook(_raising)
    register_reindex_queue_hook(received.append)

    result = _stale_service().retrieve(RetrievalRequest(query="policy"))

    assert result.used is True
    assert [(r.source_path, r.reason) for r in received] == [("docs/a.md", "source_revision_stale")]


def test_f_d2_11_the_hook_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    register_reindex_queue_hook(_raising)
    with caplog.at_level("WARNING"):
        _notify_reindex_needed(source_path="docs/a.md", current_version="1", target_version="2", reason="r")
    assert any("operator hook exploded" in (r.exc_text or "") or "reindex" in r.getMessage() for r in caplog.records)


def test_f_d2_11_a_raising_hook_does_not_break_ingest_version_evaluation() -> None:
    register_reindex_queue_hook(_raising)
    profile = RagProfile(embedding_model_version="v2", embedding_version_warn_on_ingest=True)

    result = evaluate_ingest_embedding_version(
        profile=profile,
        base_metadata={"embedding_model_version": "v1"},
        source_path="docs/a.md",
    )

    assert result.warnings == ["incoming_metadata_version_mismatch:v1!=v2"]
    assert result.reindex_enqueued is False


def test_f_d2_11_enqueued_is_true_when_a_healthy_hook_accepted_the_request() -> None:
    register_reindex_queue_hook(_raising)
    register_reindex_queue_hook(lambda request: None)
    assert _notify_reindex_needed(source_path="docs/a.md", current_version="1", target_version="2", reason="r") is True


def test_f_d2_11_without_hooks_nothing_is_enqueued() -> None:
    assert _notify_reindex_needed(source_path="docs/a.md", current_version="1", target_version="2", reason="r") is False


def test_f_d2_11_a_healthy_hook_alone_keeps_todays_behavior() -> None:
    received: list[ReindexQueueRequest] = []
    register_reindex_queue_hook(received.append)

    result = _stale_service().retrieve(RetrievalRequest(query="policy"))

    assert result.freshness is not None and result.freshness.reindex_requested is True
    assert len(received) == 1
