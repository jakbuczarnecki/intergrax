# © Artur Czarnecki. All rights reserved.

"""P2.1-S2C3 — delegated correlation list/query read model."""

from __future__ import annotations

import ast
import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.contracts.delegated_execution_invocation_binding import (
    DelegatedExecutionInvocationBinding,
    mint_delegated_execution_invocation_binding,
)
from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionBudgetMode,
    DelegatedExecutionBudgetProjection,
    DelegatedExecutionContext,
    DelegatedExecutionOperationMetadata,
    DelegatedExecutionProvider,
    DelegatedExecutionRequest,
    digest_delegated_execution_request,
)
from intergrax.contracts.delegated_execution_query import (
    DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE,
    DELEGATED_EXECUTION_QUERY_VALIDATION_FAILURE_MESSAGE,
    DEFAULT_DELEGATED_CORRELATION_QUERY_PAGE_SIZE,
    MAX_DELEGATED_CORRELATION_QUERY_PAGE_SIZE,
    DelegatedExecutionCorrelationView,
    DelegatedExecutionQueryInvalidCursorError,
    DelegatedExecutionQueryValidationError,
    DelegatedInvocationCorrelationQuery,
    DelegatedInvocationCorrelationQueryStore,
)
from intergrax.contracts.delegated_invocation_correlation import (
    DELEGATED_INVOCATION_CORRELATION_INTEGRITY_FAILURE_MESSAGE,
    DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE,
    DelegatedInvocationCorrelationIntegrityError,
    DelegatedInvocationCorrelationPersistenceError,
    DelegatedInvocationCorrelationRecord,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    ExecutionId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.contracts.provider_invocation import ProviderInvocation
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.execution.delegated_execution.correlation_persistence import (
    DocumentStoreDelegatedInvocationCorrelationStore,
    InMemoryDelegatedInvocationCorrelationStore,
    _DOCUMENT_PARTITION,
)
from intergrax.runtime.execution.delegated_execution.correlation_query_cursor import (
    DelegatedCorrelationQueryCursorCodec,
)
from intergrax.runtime.execution.delegated_execution.correlation_query_persistence import (
    DocumentStoreDelegatedInvocationCorrelationQueryStore,
    paired_in_memory_correlation_stores,
)
from intergrax.contracts.delegated_execution_query import (
    DelegatedInvocationCorrelationQueryStorePage,
    delegated_correlation_backend_scan_limit,
)
from intergrax.runtime.execution.delegated_execution.correlation_persistence import (
    backfill_correlation_document_query_index,
    encode_correlation_record,
)
from intergrax.runtime.execution.delegated_execution.correlation_service import (
    DelegatedInvocationCorrelationService,
)
from intergrax.runtime.execution.delegated_execution.query_service import (
    DelegatedExecutionQueryService,
)
from intergrax.runtime.execution.delegated_execution.status_service import (
    DelegatedExecutionStatusReadService,
)
from intergrax.contracts.delegated_execution_status import (
    DelegatedExecutionStatusProvider,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_QUERY_SERVICE_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "delegated_execution"
    / "query_service.py"
)
_QUERY_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "delegated_execution_query.py"
_T0 = datetime(2026, 9, 7, 8, 0, 0, tzinfo=timezone.utc)
_PAYLOAD_DIGEST = "sha256:" + ("ab" * 32)
_CURSOR_SECRET = b"c" * 32


@dataclass(frozen=True)
class EchoPayload:
    value: str


@dataclass(frozen=True)
class EchoResult:
    value: str


def _operation() -> DelegatedExecutionOperationMetadata:
    return DelegatedExecutionOperationMetadata.model_validate(
        {"operation": "execute_delegate", "task_id": "task_query"},
    )


def _context(
    *,
    parent_execution_id: ExecutionId | None = None,
    run_id=None,
) -> DelegatedExecutionContext:
    return DelegatedExecutionContext(
        execution_id=mint_execution_id(),
        parent_execution_id=parent_execution_id or mint_execution_id(),
        run_id=run_id or mint_run_id(),
        attempt_id=mint_attempt_id(),
        authority=ParentExecutionAuthority.scoped(("delegated.query",)),
        budget=DelegatedExecutionBudgetProjection(
            allocation_mode=DelegatedExecutionBudgetMode.SHARED,
        ),
    )


def _invocation(
    *,
    provider_id: str,
    context: DelegatedExecutionContext,
    invocation_id: str,
) -> ProviderInvocation:
    digest = digest_delegated_execution_request(
        context=context,
        operation=_operation(),
        payload_digest=_PAYLOAD_DIGEST,
    )
    return ProviderInvocation.model_validate(
        {
            "invocation_id": invocation_id,
            "provider_id": provider_id,
            "operation": _operation().operation,
            "task_id": _operation().task_id,
            "run_id": str(context.run_id),
            "request_digest": digest,
            "started_at": _T0.isoformat(),
            "provider_request_id": f"preq-{invocation_id}",
            "provider_operation_id": f"pop-{invocation_id}",
        }
    )


def _binding(
    *,
    provider_id: str = "query_fake",
    parent: ExecutionId | None = None,
    run_id=None,
    invocation_id: str = "inv-q",
) -> DelegatedExecutionInvocationBinding:
    ctx = _context(parent_execution_id=parent, run_id=run_id)
    return mint_delegated_execution_invocation_binding(
        context=ctx,
        operation=_operation(),
        payload_digest=_PAYLOAD_DIGEST,
        provider_invocation=_invocation(
            provider_id=provider_id,
            context=ctx,
            invocation_id=invocation_id,
        ),
    )


def _query_stack() -> tuple[
    DelegatedInvocationCorrelationService,
    DelegatedExecutionQueryService,
]:
    write_store, query_store = paired_in_memory_correlation_stores()
    correlation = DelegatedInvocationCorrelationService(write_store)
    query = DelegatedExecutionQueryService(query_store)
    return correlation, query


def _persist_many(
    correlation: DelegatedInvocationCorrelationService,
    count: int,
    *,
    parent: ExecutionId | None = None,
    provider_id: str = "query_fake",
    base_time: datetime = _T0,
) -> list[DelegatedExecutionInvocationBinding]:
    bindings: list[DelegatedExecutionInvocationBinding] = []
    for index in range(count):
        binding = _binding(
            provider_id=provider_id,
            parent=parent,
            invocation_id=f"inv-{index}",
        )
        correlation.persist_binding(
            binding,
            persisted_at=base_time + timedelta(seconds=index),
        )
        bindings.append(binding)
    return bindings


def test_s2c3_t1_list_first_page() -> None:
    correlation, query_service = _query_stack()
    _persist_many(correlation, 5)
    page = query_service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=2),
    )
    assert len(page.items) == 2
    assert page.has_more is True
    assert page.next_cursor is not None
    times = [item.persisted_at for item in page.items]
    assert times[0] >= times[1]


def test_s2c3_t2_second_page_no_duplicates() -> None:
    correlation, query_service = _query_stack()
    bindings = _persist_many(correlation, 5)
    first = query_service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=2),
    )
    second = query_service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=2, cursor=first.next_cursor),
    )
    first_ids = {str(item.execution_id) for item in first.items}
    second_ids = {str(item.execution_id) for item in second.items}
    assert first_ids.isdisjoint(second_ids)
    assert len(first_ids | second_ids) == 4
    all_expected = {str(binding.execution_id) for binding in bindings}
    assert (first_ids | second_ids).issubset(all_expected)


def test_s2c3_t3_empty_result() -> None:
    _, query_service = _query_stack()
    page = query_service.query_delegated_executions(DelegatedInvocationCorrelationQuery())
    assert page.items == ()
    assert page.next_cursor is None
    assert page.has_more is False


def test_s2c3_t4_page_limit_max_validation() -> None:
    with pytest.raises(ValidationError):
        DelegatedInvocationCorrelationQuery(
            page_size=MAX_DELEGATED_CORRELATION_QUERY_PAGE_SIZE + 1,
        )


def test_s2c3_t5_page_limit_zero_validation() -> None:
    with pytest.raises(ValidationError):
        DelegatedInvocationCorrelationQuery(page_size=0)


def test_s2c3_t6_filter_parent_execution() -> None:
    correlation, query_service = _query_stack()
    parent = mint_execution_id()
    other_parent = mint_execution_id()
    _persist_many(correlation, 2, parent=parent)
    _persist_many(correlation, 3, parent=other_parent)
    page = query_service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(parent_execution_id=parent, page_size=10),
    )
    assert len(page.items) == 2
    assert all(item.parent_execution_id == parent for item in page.items)


def test_s2c3_t7_filter_provider() -> None:
    correlation, query_service = _query_stack()
    _persist_many(correlation, 2, provider_id="alpha")
    _persist_many(correlation, 3, provider_id="beta")
    page = query_service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(provider_id="alpha", page_size=10),
    )
    assert len(page.items) == 2
    assert all(item.provider_id == "alpha" for item in page.items)


def test_s2c3_t8_filter_time_window() -> None:
    correlation, query_service = _query_stack()
    _persist_many(correlation, 1, base_time=_T0)
    _persist_many(correlation, 1, base_time=_T0 + timedelta(hours=2))
    page = query_service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(
            persisted_from=_T0 + timedelta(hours=1),
            persisted_to=_T0 + timedelta(hours=3),
            page_size=10,
        ),
    )
    assert len(page.items) == 1
    assert page.items[0].persisted_at == _T0 + timedelta(hours=2)


def test_s2c3_t9_combined_filters_and() -> None:
    correlation, query_service = _query_stack()
    parent = mint_execution_id()
    _persist_many(correlation, 2, parent=parent, provider_id="keep")
    _persist_many(correlation, 2, parent=parent, provider_id="drop")
    page = query_service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(
            parent_execution_id=parent,
            provider_id="keep",
            page_size=10,
        ),
    )
    assert len(page.items) == 2


def test_s2c3_t10_deterministic_order_equal_persisted_at() -> None:
    correlation, query_service = _query_stack()
    same_time = _T0 + timedelta(minutes=5)
    bindings = []
    for index in range(3):
        binding = _binding(invocation_id=f"tie-{index}")
        correlation.persist_binding(binding, persisted_at=same_time)
        bindings.append(binding)
    page = query_service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=10),
    )
    ids = [str(item.execution_id) for item in page.items]
    assert ids == sorted(ids, reverse=True)


def test_s2c3_t11_cursor_opaque() -> None:
    correlation, query_service = _query_stack()
    _persist_many(correlation, 4)
    first = query_service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=2),
    )
    assert first.next_cursor is not None
    raw = base64.urlsafe_b64decode(first.next_cursor + "==")
    assert b"query_fake" not in raw


def test_s2c3_t12_cursor_invalid() -> None:
    _, query_service = _query_stack()
    with pytest.raises(DelegatedExecutionQueryInvalidCursorError) as exc:
        query_service.query_delegated_executions(
            DelegatedInvocationCorrelationQuery(cursor="not-a-valid-cursor"),
        )
    assert str(exc.value) == DELEGATED_EXECUTION_QUERY_INVALID_CURSOR_MESSAGE


def test_s2c3_t13_corrupted_record_fail_closed() -> None:
    doc = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    write = DocumentStoreDelegatedInvocationCorrelationStore(doc)
    query_store = DocumentStoreDelegatedInvocationCorrelationQueryStore(
        doc,
        cursor_secret=_CURSOR_SECRET,
    )
    correlation = DelegatedInvocationCorrelationService(write)
    binding = _binding()
    correlation.persist_binding(binding, persisted_at=_T0)
    doc.put(
        DocumentRecord(
            partition_key=_DOCUMENT_PARTITION,
            row_key="corrupt-row",
            data={
                "correlation": "not-json",
                "query_parent_execution_id": str(mint_execution_id()),
                "query_provider_id": "corrupt",
                "query_persisted_at": (_T0 + timedelta(hours=1)).isoformat(),
            },
        ),
    )
    service = DelegatedExecutionQueryService(query_store)
    with pytest.raises(DelegatedInvocationCorrelationIntegrityError):
        service.query_delegated_executions(DelegatedInvocationCorrelationQuery(page_size=10))


def test_s2c3_t14_storage_unavailable() -> None:
    class BrokenStore(DelegatedInvocationCorrelationQueryStore):
        @property
        def cursor_codec(self) -> DelegatedCorrelationQueryCursorCodec:
            return DelegatedCorrelationQueryCursorCodec(secret=_CURSOR_SECRET)

        def query_page(
            self,
            query: DelegatedInvocationCorrelationQuery,
        ) -> DelegatedInvocationCorrelationQueryStorePage:
            raise RuntimeError("mongo socket reset")

    service = DelegatedExecutionQueryService(BrokenStore())
    with pytest.raises(DelegatedInvocationCorrelationPersistenceError) as exc:
        service.query_delegated_executions(DelegatedInvocationCorrelationQuery())
    assert str(exc.value) == DELEGATED_INVOCATION_CORRELATION_PERSISTENCE_UNAVAILABLE_MESSAGE
    assert "mongo" not in str(exc.value)


@pytest.mark.asyncio
async def test_s2c3_t15_no_provider_calls() -> None:
    class ExplodingProvider(DelegatedExecutionProvider[EchoPayload, EchoResult]):
        @property
        def provider_id(self) -> str:
            return "query_fake"

        @property
        def provider_version(self) -> str:
            return "0.0.1"

        @property
        def capabilities(self):
            raise AssertionError("provider must not be invoked during query")

        async def execute(self, request: DelegatedExecutionRequest[EchoPayload]):
            raise AssertionError("provider execute must not run during query")

    correlation, query_service = _query_stack()
    _persist_many(correlation, 1)
    _ = query_service.query_delegated_executions(DelegatedInvocationCorrelationQuery())
    _ = ExplodingProvider()


def test_s2c3_t16_no_correlation_writes() -> None:
    write_store, query_store = paired_in_memory_correlation_stores()
    correlation = DelegatedInvocationCorrelationService(write_store)
    bindings = _persist_many(correlation, 2)
    service = DelegatedExecutionQueryService(query_store)
    service.query_delegated_executions(DelegatedInvocationCorrelationQuery(page_size=10))
    for binding in bindings:
        assert (
            correlation.load_binding_by_execution_id(binding.execution_id) == binding
        )


def test_s2c3_t17_query_store_pluginability() -> None:
    class StaticQueryStore(DelegatedInvocationCorrelationQueryStore):
        @property
        def cursor_codec(self) -> DelegatedCorrelationQueryCursorCodec:
            return DelegatedCorrelationQueryCursorCodec(secret=_CURSOR_SECRET)

        def query_page(
            self,
            query: DelegatedInvocationCorrelationQuery,
        ) -> DelegatedInvocationCorrelationQueryStorePage:
            return DelegatedInvocationCorrelationQueryStorePage(records=(), has_more=False)

    page = DelegatedExecutionQueryService(StaticQueryStore()).query_delegated_executions(
        DelegatedInvocationCorrelationQuery(),
    )
    assert page.items == ()


def test_s2c3_t18_no_concrete_db_import_in_service() -> None:
    source = _QUERY_SERVICE_MODULE.read_text(encoding="utf-8")
    forbidden = ("mongodb", "cassandra", "dynamodb", "pymongo", "InMemoryDocumentStore")
    for token in forbidden:
        assert token not in source


def test_s2c3_t19_no_reflection() -> None:
    tree = ast.parse(_QUERY_SERVICE_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"getattr", "hasattr", "setattr"}


def test_s2c3_t20_no_global_cursor_registry() -> None:
    source = _QUERY_SERVICE_MODULE.read_text(encoding="utf-8")
    assert "cursor_registry" not in source.lower()


def test_s2c3_t21_no_any_abi() -> None:
    source = _QUERY_CONTRACT.read_text(encoding="utf-8")
    assert "Any" not in source
    assert "dict[str, Any]" not in source


def test_s2c3_t22_s2c1_regression_exact_lookup() -> None:
    store = InMemoryDelegatedInvocationCorrelationStore()
    correlation = DelegatedInvocationCorrelationService(store)
    binding = _binding()
    correlation.persist_binding(binding, persisted_at=_T0)
    loaded = correlation.load_binding_by_execution_id(binding.execution_id)
    assert loaded == binding


@pytest.mark.asyncio
async def test_s2c3_t23_s2c2_regression_status_lookup() -> None:
    from intergrax.contracts.delegated_execution_status import (
        DelegatedExecutionProviderPhysicalStatus,
        DelegatedExecutionStatusOutcomeCategory,
        DelegatedExecutionStatusProvider,
    )
    from intergrax.runtime.execution.delegated_execution.provider_resolver import (
        MappingDelegatedExecutionProviderResolver,
    )

    class MinimalStatusProvider(DelegatedExecutionStatusProvider):
        @property
        def provider_id(self) -> str:
            return "query_fake"

        @property
        def capabilities(self):
            from intergrax.contracts.delegated_execution_provider import (
                DelegatedExecutionCapabilities,
            )

            return DelegatedExecutionCapabilities(
                provider_id="query_fake",
                supports_cancel=False,
                supports_status_read=True,
            )

        async def read_delegated_execution_status(self, request):
            from intergrax.contracts.delegated_execution_status import (
                DelegatedExecutionProviderStatusObservation,
            )

            inv = request.provider_invocation
            return DelegatedExecutionProviderStatusObservation(
                physical_status=DelegatedExecutionProviderPhysicalStatus.RUNNING,
                provider_id=self.provider_id,
                invocation_id=inv.invocation_id,
                provider_request_id=inv.provider_request_id,
                provider_operation_id=inv.provider_operation_id,
            )

    write_store, _ = paired_in_memory_correlation_stores()
    correlation = DelegatedInvocationCorrelationService(write_store)
    binding = _binding()
    correlation.persist_binding(binding, persisted_at=_T0)
    status = DelegatedExecutionStatusReadService(
        correlation,
        MappingDelegatedExecutionProviderResolver(
            {MinimalStatusProvider().provider_id: MinimalStatusProvider()},
        ),
        clock=lambda: _T0,
    )
    outcome = await status.read_status_by_execution_id(binding.execution_id)
    assert outcome.category is DelegatedExecutionStatusOutcomeCategory.AVAILABLE


def test_s2c3_t24_restart_like_durable_query() -> None:
    doc = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    write = DocumentStoreDelegatedInvocationCorrelationStore(doc)
    query_a = DocumentStoreDelegatedInvocationCorrelationQueryStore(
        doc,
        cursor_secret=_CURSOR_SECRET,
    )
    correlation = DelegatedInvocationCorrelationService(write)
    binding = _binding()
    correlation.persist_binding(binding, persisted_at=_T0)
    page_a = DelegatedExecutionQueryService(query_a).query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=10),
    )
    query_b = DocumentStoreDelegatedInvocationCorrelationQueryStore(
        doc,
        cursor_secret=_CURSOR_SECRET,
    )
    page_b = DelegatedExecutionQueryService(query_b).query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=10),
    )
    assert len(page_a.items) == 1
    assert page_a.items[0].execution_id == binding.execution_id
    assert page_b.items[0].execution_id == binding.execution_id


def test_s2c3_t26_query_view_has_no_physical_status() -> None:
    assert "physical_status" not in DelegatedExecutionCorrelationView.model_fields


def test_s2c3_time_window_validation() -> None:
    with pytest.raises(ValidationError):
        DelegatedInvocationCorrelationQuery(
            persisted_from=_T0 + timedelta(hours=2),
            persisted_to=_T0,
        )


def test_s2c3_default_page_size_matches_canonical() -> None:
    query = DelegatedInvocationCorrelationQuery()
    assert query.page_size == DEFAULT_DELEGATED_CORRELATION_QUERY_PAGE_SIZE


def test_s2c3_architecture_gate_no_nexus() -> None:
    tree = ast.parse(_QUERY_SERVICE_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "runtime.nexus" not in node.module


class _InstrumentedDocumentStore(InMemoryDocumentStore):
    def __init__(self, *, cursor_secret: bytes) -> None:
        super().__init__(cursor_secret=cursor_secret)
        self.query_call_count = 0
        self.query_cursors_used: list[str | None] = []
        self.last_query_limit: int | None = None

    def query(self, partition_key: str, **kwargs):
        self.query_call_count += 1
        self.query_cursors_used.append(kwargs.get("cursor"))
        self.last_query_limit = kwargs.get("limit")
        return super().query(partition_key, **kwargs)


def _seed_document_store(
    doc: InMemoryDocumentStore,
    *,
    count: int,
    provider_id: str = "rare_provider",
    parent: ExecutionId | None = None,
    base_time: datetime = _T0,
) -> None:
    write = DocumentStoreDelegatedInvocationCorrelationStore(doc)
    correlation = DelegatedInvocationCorrelationService(write)
    for index in range(count):
        binding = _binding(
            provider_id=provider_id,
            parent=parent,
            invocation_id=f"seed-{index}",
        )
        correlation.persist_binding(
            binding,
            persisted_at=base_time + timedelta(seconds=index),
        )


def test_c1_t1_rare_filter_bounded_backend_scan() -> None:
    doc = _InstrumentedDocumentStore(cursor_secret=_CURSOR_SECRET)
    _seed_document_store(doc, count=200, provider_id="common")
    _seed_document_store(
        doc,
        count=1,
        provider_id="needle",
        base_time=_T0 + timedelta(days=1),
    )
    query_store = DocumentStoreDelegatedInvocationCorrelationQueryStore(
        doc,
        cursor_secret=_CURSOR_SECRET,
    )
    service = DelegatedExecutionQueryService(query_store)
    page = service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(provider_id="needle", page_size=100),
    )
    assert len(page.items) == 1
    assert doc.query_call_count == 1
    assert doc.last_query_limit == delegated_correlation_backend_scan_limit(100)


def test_c1_t2_short_page_when_scan_budget_exhausted() -> None:
    doc = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    write = DocumentStoreDelegatedInvocationCorrelationStore(doc)
    correlation = DelegatedInvocationCorrelationService(write)
    for index in range(87):
        binding = _binding(provider_id="drop", invocation_id=f"drop-a-{index}")
        correlation.persist_binding(
            binding,
            persisted_at=_T0 + timedelta(hours=1, seconds=index),
        )
    for index in range(13):
        binding = _binding(provider_id="keep", invocation_id=f"keep-{index}")
        correlation.persist_binding(
            binding,
            persisted_at=_T0 + timedelta(hours=2, seconds=index),
        )
    for index in range(40):
        binding = _binding(provider_id="drop", invocation_id=f"drop-b-{index}")
        correlation.persist_binding(binding, persisted_at=_T0 + timedelta(seconds=index))
    query_store = DocumentStoreDelegatedInvocationCorrelationQueryStore(
        doc,
        cursor_secret=_CURSOR_SECRET,
    )
    service = DelegatedExecutionQueryService(query_store)
    page = service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(provider_id="keep", page_size=100),
    )
    assert len(page.items) == 13
    assert page.has_more is True
    assert page.next_cursor is not None


def test_c1_t3_next_page_resumes_backend_cursor() -> None:
    doc = _InstrumentedDocumentStore(cursor_secret=_CURSOR_SECRET)
    _seed_document_store(doc, count=250)
    query_store = DocumentStoreDelegatedInvocationCorrelationQueryStore(
        doc,
        cursor_secret=_CURSOR_SECRET,
    )
    service = DelegatedExecutionQueryService(query_store)
    first = service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=100),
    )
    assert first.has_more is True
    second = service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=100, cursor=first.next_cursor),
    )
    assert doc.query_call_count == 2
    assert doc.query_cursors_used[1] is not None
    assert doc.query_cursors_used[1] != doc.query_cursors_used[0]


def test_c1_t4_no_rescan_of_previous_backend_pages() -> None:
    doc = _InstrumentedDocumentStore(cursor_secret=_CURSOR_SECRET)
    _seed_document_store(doc, count=180)
    query_store = DocumentStoreDelegatedInvocationCorrelationQueryStore(
        doc,
        cursor_secret=_CURSOR_SECRET,
    )
    service = DelegatedExecutionQueryService(query_store)
    first = service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=100),
    )
    second = service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=100, cursor=first.next_cursor),
    )
    assert doc.query_cursors_used[0] is None
    assert doc.query_cursors_used[1] is not None
    assert doc.query_cursors_used[1] != doc.query_cursors_used[0]


def test_c1_t7_cursor_query_binding_rejects_filter_change() -> None:
    correlation, query_service = _query_stack()
    _persist_many(correlation, 4)
    first = query_service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=2),
    )
    with pytest.raises(DelegatedExecutionQueryInvalidCursorError):
        query_service.query_delegated_executions(
            DelegatedInvocationCorrelationQuery(
                page_size=2,
                cursor=first.next_cursor,
                provider_id="other",
            ),
        )


def test_c1_t8_cursor_tampering_rejected() -> None:
    correlation, query_service = _query_stack()
    _persist_many(correlation, 3)
    first = query_service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=1),
    )
    tampered = f"{first.next_cursor}invalid"
    with pytest.raises(DelegatedExecutionQueryInvalidCursorError):
        query_service.query_delegated_executions(
            DelegatedInvocationCorrelationQuery(page_size=1, cursor=tampered),
        )


def test_c1_t9_cursor_backend_continuation_roundtrip() -> None:
    doc = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    _seed_document_store(doc, count=150)
    query_store = DocumentStoreDelegatedInvocationCorrelationQueryStore(
        doc,
        cursor_secret=_CURSOR_SECRET,
    )
    service = DelegatedExecutionQueryService(query_store)
    first = service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=50),
    )
    second = service.query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=50, cursor=first.next_cursor),
    )
    first_ids = {str(item.execution_id) for item in first.items}
    second_ids = {str(item.execution_id) for item in second.items}
    assert first_ids.isdisjoint(second_ids)


def test_c1_t11_legacy_record_exact_lookup() -> None:
    doc = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    write = DocumentStoreDelegatedInvocationCorrelationStore(doc)
    correlation = DelegatedInvocationCorrelationService(write)
    binding = _binding()
    correlation.persist_binding(binding, persisted_at=_T0)
    loaded = correlation.load_binding_by_execution_id(binding.execution_id)
    assert loaded == binding


def test_c1_t12_legacy_record_query_discoverability() -> None:
    doc = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    binding = _binding()
    record = DelegatedInvocationCorrelationRecord(
        binding=binding,
        persisted_at=_T0,
    )
    doc.put(
        DocumentRecord(
            partition_key=_DOCUMENT_PARTITION,
            row_key=str(binding.execution_id),
            data={"correlation": encode_correlation_record(record).decode("utf-8")},
        ),
    )
    query_store = DocumentStoreDelegatedInvocationCorrelationQueryStore(
        doc,
        cursor_secret=_CURSOR_SECRET,
    )
    page = DelegatedExecutionQueryService(query_store).query_delegated_executions(
        DelegatedInvocationCorrelationQuery(
            parent_execution_id=binding.parent_execution_id,
            page_size=10,
        ),
    )
    assert len(page.items) == 1
    assert page.items[0].execution_id == binding.execution_id


def test_c1_t14_mixed_legacy_and_indexed_dataset() -> None:
    doc = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    write = DocumentStoreDelegatedInvocationCorrelationStore(doc)
    correlation = DelegatedInvocationCorrelationService(write)
    indexed = _binding(invocation_id="indexed")
    correlation.persist_binding(indexed, persisted_at=_T0)
    legacy_binding = _binding(invocation_id="legacy")
    legacy_record = DelegatedInvocationCorrelationRecord(
        binding=legacy_binding,
        persisted_at=_T0 + timedelta(seconds=1),
    )
    doc.put(
        DocumentRecord(
            partition_key=_DOCUMENT_PARTITION,
            row_key=str(legacy_binding.execution_id),
            data={"correlation": encode_correlation_record(legacy_record).decode("utf-8")},
        ),
    )
    backfill_correlation_document_query_index(doc, batch_size=10)
    query_store = DocumentStoreDelegatedInvocationCorrelationQueryStore(
        doc,
        cursor_secret=_CURSOR_SECRET,
    )
    page = DelegatedExecutionQueryService(query_store).query_delegated_executions(
        DelegatedInvocationCorrelationQuery(page_size=10),
    )
    assert len(page.items) == 2


def test_c1_t15_run_id_filter_bounded() -> None:
    doc = _InstrumentedDocumentStore(cursor_secret=_CURSOR_SECRET)
    write = DocumentStoreDelegatedInvocationCorrelationStore(doc)
    correlation = DelegatedInvocationCorrelationService(write)
    target_run = mint_run_id()
    for index in range(30):
        binding = _binding(run_id=target_run if index == 29 else None, invocation_id=f"r-{index}")
        correlation.persist_binding(binding, persisted_at=_T0 + timedelta(seconds=index))
    query_store = DocumentStoreDelegatedInvocationCorrelationQueryStore(
        doc,
        cursor_secret=_CURSOR_SECRET,
    )
    page = DelegatedExecutionQueryService(query_store).query_delegated_executions(
        DelegatedInvocationCorrelationQuery(run_id=target_run, page_size=30),
    )
    assert len(page.items) == 1
    assert doc.query_call_count == 1


def test_c1_backfill_restores_query_index() -> None:
    doc = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    binding = _binding()
    record = DelegatedInvocationCorrelationRecord(binding=binding, persisted_at=_T0)
    doc.put(
        DocumentRecord(
            partition_key=_DOCUMENT_PARTITION,
            row_key=str(binding.execution_id),
            data={"correlation": encode_correlation_record(record).decode("utf-8")},
        ),
    )
    updated = backfill_correlation_document_query_index(doc, batch_size=10)
    assert updated == 1
    stored = doc.get(_DOCUMENT_PARTITION, str(binding.execution_id))
    assert stored is not None
    assert "query_run_id" in stored.data
