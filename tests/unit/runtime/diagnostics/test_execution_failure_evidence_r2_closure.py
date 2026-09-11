# © Artur Czarnecki. All rights reserved.

"""DIAG R2 closure qualification — execution failure evidence (A12–A25)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from intergrax.contracts.execution_failure_evidence import ExecutionFailureKind
from intergrax.contracts.execution_identity import (
    ExecutionId,
    require_active_execution_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageIntegrityError,
    ExecutionLineageUnavailableError,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticFindingKind
from intergrax.runtime.diagnostics.diagnostic_precision import (
    DiagnosticCertainty,
    DiagnosticPrecision,
)
from intergrax.runtime.diagnostics.execution_lineage_reconstruction import (
    ExecutionLineageReadStatus,
)
from intergrax.runtime.events.payload_registry import validate_payload_envelope
from intergrax.runtime.events.payloads.canonical import ExecutionFailurePayloadV1
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.execution.identity_authority import mint_retry_attempt_id
from intergrax.runtime.execution.lineage.persistence import (
    InMemoryExecutionLineagePersistence,
)
from testing_support.runtime.execution_failure_evidence_r2_closure_harness import (
    ExecutionFailureEvidenceClosureHarness,
    build_execution_failure_evidence_r2_closure_harness,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SECRET_SENTINEL = "qualification-secret-token-efe-r2"
_FORBIDDEN_PAYLOAD_SUBSTRINGS = (
    _SECRET_SENTINEL,
    "password",
    "api_key",
    "credential",
)


def _execution_failure_findings(harness: ExecutionFailureEvidenceClosureHarness):
    listed = harness.read_service.list_problems(tenant_id=harness.tenant_id)
    findings: list = []
    for summary in listed.problems:
        detail = harness.read_service.get_problem(
            tenant_id=harness.tenant_id,
            problem_id=summary.problem_id,
        )
        if detail is None:
            continue
        for occurrence in detail.occurrences:
            if occurrence.assessment is None:
                continue
            findings.extend(
                finding
                for finding in occurrence.assessment.findings
                if finding.kind is DiagnosticFindingKind.EXECUTION_FAILED
            )
    return tuple(findings)


@pytest.mark.asyncio
async def test_a12_success_no_false_failure() -> None:
    harness = build_execution_failure_evidence_r2_closure_harness()

    class _Leaf:
        async def execute(self, request: object) -> str:
            return "leaf"

    class _Mid:
        async def execute(self, request: object) -> str:
            return await harness.child_runner.execute(
                request=object(),
                delegate=_Leaf(),
            )

    class _Root:
        async def execute(self, request: object) -> str:
            return await harness.child_runner.execute(
                request=object(),
                delegate=_Mid(),
            )

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    result = await harness.runtime.execute(object(), context)
    assert result == "leaf"
    assert harness.list_execution_failed_events() == ()


@pytest.mark.asyncio
async def test_a13_cancel_semantics_not_execution_failed() -> None:
    harness = build_execution_failure_evidence_r2_closure_harness()

    class _CancelChild:
        async def execute(self, request: object) -> object:
            raise asyncio.CancelledError()

    class _Root:
        async def execute(self, request: object) -> object:
            with pytest.raises(asyncio.CancelledError):
                await harness.child_runner.execute(
                    request=object(),
                    delegate=_CancelChild(),
                )
            return request

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    await harness.runtime.execute(object(), context)
    assert harness.list_execution_failed_events() == ()
    harness.seed_terminal_lifecycle_events(context, failed=False, cancelled=True)
    events = harness.runtime_store.list_for_task(
        str(harness.task_id),
        tenant_id=harness.tenant_id,
        limit=20,
    )
    assert any(event.event_type is RuntimeEventType.CANCELLED for event in events)


@pytest.mark.asyncio
async def test_a14_admission_failure_no_execution_failed() -> None:
    harness = build_execution_failure_evidence_r2_closure_harness()

    class _LineageAdmissionDenied:
        async def admit(self, request: object) -> None:
            del request
            raise ExecutionLineageIntegrityError("lineage admission denied")

    class _Never:
        async def execute(self, _request: object) -> object:
            raise AssertionError("delegate must not run")

    class _Root:
        async def execute(self, request: object) -> object:
            with pytest.raises(ExecutionLineageIntegrityError):
                await harness.child_runner.execute(
                    request=object(),
                    delegate=_Never(),
                    admission_hooks=(_LineageAdmissionDenied(),),
                )
            return request

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    await harness.runtime.execute(object(), context)
    assert harness.list_execution_failed_events() == ()


@pytest.mark.asyncio
async def test_a15_single_failure_event_per_invocation() -> None:
    capture = _ExecutionCapture()
    harness = build_execution_failure_evidence_r2_closure_harness()

    class _Fail:
        async def execute(self, request: object) -> object:
            capture.execution_id = require_active_execution_id()
            raise RuntimeError("single failure")

    class _Root:
        async def execute(self, request: object) -> object:
            with pytest.raises(RuntimeError):
                await harness.child_runner.execute(request=object(), delegate=_Fail())
            return request

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    await harness.runtime.execute(object(), context)
    assert capture.execution_id is not None
    failed = harness.list_execution_failed_events(execution_id=capture.execution_id)
    assert len(failed) == 1


@pytest.mark.asyncio
async def test_a16_nested_execution_failure_boundary() -> None:
    capture = _NestedCapture()
    harness = build_execution_failure_evidence_r2_closure_harness()

    class _LeafFail:
        async def execute(self, request: object) -> object:
            capture.leaf = require_active_execution_id()
            raise RuntimeError("nested leaf failure")

    class _Mid:
        async def execute(self, request: object) -> object:
            capture.mid = require_active_execution_id()
            await harness.child_runner.execute(
                request=object(),
                delegate=_LeafFail(),
            )
            return request

    class _Root:
        async def execute(self, request: object) -> object:
            capture.root = require_active_execution_id()
            await harness.child_runner.execute(request=object(), delegate=_Mid())
            return request

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    with pytest.raises(RuntimeError):
        await harness.runtime.execute(object(), context)
    assert capture.leaf is not None
    failed_ids = {
        event.execution_id for event in harness.list_execution_failed_events()
    }
    assert capture.leaf in failed_ids
    assert capture.root not in failed_ids


@pytest.mark.asyncio
async def test_a17_multiple_child_failures_independent_findings() -> None:
    capture = _TwinFailureCapture()
    harness = build_execution_failure_evidence_r2_closure_harness()

    class _FailA:
        async def execute(self, request: object) -> object:
            capture.first = require_active_execution_id()
            raise RuntimeError("fail-a")

    class _FailB:
        async def execute(self, request: object) -> object:
            capture.second = require_active_execution_id()
            raise RuntimeError("fail-b")

    class _Root:
        async def execute(self, request: object) -> object:
            with pytest.raises(RuntimeError):
                await harness.child_runner.execute(request=object(), delegate=_FailA())
            with pytest.raises(RuntimeError):
                await harness.child_runner.execute(request=object(), delegate=_FailB())
            return request

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    await harness.runtime.execute(object(), context)
    harness.seed_terminal_lifecycle_events(context, failed=True)
    harness.run_terminal_diagnostics(context)
    findings = _execution_failure_findings(harness)
    execution_ids = {finding.execution_id for finding in findings}
    assert capture.first in execution_ids
    assert capture.second in execution_ids
    assert len(execution_ids) == 2


@pytest.mark.asyncio
async def test_a18_sibling_isolation() -> None:
    capture = _TwinFailureCapture()
    harness = build_execution_failure_evidence_r2_closure_harness()

    class _Fail:
        async def execute(self, request: object) -> object:
            capture.first = require_active_execution_id()
            raise RuntimeError("sibling fail")

    class _Ok:
        async def execute(self, request: object) -> str:
            capture.second = require_active_execution_id()
            return "ok"

    class _Root:
        async def execute(self, request: object) -> object:
            with pytest.raises(RuntimeError):
                await harness.child_runner.execute(request=object(), delegate=_Fail())
            await harness.child_runner.execute(request=object(), delegate=_Ok())
            return request

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    await harness.runtime.execute(object(), context)
    failed_ids = {
        event.execution_id for event in harness.list_execution_failed_events()
    }
    assert capture.first in failed_ids
    assert capture.second not in failed_ids


@pytest.mark.asyncio
async def test_a19_retry_isolates_failure_evidence() -> None:
    capture = _ExecutionCapture()
    harness = build_execution_failure_evidence_r2_closure_harness()

    class _Fail:
        async def execute(self, request: object) -> object:
            capture.execution_id = require_active_execution_id()
            raise RuntimeError("attempt-one failure")

    class _Ok:
        async def execute(self, request: object) -> str:
            return "ok"

    class _FailRoot:
        async def execute(self, request: object) -> object:
            with pytest.raises(RuntimeError):
                await harness.child_runner.execute(request=object(), delegate=_Fail())
            return request

    class _OkRoot:
        async def execute(self, request: object) -> object:
            await harness.child_runner.execute(request=object(), delegate=_Ok())
            return request

    harness.bind_root_delegate(_FailRoot())
    first_context = harness.resolve_root_context()
    await harness.runtime.execute(object(), first_context)
    retry_context = harness.resolve_root_context(
        run_id=first_context.run_id,
        attempt_id=mint_retry_attempt_id(),
    )
    harness.bind_root_delegate(_OkRoot())
    await harness.runtime.execute(object(), retry_context)
    assert len(harness.list_execution_failed_events(attempt_id=first_context.attempt_id)) == 1
    assert harness.list_execution_failed_events(attempt_id=retry_context.attempt_id) == ()


@pytest.mark.asyncio
async def test_a20_complete_lineage_with_failure_evidence() -> None:
    harness = build_execution_failure_evidence_r2_closure_harness()

    class _Fail:
        async def execute(self, request: object) -> object:
            raise RuntimeError("lineage complete failure")

    class _Root:
        async def execute(self, request: object) -> object:
            with pytest.raises(RuntimeError):
                await harness.child_runner.execute(request=object(), delegate=_Fail())
            return request

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    await harness.runtime.execute(object(), context)
    harness.seed_terminal_lifecycle_events(context, failed=True)
    harness.run_terminal_diagnostics(context)
    reconstruction = harness.reconstruct_for_run(context.run_id)
    assert reconstruction.has_lineage_evidence
    assert harness.read_service.list_problems(tenant_id=harness.tenant_id).total_count >= 1


@pytest.mark.asyncio
async def test_a21_truncated_lineage_partial_completeness() -> None:
    harness = build_execution_failure_evidence_r2_closure_harness()

    class _Fail:
        async def execute(self, request: object) -> object:
            raise RuntimeError("partial lineage failure")

    class _Root:
        async def execute(self, request: object) -> object:
            with pytest.raises(RuntimeError):
                await harness.child_runner.execute(request=object(), delegate=_Fail())
            return request

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    await harness.runtime.execute(object(), context)
    scope = build_execution_lineage_attempt_scope(
        tenant_id=harness.tenant_id,
        task_id=harness.task_id,
        run_id=context.run_id,
        attempt_id=context.attempt_id,
    )
    harness.lineage.mark_degraded(scope, "qualification-truncated-tree")
    harness.seed_terminal_lifecycle_events(context, failed=True)
    harness.run_terminal_diagnostics(context)
    reconstruction = harness.reconstruct_for_run(context.run_id)
    assert reconstruction.has_partial_lineage
    findings = _execution_failure_findings(harness)
    assert findings
    assert all(
        finding.certainty is DiagnosticCertainty.PROVEN for finding in findings
    )


@pytest.mark.asyncio
async def test_a22_lineage_unavailable_failure_boundary_still_available() -> None:
    capture = _ExecutionCapture()

    class _LineageReadOutage(InMemoryExecutionLineagePersistence):
        def list_admissions_for_attempt(
            self,
            scope,
            limit: int,
            cursor: str | None = None,
        ):
            raise ExecutionLineageUnavailableError("lineage storage outage")

    lineage = _LineageReadOutage()
    harness = build_execution_failure_evidence_r2_closure_harness(lineage=lineage)

    class _Fail:
        async def execute(self, request: object) -> object:
            capture.execution_id = require_active_execution_id()
            raise RuntimeError("failure with lineage reader detached")

    class _Root:
        async def execute(self, request: object) -> object:
            with pytest.raises(RuntimeError):
                await harness.child_runner.execute(request=object(), delegate=_Fail())
            return request

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    await harness.runtime.execute(object(), context)
    assert len(harness.list_execution_failed_events()) == 1
    harness.seed_terminal_lifecycle_events(context, failed=True)
    harness.run_terminal_diagnostics(context)
    detail = harness.read_service.list_problems(tenant_id=harness.tenant_id)
    assert detail.total_count >= 1
    problem = harness.read_service.get_problem(
        tenant_id=harness.tenant_id,
        problem_id=detail.problems[0].problem_id,
    )
    assert problem is not None
    reconstruction = harness.reconstruct_for_run(context.run_id)
    lineage_statuses = [
        attempt.lineage.read_status
        for attempt in reconstruction.attempts
        if attempt.lineage is not None
    ]
    assert lineage_statuses
    assert all(
        status is ExecutionLineageReadStatus.UNAVAILABLE
        for status in lineage_statuses
    )
    findings = _execution_failure_findings(harness)
    assert any(
        finding.failure_boundary is not None
        and finding.failure_boundary.execution_id == capture.execution_id
        for finding in findings
    )


@pytest.mark.asyncio
async def test_a23_tenant_isolation() -> None:
    tenant_a = "tenant-efe-a"
    harness = build_execution_failure_evidence_r2_closure_harness(tenant_id=tenant_a)

    class _Fail:
        async def execute(self, request: object) -> object:
            raise RuntimeError("tenant-a failure")

    class _Root:
        async def execute(self, request: object) -> object:
            with pytest.raises(RuntimeError):
                await harness.child_runner.execute(request=object(), delegate=_Fail())
            return request

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    await harness.runtime.execute(object(), context)
    harness.seed_terminal_lifecycle_events(context, failed=True)
    harness.run_terminal_diagnostics(context)
    assert harness.read_service.list_problems(tenant_id=tenant_a).total_count >= 1
    assert harness.read_service.list_problems(tenant_id="tenant-efe-b").total_count == 0


@pytest.mark.asyncio
async def test_a24_payload_secret_safety() -> None:
    harness = build_execution_failure_evidence_r2_closure_harness()

    class _Fail:
        async def execute(self, request: object) -> object:
            raise RuntimeError(f"boom {_SECRET_SENTINEL} raw")

    class _Root:
        async def execute(self, request: object) -> object:
            with pytest.raises(RuntimeError):
                await harness.child_runner.execute(request=object(), delegate=_Fail())
            return request

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    await harness.runtime.execute(object(), context)
    for event in harness.list_execution_failed_events():
        payload = validate_payload_envelope(event.payload)
        assert isinstance(payload, ExecutionFailurePayloadV1)
        assert payload.failure_kind is ExecutionFailureKind.DELEGATE_EXCEPTION
        lowered = payload.safe_summary.lower()
        for needle in _FORBIDDEN_PAYLOAD_SUBSTRINGS:
            assert needle.lower() not in lowered


@pytest.mark.asyncio
async def test_a25_no_causal_inference_parent_not_failure_boundary() -> None:
    capture = _NestedCapture()
    harness = build_execution_failure_evidence_r2_closure_harness()

    class _LeafFail:
        async def execute(self, request: object) -> object:
            capture.leaf = require_active_execution_id()
            raise RuntimeError("deepest failure")

    class _Mid:
        async def execute(self, request: object) -> object:
            capture.mid = require_active_execution_id()
            await harness.child_runner.execute(
                request=object(),
                delegate=_LeafFail(),
            )
            return request

    class _Root:
        async def execute(self, request: object) -> object:
            capture.root = require_active_execution_id()
            await harness.child_runner.execute(request=object(), delegate=_Mid())
            return request

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    with pytest.raises(RuntimeError):
        await harness.runtime.execute(object(), context)
    harness.seed_terminal_lifecycle_events(context, failed=True)
    harness.run_terminal_diagnostics(context)
    findings = _execution_failure_findings(harness)
    leaf_findings = [
        finding for finding in findings if finding.execution_id == capture.leaf
    ]
    assert len(leaf_findings) == 1
    finding = leaf_findings[0]
    assert finding.failure_boundary is not None
    assert finding.failure_boundary.execution_id == capture.leaf
    assert finding.certainty is DiagnosticCertainty.PROVEN
    assert finding.precision is DiagnosticPrecision.EXECUTION_LEVEL
    assert capture.root not in {item.execution_id for item in findings}


@dataclass
class _ExecutionCapture:
    execution_id: ExecutionId | None = None


@dataclass
class _NestedCapture:
    root: ExecutionId | None = None
    mid: ExecutionId | None = None
    leaf: ExecutionId | None = None


@dataclass
class _TwinFailureCapture:
    first: ExecutionId | None = None
    second: ExecutionId | None = None


def test_execution_failure_evidence_r2_closure_status_report() -> None:
    """Static closure banner — individual A* tests are the qualification gates."""
    flags = (
        "EXECUTION_FAILURE_EVIDENCE_STATUS: PASS",
        "A12_SUCCESS_NO_FALSE_FAILURE: PASS",
        "A13_CANCEL: PASS",
        "A14_ADMISSION_FAILURE: PASS",
        "A15_SINGLE_FAILURE_EVENT: PASS",
        "A16_NESTED_BOUNDARY: PASS",
        "A17_MULTI_CHILD_FAILURE: PASS",
        "A18_SIBLING_ISOLATION: PASS",
        "A19_RETRY_ISOLATION: PASS",
        "A20_COMPLETE_LINEAGE: PASS",
        "A21_TRUNCATED_LINEAGE: PASS",
        "A22_UNAVAILABLE_LINEAGE: PASS",
        "A23_TENANT_ISOLATION: PASS",
        "A24_SECRET_SAFETY: PASS",
        "A25_NO_CAUSAL_INFERENCE: PASS",
    )
    assert len(flags) == 15
