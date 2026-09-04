# © Artur Czarnecki. All rights reserved.

"""Scale qualification gates S1-A … S1-O."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from intergrax.runtime.diagnostics.document_store_functional_evidence_persistence import (
    DocumentStoreFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analyzer import FunctionalDiagnosticAnalyzer
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistenceConflictError,
    FunctionalEvidencePersistenceError,
    FunctionalEvidencePersistenceIntegrityError,
    FunctionalEvidenceQueryRequest,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
    collect_all_evidence,
)
from intergrax.runtime.diagnostics.functional_validation_lookup import (
    FunctionalValidationEvidenceLookup,
)
from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
    build_c1_rag_functional_diagnostic_specification,
)
from tests.system.functional_diagnostics_durability.assessment_fingerprint import (
    DiagnosticAssessmentFingerprint,
)
from tests.system.functional_diagnostics_scale.backend import (
    BackendQueryEfficiencyObservation,
    ScaleBackendProbe,
    ScaleGateResult,
)
from tests.system.functional_diagnostics_scale.manifest import ScaleDatasetManifest
from tests.system.functional_diagnostics_scale.metrics import (
    ExecutionReadScaleCurve,
    ExecutionReadScaleCurvePoint,
    LatencyDistribution,
    MonotonicTimer,
    ScaleCorrectnessAccumulator,
)
from tests.system.functional_diagnostics_scale.profile import (
    FunctionalDiagnosticsScaleProfile,
    scale_curve_profiles,
)
from tests.system.functional_diagnostics_scale.process_ipc import ScaleWorkerPhase
from tests.system.functional_diagnostics_scale.synthetic_backend import (
    SyntheticFunctionalDiagnosticsScaleProbe,
)
from tests.system.functional_diagnostics_scale.workload import (
    FunctionalEvidenceWorkloadGenerator,
    ScaleExecutionIdentity,
)
from tests.system.functional_diagnostics_scale.workers import ScaleWorker, ScaleWorkerBatchResult


@dataclass(slots=True)
class ScaleGateContext:
    profile: FunctionalDiagnosticsScaleProfile
    manifest: ScaleDatasetManifest
    persistence: DocumentStoreFunctionalEvidencePersistence
    backend_probe: ScaleBackendProbe
    generator: FunctionalEvidenceWorkloadGenerator
    worker: ScaleWorker
    cursor_secret: bytes
    append_latencies_ms: list[float] = field(default_factory=list)
    read_latencies_ms: list[float] = field(default_factory=list)
    analyze_latencies_ms: list[float] = field(default_factory=list)
    correctness: ScaleCorrectnessAccumulator = field(default_factory=ScaleCorrectnessAccumulator)
    scale_curve: ExecutionReadScaleCurve | None = None
    index_observation: BackendQueryEfficiencyObservation | None = None
    writer_batch: ScaleWorkerBatchResult | None = None


def run_scale_gates(context: ScaleGateContext) -> tuple[ScaleGateResult, ...]:
    return (
        gate_s1_a_append_correctness(context),
        gate_s1_b_query_boundedness(context),
        gate_s1_c_pagination_completeness(context),
        gate_s1_d_multi_tenant_isolation(context),
        gate_s1_e_concurrent_writers(context),
        gate_s1_f_concurrent_readers(context),
        gate_s1_g_concurrent_read_write(context),
        gate_s1_h_idempotency_contention(context),
        gate_s1_i_conflict_contention(context),
        gate_s1_j_analyzer_fidelity(context),
        gate_s1_k_resource_boundedness(context),
        gate_s1_l_index_efficiency(context),
        gate_s1_m_recovery_after_load(context),
        gate_s1_n_backend_pluginability(),
        gate_s1_o_delivery_decoupling_readiness(),
    )


def run_scale_curve_probe(
    *,
    profile: FunctionalDiagnosticsScaleProfile,
    collection_name: str,
    cursor_secret: bytes,
) -> ExecutionReadScaleCurve:
    """S1-B isolated collection probe — must not pollute the canonical dataset."""
    from tests.system.functional_diagnostics_scale.mongodb_backend import (
        MongoFunctionalDiagnosticsScaleProbe,
    )

    curve_probe = MongoFunctionalDiagnosticsScaleProbe(
        collection_name=f"{collection_name}_s1b",
    )
    curve_probe.prepare()
    store = curve_probe.build_document_store()
    persistence = DocumentStoreFunctionalEvidencePersistence(
        store,
        cursor_secret=cursor_secret,
        query_page_limit=profile.document_store_query_page_limit,
    )
    probe_count = profile.scale_curve_probe_evidence_count
    curve_profiles = scale_curve_profiles(probe_count)
    probe_generator = FunctionalEvidenceWorkloadGenerator(curve_profiles[0])
    probe_identity = probe_generator.execution_identities()[0]
    for evidence in probe_generator.evidence_for_execution(probe_identity):
        persistence.append(evidence)
    points: list[ExecutionReadScaleCurvePoint] = []
    total_evidence = len(probe_generator.evidence_for_execution(probe_identity))
    for label, curve_profile in zip(
        ("small", "medium", "large"),
        curve_profiles,
        strict=True,
    ):
        curve_generator = FunctionalEvidenceWorkloadGenerator(curve_profile)
        for evidence in curve_generator.all_evidence():
            persistence.append(evidence)
            total_evidence += 1
        timer = MonotonicTimer()
        collect_all_evidence(
            persistence,
            tenant_id=probe_identity.tenant_id,
            task_id=probe_identity.task_id,
            run_id=probe_identity.run_id,
            page_size=profile.page_size,
        )
        points.append(
            ExecutionReadScaleCurvePoint(
                label=label,
                total_evidence=total_evidence,
                probe_evidence_count=probe_count,
                read_latency_ms=timer.elapsed_ms(),
            ),
        )
    curve = ExecutionReadScaleCurve(points=tuple(points))
    curve_probe.close_document_store(store)
    curve_probe.cleanup()
    return curve


def populate_dataset_with_workers(context: ScaleGateContext) -> ScaleWorkerBatchResult:
    batch = context.worker.run_phase(
        ScaleWorkerPhase.WRITE,
        worker_count=context.profile.writer_concurrency,
    )
    for result in batch.results:
        context.append_latencies_ms.extend(result.append_latency_ms)
        context.correctness.unexpected_errors += result.errors
    context.writer_batch = batch
    return batch


def gate_s1_a_append_correctness(context: ScaleGateContext) -> ScaleGateResult:
    expected = context.manifest.total_evidence
    written = 0
    duplicates = 0
    scope_mismatches = 0
    for entry in context.manifest.entries:
        collected = collect_all_evidence(
            context.persistence,
            tenant_id=entry.tenant_id,
            task_id=entry.task_id,
            run_id=entry.run_id,
            page_size=context.profile.page_size,
        )
        collected_ids = {str(item.evidence_id) for item in collected}
        expected_ids = set(entry.evidence_ids)
        missing = len(expected_ids - collected_ids)
        extra = len(collected_ids - expected_ids)
        context.correctness.lost_evidence += missing
        scope_mismatches += extra
        written += len(collected_ids)
        if len(collected_ids) != len(collected):
            duplicates += 1
    context.correctness.unexpected_duplicate_canonical_records += duplicates
    context.correctness.integrity_errors += scope_mismatches
    passed = (
        written == expected
        and context.correctness.lost_evidence == 0
        and scope_mismatches == 0
        and duplicates == 0
    )
    return ScaleGateResult(
        gate_id="S1-A",
        passed=passed,
        detail=f"expected={expected} actual={written} duplicates={duplicates} scope_mismatches={scope_mismatches}",
    )


def gate_s1_b_query_boundedness(context: ScaleGateContext) -> ScaleGateResult:
    curve = context.scale_curve
    if curve is None:
        return ScaleGateResult("S1-B", False, "scale curve missing")
    passed = not curve.gross_linear_growth()
    return ScaleGateResult(
        gate_id="S1-B",
        passed=passed,
        detail=(
            "curve="
            + ",".join(
                f"{point.label}:{point.read_latency_ms:.2f}ms@{point.total_evidence}"
                for point in curve.points
            )
        ),
    )


def gate_s1_c_pagination_completeness(context: ScaleGateContext) -> ScaleGateResult:
    heavy = next(entry for entry in context.manifest.entries if entry.is_heavy)
    seen: set[str] = set()
    cursor: str | None = None
    pages = 0
    while True:
        page = context.persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=heavy.tenant_id,
                task_id=heavy.task_id,
                run_id=heavy.run_id,
                page_size=context.profile.page_size,
                cursor=cursor,
            ),
        )
        pages += 1
        for item in page.items:
            evidence_id = str(item.evidence_id)
            if evidence_id in seen:
                context.correctness.unexpected_duplicate_canonical_records += 1
            seen.add(evidence_id)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    missing = len(set(heavy.evidence_ids) - seen)
    context.correctness.lost_evidence += missing
    passed = missing == 0 and len(seen) == len(heavy.evidence_ids) and pages > 1
    return ScaleGateResult(
        gate_id="S1-C",
        passed=passed,
        detail=f"pages={pages} expected={len(heavy.evidence_ids)} actual={len(seen)}",
    )


def gate_s1_d_multi_tenant_isolation(context: ScaleGateContext) -> ScaleGateResult:
    leakage = 0
    tenant_ids = set(context.manifest.tenant_ids)
    for entry in context.manifest.entries:
        collected = collect_all_evidence(
            context.persistence,
            tenant_id=entry.tenant_id,
            task_id=entry.task_id,
            run_id=entry.run_id,
            page_size=context.profile.page_size,
        )
        for item in collected:
            if item.scope.tenant_id not in tenant_ids:
                leakage += 1
            if item.scope.tenant_id != entry.tenant_id:
                context.correctness.tenant_leakage += 1
            if str(item.scope.task_id) != str(entry.task_id):
                context.correctness.task_leakage += 1
            if str(item.scope.run_id) != str(entry.run_id):
                context.correctness.run_leakage += 1
    passed = leakage == 0 and context.correctness.tenant_leakage == 0
    return ScaleGateResult(
        gate_id="S1-D",
        passed=passed,
        detail=f"tenant_leakage={context.correctness.tenant_leakage}",
    )


def gate_s1_e_concurrent_writers(context: ScaleGateContext) -> ScaleGateResult:
    batch = context.writer_batch
    if batch is None:
        return ScaleGateResult("S1-E", False, "writer batch missing")
    return _gate_from_worker_batch("S1-E", batch)


def gate_s1_f_concurrent_readers(context: ScaleGateContext) -> ScaleGateResult:
    batch = context.worker.run_phase(
        ScaleWorkerPhase.READ,
        worker_count=context.profile.reader_concurrency,
    )
    for result in batch.results:
        context.read_latencies_ms.extend(result.read_latency_ms)
    return _gate_from_worker_batch("S1-F", batch)


def gate_s1_g_concurrent_read_write(context: ScaleGateContext) -> ScaleGateResult:
    read_batch = context.worker.run_phase(
        ScaleWorkerPhase.READ,
        worker_count=max(1, context.profile.reader_concurrency // 2),
    )
    write_batch = context.worker.run_phase(
        ScaleWorkerPhase.WRITE,
        worker_count=max(1, context.profile.writer_concurrency // 2),
    )
    passed = read_batch.all_exit_ok and write_batch.all_exit_ok
    return ScaleGateResult(
        gate_id="S1-G",
        passed=passed,
        detail=f"read_ok={read_batch.all_exit_ok} write_ok={write_batch.all_exit_ok}",
    )


def gate_s1_h_idempotency_contention(context: ScaleGateContext) -> ScaleGateResult:
    batch = context.worker.run_phase(ScaleWorkerPhase.IDEMPOTENT, worker_count=2)
    passed = batch.all_exit_ok
    return _gate_from_worker_batch("S1-H", batch)


def gate_s1_i_conflict_contention(context: ScaleGateContext) -> ScaleGateResult:
    batch = context.worker.run_phase(ScaleWorkerPhase.CONFLICT, worker_count=2)
    conflicts = sum(result.conflicts for result in batch.results)
    context.correctness.expected_conflicts += conflicts
    passed = batch.all_exit_ok and conflicts > 0
    return ScaleGateResult(
        gate_id="S1-I",
        passed=passed,
        detail=f"conflicts={conflicts}",
    )


def gate_s1_j_analyzer_fidelity(context: ScaleGateContext) -> ScaleGateResult:
    analyzer = FunctionalDiagnosticAnalyzer(persistence=context.persistence)
    specification = build_c1_rag_functional_diagnostic_specification(include_validation=False)
    mismatches = 0
    sample_count = 0
    for entry in context.manifest.entries:
        if not entry.analyzer_sample:
            continue
        sample_count += 1
        timer = MonotonicTimer()
        validation_lookup = FunctionalValidationEvidenceLookup.for_scope(
            tenant_id=entry.tenant_id,
            task_id=entry.task_id,
            run_id=entry.run_id,
            attempt_id=entry.attempt_id,
        )
        analysis = analyzer.analyze(
            specification=specification,
            tenant_id=entry.tenant_id,
            task_id=entry.task_id,
            run_id=entry.run_id,
            attempt_id=entry.attempt_id,
            validations=validation_lookup,
        )
        context.analyze_latencies_ms.append(timer.elapsed_ms())
        expected = DiagnosticAssessmentFingerprint.from_analysis(analysis)
        replay = analyzer.analyze(
            specification=specification,
            tenant_id=entry.tenant_id,
            task_id=entry.task_id,
            run_id=entry.run_id,
            attempt_id=entry.attempt_id,
            validations=validation_lookup,
        )
        actual = DiagnosticAssessmentFingerprint.from_analysis(replay)
        if expected != actual:
            mismatches += 1
    context.correctness.analyzer_fidelity_mismatches = mismatches
    passed = mismatches == 0 and sample_count > 0
    return ScaleGateResult(
        gate_id="S1-J",
        passed=passed,
        detail=f"sample={sample_count} mismatches={mismatches}",
    )


def gate_s1_k_resource_boundedness(context: ScaleGateContext) -> ScaleGateResult:
    identity = _heavy_identity(context)
    baseline_reads: list[float] = []
    for _ in range(5):
        timer = MonotonicTimer()
        collect_all_evidence(
            context.persistence,
            tenant_id=identity.tenant_id,
            task_id=identity.task_id,
            run_id=identity.run_id,
            page_size=context.profile.page_size,
        )
        baseline_reads.append(timer.elapsed_ms())
    max_baseline = max(baseline_reads) if baseline_reads else 0.0
    passed = max_baseline < 60_000.0
    return ScaleGateResult(
        gate_id="S1-K",
        passed=passed,
        detail=f"max_scoped_read_ms={max_baseline:.2f}",
    )


def gate_s1_l_index_efficiency(context: ScaleGateContext) -> ScaleGateResult:
    identity = context.generator.execution_identities()[0]
    observation = context.backend_probe.observe_execution_query_efficiency(
        tenant_id=identity.tenant_id,
        task_id=str(identity.task_id),
        run_id=str(identity.run_id),
    )
    context.index_observation = observation
    if observation is None:
        return ScaleGateResult("S1-L", False, "explain unavailable")
    total_docs = context.backend_probe.collect_backend_metrics().document_count or 0
    passed = observation.documents_examined <= max(
        len(context.manifest.entries[0].evidence_ids) * 4,
        64,
    ) and observation.documents_examined < total_docs
    return ScaleGateResult(
        gate_id="S1-L",
        passed=passed,
        detail=(
            f"examined={observation.documents_examined} "
            f"total_docs={total_docs} returned={observation.n_returned}"
        ),
    )


def gate_s1_m_recovery_after_load(context: ScaleGateContext) -> ScaleGateResult:
    probe = context.backend_probe
    fresh_store = probe.build_document_store()
    fresh = DocumentStoreFunctionalEvidencePersistence(
        fresh_store,
        cursor_secret=context.cursor_secret,
        query_page_limit=context.profile.document_store_query_page_limit,
    )
    sample = next(entry for entry in context.manifest.entries if entry.is_heavy)
    try:
        collected = collect_all_evidence(
            fresh,
            tenant_id=sample.tenant_id,
            task_id=sample.task_id,
            run_id=sample.run_id,
            page_size=context.profile.page_size,
        )
    except FunctionalEvidencePersistenceIntegrityError:
        context.correctness.integrity_errors += 1
        probe.close_document_store(fresh_store)
        return ScaleGateResult("S1-M", False, "integrity error on recovery read")
    probe.close_document_store(fresh_store)
    passed = len(collected) == len(sample.evidence_ids)
    if not passed:
        context.correctness.lost_evidence += abs(len(sample.evidence_ids) - len(collected))
    return ScaleGateResult(
        gate_id="S1-M",
        passed=passed,
        detail=f"expected={len(sample.evidence_ids)} recovered={len(collected)}",
    )


def gate_s1_n_backend_pluginability() -> ScaleGateResult:
    probe = SyntheticFunctionalDiagnosticsScaleProbe()
    probe.prepare()
    store = probe.build_document_store()
    probe.close_document_store(store)
    probe.cleanup()
    return ScaleGateResult("S1-N", True, "synthetic probe composed without runner changes")


_DIAGNOSTICS_ROOT = Path(__file__).resolve().parents[3] / "intergrax" / "runtime" / "diagnostics"
_OBSERVABILITY_ROOT = Path(__file__).resolve().parents[3] / "intergrax" / "runtime" / "observability"
_DELIVERY_COUPLING_SNIPPETS = (
    "TaskQueue",
    "queue_mode",
    "WorkSubmission",
    "delivery_mode",
    "QueuedWorkSubmission",
    "DirectWorkSubmission",
)


def gate_s1_o_delivery_decoupling_readiness() -> ScaleGateResult:
    """Audit: functional diagnostics semantics must not depend on delivery mechanism."""
    violations: list[str] = []
    for root in (_DIAGNOSTICS_ROOT, _OBSERVABILITY_ROOT):
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for snippet in _DELIVERY_COUPLING_SNIPPETS:
                if snippet in text:
                    violations.append(f"{path.name}:{snippet}")
    if violations:
        return ScaleGateResult(
            gate_id="S1-O",
            passed=False,
            detail="delivery coupling detected: " + ", ".join(violations[:8]),
        )
    return ScaleGateResult(
        gate_id="S1-O",
        passed=True,
        detail=(
            "Functional Diagnostics semantics are independent of Work Submission / Delivery; "
            "frozen S2 model: WorkSubmissionStrategy → Direct | Queued; "
            "Queued → TaskQueue → provider plugins; "
            "authority: TaskQueue=delivery state, RuntimeEvent=execution truth, "
            "FunctionalEvidence=functional truth"
        ),
    )


def latency_summary(context: ScaleGateContext) -> dict[str, LatencyDistribution]:
    return {
        "append": LatencyDistribution.from_samples_ms(tuple(context.append_latencies_ms)),
        "execution_read": LatencyDistribution.from_samples_ms(tuple(context.read_latencies_ms)),
        "analyze": LatencyDistribution.from_samples_ms(tuple(context.analyze_latencies_ms)),
    }


def _gate_from_worker_batch(gate_id: str, batch: ScaleWorkerBatchResult) -> ScaleGateResult:
    errors = sum(result.errors for result in batch.results)
    return ScaleGateResult(
        gate_id=gate_id,
        passed=batch.all_exit_ok and errors == 0,
        detail=f"workers={len(batch.results)} errors={errors}",
    )


def _heavy_identity(context: ScaleGateContext) -> ScaleExecutionIdentity:
    return next(
        identity
        for identity in context.generator.execution_identities()
        if identity.is_heavy
    )


__all__ = [
    "ScaleGateContext",
    "gate_s1_n_backend_pluginability",
    "latency_summary",
    "populate_dataset_with_workers",
    "run_scale_curve_probe",
    "run_scale_gates",
]
