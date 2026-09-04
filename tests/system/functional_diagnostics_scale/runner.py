# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-SCALE-S1 enterprise qualification orchestrator."""

from __future__ import annotations

import argparse
import binascii
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from intergrax.runtime.diagnostics.document_store_functional_evidence_persistence import (
    DocumentStoreFunctionalEvidencePersistence,
)
from tests.system.functional_diagnostics_durability.runner import run_contract_qualification
from tests.system.functional_diagnostics_scale.gates import (
    ScaleGateContext,
    latency_summary,
    populate_dataset_with_workers,
    run_scale_curve_probe,
    run_scale_gates,
)
from tests.system.functional_diagnostics_scale.manifest import ScaleDatasetManifest
from tests.system.functional_diagnostics_scale.metrics import (
    MonotonicTimer,
    ScaleResourceMetrics,
    ThroughputMeasurement,
)
from tests.system.functional_diagnostics_scale.mongodb_backend import (
    MongoFunctionalDiagnosticsScaleProbe,
    mongodb_available,
)
from tests.system.functional_diagnostics_scale.profile import (
    FunctionalDiagnosticsScaleProfileName,
    canonical_s1_profile,
    resolve_scale_profile,
)
from tests.system.functional_diagnostics_scale.reporting import (
    ScaleQualificationReport,
    write_qualification_artifacts,
)
from tests.system.functional_diagnostics_scale.resource_probe import (
    cpu_core_count,
    process_rss_bytes,
)
from tests.system.functional_diagnostics_scale.workers import ScaleWorker
from tests.system.functional_diagnostics_scale.workload import (
    FunctionalEvidenceWorkloadGenerator,
)

_EXIT_OK = 0
_EXIT_FAILED = 1
_EXIT_BLOCKED = 2
_ARTIFACT_DIR = Path(
    os.environ.get(
        "DIAG_FUNCTIONAL_SCALE_S1_ARTIFACT_DIR",
        ".tmp/session/diag-functional-scale-s1",
    ),
)
_CURSOR_SECRET = b"diag-functional-scale-s1-secret-32-bytes!!"
_COLLECTION_PREFIX = "intergrax_diag_scale_s1_"
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class ScaleRunContext:
    collection_name: str
    artifact_dir: Path
    profile: object
    backend_probe: MongoFunctionalDiagnosticsScaleProbe


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def _blocked_report(
    *,
    start_head: str,
    blocker: str,
    profile: object | None = None,
) -> ScaleQualificationReport:
    resolved_profile = profile if profile is not None else canonical_s1_profile()
    manifest = FunctionalEvidenceWorkloadGenerator(resolved_profile).build_manifest()
    return ScaleQualificationReport(
        verdict="BLOCKED",
        blocker=blocker,
        start_head=start_head,
        final_head=start_head,
        profile=resolved_profile,
        manifest=manifest,
        gates=(),
        correctness=_empty_correctness(),
        latency={},
        throughput={},
        resources=_empty_resources(),
        scale_curve=None,
        backend_provider="mongodb",
        backend_document_store_type="",
        database_name="intergrax_diag_scale_s1",
        collection_name="",
        production_provider_factory_used=True,
        backend_mocked=False,
        backend_in_memory=False,
        first_canonical_run=True,
    )


def _empty_correctness():
    from tests.system.functional_diagnostics_scale.metrics import ScaleCorrectnessAccumulator

    return ScaleCorrectnessAccumulator().freeze()


def _empty_resources() -> ScaleResourceMetrics:
    return ScaleResourceMetrics(
        rss_before_bytes=None,
        rss_after_bytes=None,
        mongo_document_count=None,
        mongo_storage_size_bytes=None,
        cpu_core_count=cpu_core_count(),
    )


def run_scale_qualification(
    *,
    profile_name: FunctionalDiagnosticsScaleProfileName | str = FunctionalDiagnosticsScaleProfileName.STANDARD,
    artifact_dir: Path = _ARTIFACT_DIR,
) -> ScaleQualificationReport:
    start_head = _git_head()
    profile = resolve_scale_profile(profile_name)
    if not mongodb_available():
        report = _blocked_report(
            start_head=start_head,
            blocker="INTERGRAX_MONGODB_URI missing",
            profile=profile,
        )
        write_qualification_artifacts(artifact_dir=artifact_dir, profile=report.profile, report=report)
        return report

    try:
        import pymongo  # noqa: F401
    except ImportError:
        report = _blocked_report(
            start_head=start_head,
            blocker="pymongo not installed",
            profile=profile,
        )
        write_qualification_artifacts(artifact_dir=artifact_dir, profile=report.profile, report=report)
        return report

    write_qualification_artifacts(
        artifact_dir=artifact_dir,
        profile=profile,
        report=_blocked_report(
            start_head=start_head,
            blocker="profile-frozen",
            profile=profile,
        ),
    )

    run_suffix = uuid.uuid4().hex
    collection_name = f"{_COLLECTION_PREFIX}{run_suffix}"
    backend_probe = MongoFunctionalDiagnosticsScaleProbe(collection_name=collection_name)
    backend_probe.prepare()
    document_store = backend_probe.build_document_store()
    persistence = DocumentStoreFunctionalEvidencePersistence(
        document_store,
        cursor_secret=_CURSOR_SECRET,
        query_page_limit=profile.document_store_query_page_limit,
    )
    generator = FunctionalEvidenceWorkloadGenerator(profile)
    manifest = generator.build_manifest()
    worker = ScaleWorker(
        work_dir=artifact_dir / "workers",
        collection_name=collection_name,
        cursor_secret_hex=binascii.hexlify(_CURSOR_SECRET).decode("ascii"),
        profile=profile,
    )
    context = ScaleGateContext(
        profile=profile,
        manifest=manifest,
        persistence=persistence,
        backend_probe=backend_probe,
        generator=generator,
        worker=worker,
        cursor_secret=_CURSOR_SECRET,
    )

    rss_before = process_rss_bytes()
    write_timer = MonotonicTimer()

    context.scale_curve = run_scale_curve_probe(
        profile=profile,
        collection_name=collection_name,
        cursor_secret=_CURSOR_SECRET,
    )
    populate_dataset_with_workers(context)

    write_elapsed = write_timer.elapsed_seconds()
    gates = run_scale_gates(context)
    latency = latency_summary(context)
    backend_metrics = backend_probe.collect_backend_metrics()
    rss_after = process_rss_bytes()

    mandatory_gates = tuple(
        gate for gate in gates if gate.gate_id not in {"S1-N"}
    )
    correctness_ok = context.correctness.all_mandatory_zero()
    gates_ok = all(gate.passed for gate in mandatory_gates)
    verdict = "PASS" if gates_ok and correctness_ok else "FAILED"

    report = ScaleQualificationReport(
        verdict=verdict,
        blocker="" if verdict == "PASS" else "one or more mandatory S1 gates failed",
        start_head=start_head,
        final_head=_git_head(),
        profile=profile,
        manifest=manifest,
        gates=gates,
        correctness=context.correctness.freeze(),
        latency=latency,
        throughput={
            "append": ThroughputMeasurement(
                operation_count=manifest.total_evidence,
                elapsed_seconds=write_elapsed,
            ),
        },
        resources=ScaleResourceMetrics(
            rss_before_bytes=rss_before,
            rss_after_bytes=rss_after,
            mongo_document_count=backend_metrics.document_count,
            mongo_storage_size_bytes=backend_metrics.storage_size_bytes,
            cpu_core_count=cpu_core_count(),
        ),
        scale_curve=context.scale_curve,
        backend_provider=backend_probe.provider_id,
        backend_document_store_type=backend_probe.backend_identity().document_store_type,
        database_name=backend_probe.backend_identity().database_name,
        collection_name=collection_name,
        production_provider_factory_used=True,
        backend_mocked=False,
        backend_in_memory=False,
        first_canonical_run=True,
    )

    write_qualification_artifacts(artifact_dir=artifact_dir, profile=profile, report=report)
    backend_probe.close_document_store(document_store)
    backend_probe.cleanup()
    return report


def run_static_regressions() -> tuple[bool, str]:
    contract = run_contract_qualification()
    if contract.verdict != "PASS":
        return False, f"D1 contract regression failed: {contract.verdict}"
    return True, "D1 contract regression PASS"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DIAG-FUNCTIONAL-SCALE-S1 qualification")
    parser.add_argument(
        "--profile",
        default=FunctionalDiagnosticsScaleProfileName.STANDARD.value,
        choices=[item.value for item in FunctionalDiagnosticsScaleProfileName],
    )
    parser.add_argument("--artifact-dir", default=str(_ARTIFACT_DIR))
    parser.add_argument("--skip-static", action="store_true")
    args = parser.parse_args(argv)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "run.log"

    if not args.skip_static:
        static_ok, static_detail = run_static_regressions()
        log_path.write_text(static_detail + "\n", encoding="utf-8")
        if not static_ok:
            print(static_detail, file=sys.stderr)
            return _EXIT_FAILED

    report = run_scale_qualification(
        profile_name=args.profile,
        artifact_dir=artifact_dir,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report.to_json_dict(), indent=2))
        handle.write("\n")

    print(f"S1 verdict: {report.verdict}")
    if report.verdict == "BLOCKED":
        print(f"Blocker: {report.blocker}")
        return _EXIT_BLOCKED
    if report.verdict == "FAILED":
        for gate in report.gates:
            if not gate.passed:
                print(f"FAILED {gate.gate_id}: {gate.detail}")
        print("S1-R1 REQUIRED")
        return _EXIT_FAILED
    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
