#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Adaptive harness signal trends report (Phase W-ADAPT-1.12)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from intergrax.runtime.adaptive.proposal_store import SQLiteProposalStore, default_proposal_store_path
from intergrax.runtime.adaptive.signal_store import SQLiteSignalStore, default_signal_store_path
from intergrax.runtime.adaptive.pattern_skill_stub import write_skill_stub_draft
from intergrax.runtime.adaptive.process_pattern_miner import (
    ProcessPatternMiner,
    ProcessPatternMinerConfig,
    ProcessPatternMinerResult,
)
from intergrax.runtime.adaptive.trace_sequence_reader import PersistedTraceSequenceReader
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from intergrax.runtime.adaptive.contracts import ProcessPatternProposal
from intergrax.runtime.adaptive.verification_loop import VerificationLoop
from intergrax.runtime.adaptive.verification_models import VerificationContext, VerificationReport


class UtilityHistogramBucket(BaseModel):
    bucket: str
    count: int


class SignalTrendReport(BaseModel):
    schema_version: str = "1.0.0"
    signal_count: int = 0
    tenant_ids: list[str] = Field(default_factory=list)
    average_utility: float | None = None
    utility_histogram: list[UtilityHistogramBucket] = Field(default_factory=list)
    eval_mode_counts: dict[str, int] = Field(default_factory=dict)
    regression_flag_counts: dict[str, int] = Field(default_factory=dict)


class ProposalGateSummary(BaseModel):
    proposal_id: str
    loop_id: str
    source_engine: str
    passed_all_gates: bool
    gate_reasons: list[str] = Field(default_factory=list)


class ProposalTrendReport(BaseModel):
    schema_version: str = "1.0.0"
    run_count: int = 0
    proposal_count: int = 0
    passed_gate_count: int = 0
    failed_gate_count: int = 0
    tenant_ids: list[str] = Field(default_factory=list)
    proposals: list[ProposalGateSummary] = Field(default_factory=list)


class ProcessPatternReport(BaseModel):
    schema_version: str = "1.0.0"
    scanned_run_count: int = 0
    pattern_count: int = 0
    patterns: list[ProcessPatternProposal] = Field(default_factory=list)
    skill_stub_paths: list[str] = Field(default_factory=list)


def _utility_bucket(utility: float) -> str:
    if utility < 0.0:
        return "negative"
    if utility < 0.25:
        return "0.00-0.24"
    if utility < 0.50:
        return "0.25-0.49"
    if utility < 0.75:
        return "0.50-0.74"
    return "0.75-1.00"


def build_signal_trend_report(store: SQLiteSignalStore, *, limit: int = 500) -> SignalTrendReport:
    signals = store.list_signals(limit=limit)
    utilities = [item.utility for item in signals if item.utility is not None]
    histogram = Counter(_utility_bucket(value) for value in utilities)
    eval_modes = Counter(item.eval_mode.value for item in signals)
    regression_flags = Counter(
        flag for item in signals for flag in item.regression_flags
    )
    tenant_ids = sorted({item.tenant_id for item in signals})
    average_utility = sum(utilities) / len(utilities) if utilities else None
    return SignalTrendReport(
        signal_count=len(signals),
        tenant_ids=tenant_ids,
        average_utility=average_utility,
        utility_histogram=[
            UtilityHistogramBucket(bucket=bucket, count=histogram[bucket])
            for bucket in sorted(histogram)
        ],
        eval_mode_counts=dict(eval_modes),
        regression_flag_counts=dict(regression_flags),
    )


def build_proposal_report(store: SQLiteProposalStore, *, limit: int = 100) -> ProposalTrendReport:
    runs = store.list_runs(limit=limit)
    summaries: list[ProposalGateSummary] = []
    passed = 0
    failed = 0
    for run in runs:
        for package in run.packages:
            if package.passed_all_gates:
                passed += 1
            else:
                failed += 1
            summaries.append(
                ProposalGateSummary(
                    proposal_id=package.proposal_id,
                    loop_id=package.candidate.loop_id,
                    source_engine=package.candidate.source_engine,
                    passed_all_gates=package.passed_all_gates,
                    gate_reasons=package.gate_reasons,
                )
            )
    tenant_ids = sorted({run.tenant_id for run in runs})
    return ProposalTrendReport(
        run_count=len(runs),
        proposal_count=len(summaries),
        passed_gate_count=passed,
        failed_gate_count=failed,
        tenant_ids=tenant_ids,
        proposals=summaries,
    )


def build_verification_report(
    *,
    signal_store: SQLiteSignalStore,
    profile_store_path: Path | None = None,
) -> VerificationReport:
    from intergrax.runtime.adaptive.profile_lifecycle import ProfileVersionLifecycleManager
    from intergrax.runtime.adaptive.profile_version_store import SQLiteProfileVersionStore

    profile_store = SQLiteProfileVersionStore(db_path=profile_store_path)
    verification_loop = VerificationLoop(
        signal_store=signal_store,
        profile_store=profile_store,
    )
    return verification_loop.verify_active_profiles(
        context=VerificationContext(auto_rollback_enabled=False),
    )


def build_process_pattern_report(
    *,
    miner_result: ProcessPatternMinerResult,
    skill_stub_output_dir: Path | None = None,
) -> ProcessPatternReport:
    stub_paths: list[str] = []
    if skill_stub_output_dir is not None:
        for proposal in miner_result.proposals:
            written = write_skill_stub_draft(proposal, output_dir=skill_stub_output_dir)
            if written is not None:
                stub_paths.append(written.as_posix())
    return ProcessPatternReport(
        scanned_run_count=miner_result.scanned_run_count,
        pattern_count=len(miner_result.proposals),
        patterns=miner_result.proposals,
        skill_stub_paths=stub_paths,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export adaptive harness signal and proposal trends.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite signal store path (default: build/adaptive_harness/signals.db)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "build" / "adaptive_harness" / "signal_trends.json",
        help="Signal trends output JSON path",
    )
    parser.add_argument(
        "--proposals-db-path",
        type=Path,
        default=None,
        help="SQLite proposal store path (default: build/adaptive_harness/proposals.db)",
    )
    parser.add_argument(
        "--proposals-output",
        type=Path,
        default=REPO_ROOT / "build" / "adaptive_harness" / "proposals.json",
        help="Proposal trends output JSON path",
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--skip-signals",
        action="store_true",
        help="Skip signal trend export (proposal-only mode)",
    )
    parser.add_argument(
        "--skip-proposals",
        action="store_true",
        help="Skip proposal trend export (signal-only mode)",
    )
    parser.add_argument(
        "--verification-output",
        type=Path,
        default=None,
        help="Optional verification report JSON path",
    )
    parser.add_argument(
        "--patterns-output",
        type=Path,
        default=None,
        help="Optional process pattern report JSON path",
    )
    parser.add_argument(
        "--trace-db-path",
        type=Path,
        default=REPO_ROOT / "build" / "intergrax_trace.db",
        help="Trace DB path for pattern mining",
    )
    parser.add_argument(
        "--patterns-tenant-id",
        type=str,
        default="lab",
        help="Tenant id for pattern mining",
    )
    parser.add_argument(
        "--skill-stub-output-dir",
        type=Path,
        default=REPO_ROOT / "build" / "adaptive_harness" / "skill_stubs",
        help="Optional skill stub draft output directory",
    )
    args = parser.parse_args()

    if not args.skip_signals:
        db_path = args.db_path or default_signal_store_path(REPO_ROOT)
        store = SQLiteSignalStore(db_path=db_path)
        report = build_signal_trend_report(store, limit=args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"adaptive signal trends written: {args.output}")
        print(f"signals={report.signal_count} average_utility={report.average_utility}")

    if not args.skip_proposals:
        proposals_db_path = args.proposals_db_path or default_proposal_store_path(REPO_ROOT)
        proposal_store = SQLiteProposalStore(db_path=proposals_db_path)
        proposal_report = build_proposal_report(proposal_store, limit=args.limit)
        args.proposals_output.parent.mkdir(parents=True, exist_ok=True)
        args.proposals_output.write_text(
            proposal_report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        print(f"adaptive proposal trends written: {args.proposals_output}")
        print(
            f"runs={proposal_report.run_count} proposals={proposal_report.proposal_count} "
            f"passed={proposal_report.passed_gate_count} failed={proposal_report.failed_gate_count}"
        )

    if args.verification_output is not None:
        db_path = args.db_path or default_signal_store_path(REPO_ROOT)
        store = SQLiteSignalStore(db_path=db_path)
        verification_report = build_verification_report(signal_store=store)
        args.verification_output.parent.mkdir(parents=True, exist_ok=True)
        args.verification_output.write_text(
            verification_report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        print(f"adaptive verification report written: {args.verification_output}")
        print(f"verification_passed={verification_report.passed}")

    if args.patterns_output is not None:
        if not args.trace_db_path.is_file():
            print(f"pattern mining skipped: trace db not found at {args.trace_db_path}")
            return 0
        signal_db_path = args.db_path or default_signal_store_path(REPO_ROOT)
        trace_reader = SQLiteRunTraceStore(db_path=args.trace_db_path)
        sequence_reader = PersistedTraceSequenceReader(
            trace_reader,
            signal_store=SQLiteSignalStore(db_path=signal_db_path),
        )
        miner = ProcessPatternMiner(sequence_reader=sequence_reader)
        miner_result = miner.mine(
            tenant_id=args.patterns_tenant_id,
            config=ProcessPatternMinerConfig(min_support=2),
        )
        pattern_report = build_process_pattern_report(
            miner_result=miner_result,
            skill_stub_output_dir=args.skill_stub_output_dir,
        )
        args.patterns_output.parent.mkdir(parents=True, exist_ok=True)
        args.patterns_output.write_text(pattern_report.model_dump_json(indent=2), encoding="utf-8")
        print(f"adaptive process pattern report written: {args.patterns_output}")
        print(
            f"patterns={pattern_report.pattern_count} scanned_runs={pattern_report.scanned_run_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
