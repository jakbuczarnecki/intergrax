"""CLI for the VPI full Data Pack storage load operator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapBatchSize,
    RelationalTargetId,
    VectorTargetId,
    VerificationMode,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.config import (
    OperatorRunMode,
    StorageLoadOperatorConfig,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.errors import (
    StorageLoadOperatorUsageError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.exit_codes import (
    StorageLoadOperatorExitCode,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator.runner import (
    run_storage_load_operator,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VPI full Data Pack storage load operator (PostgreSQL + Qdrant)",
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--relational-target", required=True)
    parser.add_argument("--vector-target", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--verification",
        choices=[mode.value.lower() for mode in VerificationMode],
        default=VerificationMode.STRICT.value.lower(),
        help="verification mode (production default: strict)",
    )
    parser.add_argument("--expected-record-count", type=int, default=None)
    parser.add_argument("--expected-content-identity", default=None)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--plan", action="store_true")
    mode_group.add_argument("--fresh", action="store_true")
    mode_group.add_argument("--resume", action="store_true")
    return parser


def _resolve_run_mode(args: argparse.Namespace) -> OperatorRunMode:
    if args.plan:
        return OperatorRunMode.PLAN
    if args.fresh:
        return OperatorRunMode.FRESH
    if args.resume:
        return OperatorRunMode.RESUME
    raise StorageLoadOperatorUsageError("one of --plan, --fresh, or --resume is required")


def _build_config(args: argparse.Namespace) -> StorageLoadOperatorConfig:
    verification_mode = VerificationMode(args.verification.upper())
    return StorageLoadOperatorConfig(
        artifact_root=args.artifact_root,
        checkpoint_root=args.checkpoint_root,
        evidence_root=args.evidence_root,
        relational_target=RelationalTargetId(args.relational_target),
        vector_target=VectorTargetId(args.vector_target),
        batch_size=BootstrapBatchSize(args.batch_size),
        run_mode=_resolve_run_mode(args),
        verification_mode=verification_mode,
        expected_record_count=args.expected_record_count,
        expected_data_pack_content_identity=args.expected_content_identity,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        config = _build_config(args)
    except SystemExit:
        return int(StorageLoadOperatorExitCode.PRECONDITION_ERROR)
    except (StorageLoadOperatorUsageError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return int(StorageLoadOperatorExitCode.PRECONDITION_ERROR)

    outcome = run_storage_load_operator(config)
    if outcome.message:
        print(outcome.message, file=sys.stderr)
    if outcome.result is not None:
        print(f"status={outcome.result.status.value}")
    return int(outcome.exit_code)
