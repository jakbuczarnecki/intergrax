"""Run the backend-neutral TOKEN-10F proof harness from a TOML file."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from intergrax.runtime.token_optimization.proofs.config import (
    load_universal_token_optimization_proof_config,
)
from intergrax.runtime.token_optimization.proofs.contracts import (
    ProofConfigurationError,
    ProofExecutionError,
    ProofProviderUnavailableError,
)
from intergrax.runtime.token_optimization.proofs.runner import (
    UniversalTokenOptimizationProofRunner,
)
from intergrax.utils import attribute_access

EXIT_OK = 0
EXIT_INVALID_CONFIG = 2
EXIT_PROVIDER_UNAVAILABLE = 3
EXIT_EXECUTION_FAILED = 4
EXIT_ARTIFACT_FAILED = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--mode", choices=("offline_smoke", "live_adapter"))
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _print_summary(result) -> None:
    print(f"proof_id={result.proof_id}")
    print(f"run_id={result.run_id}")
    print(f"mode={result.run_mode}")
    print(f"cases={result.completed_count}/{result.case_count}")
    print(f"success={str(result.success).lower()}")
    if result.artifact_manifest.files:
        print(f"artifact_path={result.proof_id}/{result.run_id}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = load_universal_token_optimization_proof_config(args.config)
        if args.mode:
            config = replace(config, run_mode=args.mode)
        if args.validate_only:
            print(f"proof_id={config.proof_id}")
            print(f"mode={config.run_mode}")
            print("valid=true")
            return EXIT_OK
        result = UniversalTokenOptimizationProofRunner().run(
            config,
            output_directory=args.output_dir,
            run_id=args.run_id,
        )
    except ProofConfigurationError as exc:
        print(f"error={exc.reason_code}", file=sys.stderr)
        return (
            EXIT_PROVIDER_UNAVAILABLE
            if exc.reason_code == "MISSING_API_KEY_ENV"
            else EXIT_INVALID_CONFIG
        )
    except ProofProviderUnavailableError as exc:
        print(f"error={exc.reason_code}", file=sys.stderr)
        return EXIT_PROVIDER_UNAVAILABLE
    except ProofExecutionError as exc:
        print(f"error={exc.reason_code}", file=sys.stderr)
        return EXIT_EXECUTION_FAILED
    except Exception as exc:
        reason_code = attribute_access.optional(
            exc, "reason_code", "ARTIFACT_PERSISTENCE_FAILED"
        )
        print(f"error={reason_code}", file=sys.stderr)
        return EXIT_ARTIFACT_FAILED
    _print_summary(result)
    return EXIT_OK if result.success else EXIT_EXECUTION_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
