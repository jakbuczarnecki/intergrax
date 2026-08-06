"""Evaluate a completed TOKEN-10F proof or run it offline first."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
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
from intergrax.runtime.token_optimization.proofs.corpus import (
    expand_proof_config_with_corpus,
    load_proof_corpus,
)
from intergrax.runtime.token_optimization.proofs.evaluation_contracts import (
    EvaluationConfigurationError,
    EvaluationProfile,
    GateStatus,
    load_cache_evidence,
    load_evaluation_config,
    load_universal_proof_run_result,
)
from intergrax.runtime.token_optimization.proofs.evaluator import (
    UniversalProofEvaluator,
)
from intergrax.runtime.token_optimization.proofs.report import (
    write_evaluation_artifacts,
)
from intergrax.runtime.token_optimization.proofs.runner import (
    UniversalTokenOptimizationProofRunner,
)

EXIT_OK = 0
EXIT_INVALID_CONFIG = 2
EXIT_REQUIRED_EVIDENCE_UNAVAILABLE = 3
EXIT_PROOF_EXECUTION_FAILED = 4
EXIT_HARD_GATE_FAILED = 5
EXIT_ARTIFACT_FAILED = 6


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-config", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--run-result", type=Path)
    parser.add_argument("--cache-evidence", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluation-id")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser


def _print_summary(evaluation) -> None:
    counts = evaluation.status_counts
    print(f"evaluation_id={evaluation.evaluation_id}")
    print(f"proof_id={evaluation.proof_id}")
    print(f"run_id={evaluation.run_id}")
    print(f"profile={evaluation.profile.value}")
    print(f"cases={len(evaluation.cases)}")
    print(
        "gates="
        + ",".join(
            f"{status.value}:{counts.get(status.value, 0)}" for status in GateStatus
        )
    )
    print(f"success={str(evaluation.success).lower()}")


def _default_corpus_path(proof_config_path: Path) -> Path:
    sibling = proof_config_path.parent / "corpus" / "universal_proof_cases.toml"
    if sibling.is_file():
        return sibling
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "token_optimization"
        / "corpus"
        / "universal_proof_cases.toml"
    )


def _prepare_offline_config_load(path: Path) -> None:
    """Satisfy TOKEN-10F's live-config key check without making network calls."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        adapter = data.get("adapter", {})
        env_name = adapter.get("api_key_env") if isinstance(adapter, dict) else None
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return
    if isinstance(env_name, str) and env_name:
        os.environ.setdefault(env_name, "TOKEN_10G_OFFLINE_PLACEHOLDER")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        proof_config = None
        if not args.evaluate_only:
            if args.offline:
                _prepare_offline_config_load(args.proof_config)
            proof_config = load_universal_token_optimization_proof_config(
                args.proof_config
            )
        corpus = load_proof_corpus(_default_corpus_path(args.proof_config))
        evaluation_config = load_evaluation_config(args.evaluation_config)
        if (
            args.offline
            and evaluation_config.profile is not EvaluationProfile.OFFLINE_COMPOSITION
        ):
            raise EvaluationConfigurationError(
                "OFFLINE_REQUIRES_OFFLINE_COMPOSITION_PROFILE"
            )
        if args.evaluate_only:
            if args.run_result is None:
                raise EvaluationConfigurationError("RUN_RESULT_REQUIRED")
            run_result = load_universal_proof_run_result(args.run_result)
        else:
            if args.offline:
                proof_config = replace(proof_config, run_mode="offline_smoke")
            proof_config = expand_proof_config_with_corpus(proof_config, corpus)
            run_result = UniversalTokenOptimizationProofRunner().run(
                proof_config,
                output_directory=(
                    (
                        args.output_dir
                        or Path(".artifacts/token_optimization/proof-evaluation")
                    )
                    / "proof-runs"
                ),
            )
        cache = load_cache_evidence(args.cache_evidence) if args.cache_evidence else ()
        evaluation = UniversalProofEvaluator().evaluate(
            run_result,
            corpus,
            evaluation_config,
            cache_evidence=cache,
            evaluation_id=args.evaluation_id or f"evaluation-{run_result.run_id}",
        )
        output_dir = args.output_dir or Path(
            ".artifacts/token_optimization/proof-evaluation"
        )
        persisted = write_evaluation_artifacts(
            evaluation,
            output_directory=output_dir,
        )
    except ProofConfigurationError as exc:
        print(f"error={exc.reason_code}", file=sys.stderr)
        return (
            EXIT_REQUIRED_EVIDENCE_UNAVAILABLE
            if exc.reason_code == "MISSING_API_KEY_ENV"
            else EXIT_INVALID_CONFIG
        )
    except ProofProviderUnavailableError as exc:
        print(f"error={exc.reason_code}", file=sys.stderr)
        return EXIT_REQUIRED_EVIDENCE_UNAVAILABLE
    except ProofExecutionError as exc:
        print(f"error={exc.reason_code}", file=sys.stderr)
        return EXIT_PROOF_EXECUTION_FAILED
    except EvaluationConfigurationError as exc:
        print(f"error={exc.reason_code}", file=sys.stderr)
        return EXIT_INVALID_CONFIG
    except Exception as exc:  # noqa: BLE001
        reason_code = getattr(exc, "reason_code", "EVALUATION_ARTIFACT_WRITE_FAILED")
        print(f"error={reason_code}", file=sys.stderr)
        return EXIT_ARTIFACT_FAILED
    _print_summary(persisted)
    if persisted.success:
        return EXIT_OK
    unavailable = any(
        gate.required
        and gate.status is GateStatus.UNAVAILABLE
        and gate.gate_id not in evaluation_config.unavailable_allowed_gate_ids
        for case in persisted.cases
        for gate in case.gates
    )
    return EXIT_REQUIRED_EVIDENCE_UNAVAILABLE if unavailable else EXIT_HARD_GATE_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
