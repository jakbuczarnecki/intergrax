#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3C).

"""Canonical executable entrypoint for TOOLS-ITERATIVE-SQL-INVESTIGATION."""

from __future__ import annotations

import argparse
import json
import os
import sys

from platform_proofs.tools.iterative_sql_investigation.dataset_identity import (
    PROOF_ID,
    DatasetIdentity,
    compute_dataset_fingerprint,
)
from platform_proofs.tools.iterative_sql_investigation.investigation_runtime import (
    ProofConfigurationError,
    ProofProviderUnavailableError,
    build_real_llm_adapter,
    model_provider_identity,
    resolve_llm_profile_from_env,
    run_investigation_scenario,
)
from platform_proofs.tools.iterative_sql_investigation.proof_result import (
    ToolsSqlInvestigationProofResult,
)
from platform_proofs.tools.iterative_sql_investigation.runtime import (
    DSN_ENV,
    DEFAULT_RUNTIME_DSN,
    build_proof_sql_runtime,
)
from platform_proofs.tools.iterative_sql_investigation.scenarios import ALL_SCENARIOS
from platform_proofs.tools.iterative_sql_investigation.setup import (
    DatasetSetupError,
    materialize_and_verify_dataset,
    verify_postgres_reachable,
)

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CONFIG = 2
EXIT_PROVIDER = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--admin-dsn",
        help="Override INTERGRAX_PP_SQL_INVESTIGATION_ADMIN_DSN for setup/materialization.",
    )
    parser.add_argument(
        "--runtime-dsn",
        help="Override INTERGRAX_PP_SQL_INVESTIGATION_DSN for read-only SQL runtime.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and dataset setup without running live scenarios.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable proof result JSON on stdout.",
    )
    return parser


def _resolve_runtime_dsn(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return os.environ.get(DSN_ENV, DEFAULT_RUNTIME_DSN).strip()


def _print_summary(result: ToolsSqlInvestigationProofResult) -> None:
    print(f"proof_id={result.proof_id}")
    print(f"overall_pass={str(result.overall_pass).lower()}")
    print(f"dataset_fingerprint={result.dataset_fingerprint_sha256}")
    print(
        "model_provider="
        f"{result.model_provider.provider}/{result.model_provider.model}"
    )
    if result.blocked_reason:
        print(f"blocked_reason={result.blocked_reason}")
    for scenario in result.scenarios:
        print(
            f"scenario_{scenario.scenario_id.value}="
            f"{'pass' if scenario.passed else 'fail'}"
        )
        if scenario.failure_reasons:
            print(
                f"scenario_{scenario.scenario_id.value}_reasons="
                + ",".join(scenario.failure_reasons)
            )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    identity = DatasetIdentity.canonical()
    fingerprint = compute_dataset_fingerprint(identity)

    try:
        verify_postgres_reachable(admin_dsn=args.admin_dsn)
        setup = materialize_and_verify_dataset(admin_dsn=args.admin_dsn, identity=identity)
    except DatasetSetupError as exc:
        print(f"error=dataset_setup:{exc}", file=sys.stderr)
        return EXIT_FAIL

    if args.validate_only:
        print(f"proof_id={PROOF_ID}")
        print(f"dataset_verified=true")
        print(f"dataset_fingerprint={fingerprint.sha256}")
        print(f"loaded_rows={setup.loaded_rows}")
        return EXIT_OK

    try:
        profile = resolve_llm_profile_from_env()
        llm = build_real_llm_adapter(profile)
    except (ProofProviderUnavailableError, ProofConfigurationError) as exc:
        blocked = ToolsSqlInvestigationProofResult.blocked(
            proof_id=PROOF_ID,
            identity=identity,
            fingerprint=fingerprint,
            reason=str(exc),
        )
        if args.json:
            print(json.dumps(blocked.model_dump(mode="json"), sort_keys=True))
        else:
            _print_summary(blocked)
        return EXIT_PROVIDER if isinstance(exc, ProofProviderUnavailableError) else EXIT_CONFIG

    runtime_dsn = _resolve_runtime_dsn(args.runtime_dsn)
    proof_runtime = build_proof_sql_runtime(dsn=runtime_dsn)
    scenario_results: list = []
    try:
        for scenario in ALL_SCENARIOS:
            scenario_results.append(
                run_investigation_scenario(
                    scenario=scenario,
                    llm=llm,
                    proof_runtime=proof_runtime,
                )
            )
    finally:
        proof_runtime.close()

    overall_pass = all(item.passed for item in scenario_results)
    result = ToolsSqlInvestigationProofResult(
        proof_id=PROOF_ID,
        dataset_identity=identity.as_dict(),
        dataset_fingerprint_sha256=fingerprint.sha256,
        db_verification_stats=setup.db_stats.as_dict(),
        model_provider=model_provider_identity(llm, profile),
        scenarios=tuple(scenario_results),
        overall_pass=overall_pass,
    )
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    else:
        _print_summary(result)
    return EXIT_OK if overall_pass else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
