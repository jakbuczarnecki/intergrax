"""Infrastructure-level PostgreSQL health verification for ERL-QUAL-004."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.infrastructure.contract import (
    DOCKER_COMPOSE_FILE,
    POSTGRES_ENV_FILE,
    COMPOSE_SERVICE_NAME,
    parse_env_file,
    validate_postgres_env_contract,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True, slots=True)
class HealthVerificationOutcome:
    container_running: bool
    postgres_accepting_connections: bool
    detail: str

    @property
    def ok(self) -> bool:
        return self.container_running and self.postgres_accepting_connections


def _compose_base_args(*, env_file: Path) -> list[str]:
    return [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-f",
        str(DOCKER_COMPOSE_FILE),
    ]


def _container_id(*, env_file: Path) -> str | None:
    command = [
        *_compose_base_args(env_file=env_file),
        "ps",
        "-q",
        COMPOSE_SERVICE_NAME,
    ]
    completed = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    container_id = completed.stdout.strip()
    return container_id or None


def verify_postgres_infrastructure(*, env_file: Path = POSTGRES_ENV_FILE) -> HealthVerificationOutcome:
    """Determine whether the lab PostgreSQL container is up and accepting connections."""
    if not env_file.is_file():
        return HealthVerificationOutcome(
            container_running=False,
            postgres_accepting_connections=False,
            detail=(
                f"configuration missing: {env_file}. "
                "Copy infrastructure/config/postgres.env.example to postgres.env."
            ),
        )

    contract_violations = validate_postgres_env_contract(parse_env_file(env_file))
    if contract_violations:
        return HealthVerificationOutcome(
            container_running=False,
            postgres_accepting_connections=False,
            detail="; ".join(contract_violations),
        )

    env_values = parse_env_file(env_file)

    container_id = _container_id(env_file=env_file)
    if container_id is None:
        return HealthVerificationOutcome(
            container_running=False,
            postgres_accepting_connections=False,
            detail=(
                "PostgreSQL container not running. "
                f"Start with: docker compose --env-file {env_file} "
                f"-f {DOCKER_COMPOSE_FILE} up -d"
            ),
        )

    pg_isready = subprocess.run(
        [
            *_compose_base_args(env_file=env_file),
            "exec",
            "-T",
            COMPOSE_SERVICE_NAME,
            "pg_isready",
            "-U",
            env_values["POSTGRES_USER"],
            "-d",
            env_values["POSTGRES_DB"],
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    accepting = pg_isready.returncode == 0
    detail = pg_isready.stdout.strip() or pg_isready.stderr.strip() or "pg_isready check"
    return HealthVerificationOutcome(
        container_running=True,
        postgres_accepting_connections=accepting,
        detail=detail,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify ERL-QUAL-004 PostgreSQL Docker infrastructure health.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=POSTGRES_ENV_FILE,
        help="Path to postgres.env (default: infrastructure/config/postgres.env)",
    )
    args = parser.parse_args(argv)
    outcome = verify_postgres_infrastructure(env_file=args.env_file)
    if outcome.ok:
        print(f"OK: {outcome.detail}")
        return 0
    print(f"BLOCKED_ENVIRONMENT: {outcome.detail}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
