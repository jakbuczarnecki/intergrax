"""Static infrastructure contract for ERL-QUAL-004 PostgreSQL Docker foundation."""

from __future__ import annotations

from pathlib import Path

INFRASTRUCTURE_ROOT = Path(__file__).resolve().parent
DOCKER_COMPOSE_FILE = INFRASTRUCTURE_ROOT / "docker" / "docker-compose.yml"
POSTGRES_ENV_EXAMPLE = INFRASTRUCTURE_ROOT / "config" / "postgres.env.example"
POSTGRES_ENV_FILE = INFRASTRUCTURE_ROOT / "config" / "postgres.env"

PINNED_POSTGRES_IMAGE = "postgres:16.6"
COMPOSE_SERVICE_NAME = "postgres"

REQUIRED_POSTGRES_ENV_KEYS: frozenset[str] = frozenset(
    {
        "COMPOSE_PROJECT_NAME",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "ERL_QUAL_004_POSTGRES_HOST_PORT",
    }
)

FORBIDDEN_PLACEHOLDER_SECRETS: frozenset[str] = frozenset(
    {
        "intergrax",
        "password",
        "secret",
        "admin123",
    }
)


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file (no export prefix, # comments)."""
    if not path.is_file():
        msg = f"missing env file: {path}"
        raise FileNotFoundError(msg)
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            msg = f"invalid env line (expected KEY=VALUE): {raw_line!r}"
            raise ValueError(msg)
        values[key.strip()] = value.strip()
    return values


def validate_postgres_env_contract(values: dict[str, str]) -> list[str]:
    """Return human-readable contract violations (empty when valid)."""
    violations: list[str] = []
    missing = REQUIRED_POSTGRES_ENV_KEYS - values.keys()
    if missing:
        violations.append(f"missing required keys: {', '.join(sorted(missing))}")
    port_raw = values.get("ERL_QUAL_004_POSTGRES_HOST_PORT", "")
    if port_raw and not port_raw.isdigit():
        violations.append("ERL_QUAL_004_POSTGRES_HOST_PORT must be a numeric port")
    password = values.get("POSTGRES_PASSWORD", "")
    if password and password.casefold() in FORBIDDEN_PLACEHOLDER_SECRETS:
        violations.append("POSTGRES_PASSWORD must not use a generic placeholder secret")
    return violations
