#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""INT-MAINT-02 — P4 provider shells must expose a minimal health probe."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

P4_SHELL_SLUGS = (
    "pgvector",
    "duckdb",
    "influxdb",
    "timescaledb",
    "grafana",
    "loki",
    "tempo",
    "aws_secrets_manager",
    "azure_key_vault",
    "gcp_secret_manager",
    "doppler",
    "unleash",
    "launchdarkly",
    "github_actions",
    "redpanda",
    "cloudflare_r2",
    "memgraph",
    "falkordb",
    "incident_io",
    "kubernetes",
    "servicenow",
    "bitbucket",
    "asana",
    "sendgrid",
    "mailgun",
    "mlflow",
    "huggingface_hub",
    "ollama",
)


def main() -> int:
    test_path = REPO_ROOT / "tests" / "unit" / "integrations" / "providers" / "test_p5_m6_p4_providers.py"
    if not test_path.is_file():
        print(f"integration P4 shell probe audit failed: missing {test_path}")
        return 1
    source = test_path.read_text(encoding="utf-8")
    missing = [slug for slug in P4_SHELL_SLUGS if slug not in source]
    if missing:
        print("integration P4 shell probe audit failed:")
        for slug in missing:
            print(f"  - {slug}: no unit test evidence in test_p5_m6_p4_providers.py")
        return 1
    if "test_p4_shell_health_probe" not in source:
        print("integration P4 shell probe audit failed: test_p4_shell_health_probe missing")
        return 1
    print("integration P4 shell probe audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
