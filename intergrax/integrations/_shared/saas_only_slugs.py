# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SaaS-only integration slugs — no local container substitute (INT-MAINT-03)."""

from __future__ import annotations

SAAS_ONLY_SLUGS: frozenset[str] = frozenset(
    {
        "launchdarkly",
        "pagerduty",
        "servicenow",
        "incident_io",
        "sendgrid",
        "mailgun",
        "asana",
        "bitbucket",
        "github_actions",
        "aws_secrets_manager",
        "azure_key_vault",
        "gcp_secret_manager",
        "cloudflare_r2",
        "huggingface_hub",
    }
)

LOCAL_CONTAINER_SLUGS: frozenset[str] = frozenset(
    {
        "pgvector",
        "timescaledb",
        "kafka",
        "redpanda",
        "minio",
        "vault",
        "kubernetes",
    }
)
