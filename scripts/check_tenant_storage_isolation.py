#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-4.2 — Postgres multi-tenant storage isolation gate."""

from __future__ import annotations

import sys

from intergrax.applications._shared.tenant_storage_wiring import (
    product_requires_tenant_storage_isolation,
    resolve_tenant_postgresql_config,
    tenant_schema_for_id,
    tenant_storage_isolation_ready,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    if not product_requires_tenant_storage_isolation(env):
        print("product_defaults must bind PostgreSQL relational_store", file=sys.stderr)
        return 1
    if not tenant_storage_isolation_ready(env):
        print("product PostgreSQL profile must declare tenant_schema", file=sys.stderr)
        return 1

    schema = tenant_schema_for_id("tenant-acme-01")
    if not schema.startswith("tenant") and not schema.startswith("t_"):
        print(f"unexpected tenant schema slug: {schema}", file=sys.stderr)
        return 1

    config = resolve_tenant_postgresql_config("tenant-acme-01")
    if config.tenant_schema != schema:
        print("resolve_tenant_postgresql_config must apply per-tenant schema", file=sys.stderr)
        return 1

    print(f"OK: tenant storage isolation (schema={schema})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
