#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""RAG-MAINT-01 — STABLE vector-store slugs must match manifest maturity labels."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    from intergrax.integrations.contracts.base import IntegrationStatus
    from intergrax.rag.vectorstore.soak.prod_slo import (
        BETA_PROMOTION_CANDIDATE_SLUGS,
        STABLE_PROD_SLO_SLUGS,
        manifest_status_for_slug,
    )

    violations: list[str] = []

    for slug in STABLE_PROD_SLO_SLUGS:
        status = manifest_status_for_slug(slug)
        if status is not IntegrationStatus.STABLE:
            violations.append(f"{slug}: listed in STABLE_PROD_SLO_SLUGS but manifest is {status.value}")

    for slug in BETA_PROMOTION_CANDIDATE_SLUGS:
        status = manifest_status_for_slug(slug)
        if status is IntegrationStatus.STABLE:
            violations.append(
                f"{slug}: beta promotion candidate must not be STABLE without soak evidence override",
            )

    if violations:
        print("rag maturity label audit: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1
    print("rag maturity label audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
