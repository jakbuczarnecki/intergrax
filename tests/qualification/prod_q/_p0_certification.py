# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0 certification report assembly."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from tests.qualification.prod_q._p0_discovery import repo_root
from tests.qualification.prod_q._p0_frz_reconciliation import build_frz_reconciliation
from tests.qualification.prod_q._p0_inventory import (
    build_closed_world_discovery_report,
    build_family_coverage,
    build_ownership_matrix,
    build_p0_matrix,
)
from tests.qualification.prod_q._p0_types import (
    HistoricalEvidenceItem,
    ProdQP0CertificationReport,
    ProdQFindingDisposition,
    ProductionReadinessEvidenceState,
    ProviderQualificationDisposition,
    R1SqliteEnv01Disposition,
    TenantIsolationAuditProdQ,
)

HISTORICAL_EVIDENCE: tuple[HistoricalEvidenceItem, ...] = (
    HistoricalEvidenceItem("TENANT-X", "5ccb72fccd4e3820ebdc9c18008afe6f7d59f558", True),
    HistoricalEvidenceItem("CTRL-X", "318a4d835534ef5c48575f4649551c957dacd8ea", True),
    HistoricalEvidenceItem("STATE-X", "bd54941d933069b8bfb2819bb819c1cbdbe71576", True),
    HistoricalEvidenceItem("CONFIG-X", "fac9a43f3b128fde63541d2b56ebd8f4a751e8fb", True),
    HistoricalEvidenceItem("COMPAT-X", "84b8739b8b6e62c8e580e26ca3011f6aa46d457a", True),
    HistoricalEvidenceItem("HARNESS-W4", "f38e455ccc391dad9a6a64d7867a33cfc5c9b21d", True),
)

_RECOMMENDED_CHILDREN: tuple[str, ...] = (
    "PROD-Q-R1 — Production Startup / Dependency Materialization Corrective (R1-SQLITE-ENV-01)",
    "PROD-Q-P1 — External sandbox provider production qualification (E2B, Modal, Daytona)",
    "PROD-Q-P2 — Production strict-vs-lab reachability and bypass closure",
    "PROD-Q-P3 — Production operability surface (health/readiness/operator semantics wiring)",
)


def current_git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=repo_root()).strip()


def _git_is_ancestor(ancestor: str, head: str) -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, head],
        cwd=repo_root(),
        capture_output=True,
    )
    return proc.returncode == 0


def historical_evidence_sha_mismatches() -> tuple[tuple[str, str], ...]:
    head = current_git_head()
    mismatches: list[tuple[str, str]] = []
    for item in HISTORICAL_EVIDENCE:
        if item.replay_required and not _git_is_ancestor(item.accepted_sha, head):
            mismatches.append((item.stage_id, item.accepted_sha))
    return tuple(mismatches)


def probe_default_relational_db_header() -> str:
    path = repo_root() / "build" / "intergrax.db"
    if not path.is_file():
        return "MISSING"
    try:
        data = path.read_bytes()[:16]
    except OSError:
        return "UNREADABLE"
    if data.startswith(b"SQLite format 3"):
        return "SQLITE_HEADER"
    return data[:24].hex()


@lru_cache(maxsize=1)
def build_r1_sqlite_disposition() -> R1SqliteEnv01Disposition:
    probe = probe_default_relational_db_header()
    return R1SqliteEnv01Disposition(
        finding_id="R1-SQLITE-ENV-01",
        default_db_path="build/intergrax.db",
        default_db_header_probe=probe,
        isolated_env_var="INTERGRAX_RELATIONAL_DB",
        evidence_state=ProductionReadinessEvidenceState.BLOCKER,
        finding_disposition=ProdQFindingDisposition.IN_SCOPE_BLOCKER,
        recommended_child="PROD-Q-R1",
        reproduction_command=(
            "uv run --with cryptography pytest "
            "applications/governed_contractor_application/tests/host/"
            "test_governed_contractor_canonical_execution.py::"
            "test_governed_contractor_http_root_uses_canonical_execution_facade "
            "-p no:xdist -q"
        ),
    )


@lru_cache(maxsize=1)
def build_provider_dispositions() -> tuple[ProviderQualificationDisposition, ...]:
    root = repo_root()
    providers = (
        ("E2B", "intergrax/integrations/providers/sandbox_host/e2b/bundle.py"),
        ("Modal", "intergrax/integrations/providers/sandbox_host/modal/bundle.py"),
        ("Daytona", "intergrax/integrations/providers/sandbox_host/daytona/bundle.py"),
    )
    out: list[ProviderQualificationDisposition] = []
    for provider_id, rel in providers:
        installed = (root / rel).is_file()
        out.append(
            ProviderQualificationDisposition(
                provider_id=provider_id,
                installed=installed,
                registered=installed,
                configured=False,
                qualified=False,
                production_approved=False,
                evidence_state=ProductionReadinessEvidenceState.EXTERNAL_PROVIDER_EVIDENCE_REQUIRED,
                finding_disposition=ProdQFindingDisposition.IN_SCOPE_BLOCKER,
                notes="AW-7C reference substrate does not substitute live provider PROD-Q qualification",
            )
        )
    return tuple(out)


@lru_cache(maxsize=1)
def build_tenant_isolation_audit() -> TenantIsolationAuditProdQ:
    return TenantIsolationAuditProdQ(
        tenant_scope_applicable=True,
        verdict="PASS",
        reuse_tenant_x_sha="5ccb72fccd4e3820ebdc9c18008afe6f7d59f558",
        production_path_contradictions=(),
        evidence_refs=(
            "tests/qualification/tenant_x/",
            "TENANT-X FRZ-TEN-01..12 PASS — not reopened by PROD-Q-P0",
            "tests/qualification/prod_q/_p0_classification.py PQ-F22 reuse",
        ),
    )


@lru_cache(maxsize=1)
def build_prod_q_p0_certification_report() -> ProdQP0CertificationReport:
    frz = build_frz_reconciliation()
    return ProdQP0CertificationReport(
        audited_head=current_git_head(),
        matrix=build_p0_matrix(len(frz)),
        discovery_report=build_closed_world_discovery_report(),
        family_coverage=build_family_coverage(),
        ownership=build_ownership_matrix(),
        frz_records=frz,
        provider_dispositions=build_provider_dispositions(),
        r1_sqlite=build_r1_sqlite_disposition(),
        tenant_audit=build_tenant_isolation_audit(),
        recommended_children=_RECOMMENDED_CHILDREN,
    )
