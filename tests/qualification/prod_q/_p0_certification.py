# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0 certification report assembly."""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from tests.qualification.prod_q._p0_discovery import repo_root
from tests.qualification.prod_q._p0_findings_register import known_prod_q_findings
from tests.qualification.prod_q._p0_frz_reconciliation import build_frz_reconciliation
from tests.qualification.prod_q._p0_inventory import (
    build_closed_world_discovery_report,
    build_family_coverage,
    build_ownership_matrix,
    build_p0_matrix,
)
from tests.qualification.prod_q._p0_ownership import build_ownership_reconciliation
from tests.qualification.prod_q._p0_types import (
    HistoricalEvidenceItem,
    ProdQP0CertificationReport,
    ProdQFindingDisposition,
    ProductionReadinessEvidenceState,
    ProviderQualificationDisposition,
    R1SqliteEnv01Disposition,
    SqliteReproductionClassification,
    TenantIsolationAuditProdQ,
)

_R1_HTTP_REPRODUCTION_ENV = "INTERGRAX_PROD_Q_R1_HTTP_REPRODUCTION"
_R1_REPRODUCTION_TIMEOUT_SEC = 60

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

_R1_GCA_NODE = (
    "applications/governed_contractor_application/tests/host/"
    "test_governed_contractor_canonical_execution.py::"
    "test_governed_contractor_http_root_uses_canonical_execution_facade"
)

_PROVIDER_REGISTER_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("E2B", "intergrax/integrations/providers/sandbox_host/e2b/bundle.py", "register_e2b_integration"),
    ("Modal", "intergrax/integrations/providers/sandbox_host/modal/bundle.py", "register_modal_integration"),
    ("Daytona", "intergrax/integrations/providers/sandbox_host/daytona/bundle.py", "register_daytona_integration"),
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


def classify_sqlite_header_bytes(data: bytes) -> str:
    if data.startswith(b"SQLite format 3"):
        return "SQLITE_HEADER"
    if not data:
        return "EMPTY"
    return data[:24].hex()


def probe_sqlite_header_at_path(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    try:
        data = path.read_bytes()[:16]
    except OSError:
        return "UNREADABLE"
    return classify_sqlite_header_bytes(data)


def probe_default_relational_db_header() -> str:
    return probe_sqlite_header_at_path(repo_root() / "build" / "intergrax.db")


def r1_http_reproduction_requested() -> bool:
    return os.environ.get(_R1_HTTP_REPRODUCTION_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _terminate_subprocess_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            cwd=repo_root(),
            capture_output=True,
            check=False,
        )
    else:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def run_prod_q_bounded_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout_sec: int = _R1_REPRODUCTION_TIMEOUT_SEC,
) -> tuple[int | None, str, bool]:
    """Run a subprocess with a hard timeout; return (exit_code, combined_output, timed_out)."""
    workdir = cwd or repo_root()
    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return proc.returncode, stdout + stderr, False
    except subprocess.TimeoutExpired:
        _terminate_subprocess_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        tail = (stdout + stderr)[-4000:]
        return None, tail or "subprocess timeout", True


def _isolated_db_path() -> Path:
    return repo_root() / ".tmp" / "session" / "prod-q-p0-r1" / "isolated-relational.db"


def _http_reproduction_db_path() -> Path:
    return _isolated_db_path().with_name("isolated-relational-http-repro.db")


def _parse_sqlite_reproduction_tail(
    tail: str,
) -> tuple[
    SqliteReproductionClassification,
    str,
    str,
    bool,
    bool,
]:
    failing_exception = ""
    failing_location = ""
    for line in tail.splitlines():
        stripped = line.strip()
        if "E   " in stripped and (
            "ToolDependencyAttemptBoundaryMaterializationError" in stripped
            or "ExecutionContinuationError" in stripped
        ):
            failing_exception = stripped.split("E   ", 1)[-1].strip()
        elif stripped.startswith("E   ") and (
            stripped.endswith("Error")
            or "Error:" in stripped
            or "Exception:" in stripped
        ):
            failing_exception = stripped.split("E   ", 1)[-1].strip()
        if ".py:" in stripped and (
            "intergrax" in stripped or "applications" in stripped
        ):
            if "FAILURES" not in stripped:
                failing_location = stripped

    lowered = tail.lower()
    sqlite_bootstrap = "sqlite" in lowered or "intergrax_relational_db" in lowered
    strict_materialization = (
        "ToolDependencyAttemptBoundaryMaterializationError" in tail
        or "dependency_attempt_boundary_composition" in tail
        or "strict production requires" in lowered
        or "ExecutionContinuationError" in tail
        or "production composition requires an explicit ExecutionContinuationStateStore" in tail
    )

    if strict_materialization:
        return (
            SqliteReproductionClassification.DEPENDENCY_MATERIALIZATION,
            failing_exception,
            failing_location,
            sqlite_bootstrap,
            True,
        )
    if "ModuleNotFoundError" in tail or "No module named" in tail:
        return (
            SqliteReproductionClassification.ENVIRONMENT_UNAVAILABLE,
            failing_exception,
            failing_location,
            sqlite_bootstrap,
            False,
        )
    if failing_exception or failing_location:
        return (
            SqliteReproductionClassification.TEST_ISOLATION_OR_HARNESS,
            failing_exception,
            failing_location,
            sqlite_bootstrap,
            strict_materialization,
        )
    return (
        SqliteReproductionClassification.ENVIRONMENT_UNAVAILABLE,
        failing_exception,
        failing_location,
        sqlite_bootstrap,
        False,
    )


def _isolated_sqlite_bootstrap_evidence() -> tuple[str, bool]:
    import sqlite3
    import uuid

    isolated = _isolated_db_path()
    isolated.parent.mkdir(parents=True, exist_ok=True)
    target = isolated
    if target.exists():
        try:
            target.unlink()
        except OSError:
            target = isolated.with_name(f"isolated-relational-bootstrap-{uuid.uuid4().hex}.db")
    with sqlite3.connect(target) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS prod_q_r1_isolated_probe (id INTEGER PRIMARY KEY)"
        )
        connection.commit()
    ok = target.read_bytes()[:16].startswith(b"SQLite format 3")
    return f"in_process_isolated_bootstrap_ok={ok}; path={target}", ok


def _execute_r1_sqlite_reproduction() -> tuple[
    int | None,
    str,
    SqliteReproductionClassification,
    str,
    str,
    bool,
    bool,
    bool,
    ProdQFindingDisposition,
]:
    isolated = _http_reproduction_db_path()
    isolated.parent.mkdir(parents=True, exist_ok=True)
    if isolated.exists():
        try:
            isolated.unlink()
        except OSError:
            pass
    env = os.environ.copy()
    env["INTERGRAX_RELATIONAL_DB"] = str(isolated)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        _R1_GCA_NODE,
        "-p",
        "no:xdist",
        "-q",
        "--tb=line",
    ]
    try:
        exit_code, combined, timed_out = run_prod_q_bounded_subprocess(cmd, env=env)
    except OSError as exc:
        return (
            None,
            str(exc),
            SqliteReproductionClassification.ENVIRONMENT_UNAVAILABLE,
            "",
            "",
            False,
            False,
            False,
            ProdQFindingDisposition.ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED,
        )
    tail = combined[-4000:]
    if timed_out:
        return (
            None,
            tail,
            SqliteReproductionClassification.ENVIRONMENT_UNAVAILABLE,
            "",
            "",
            False,
            False,
            True,
            ProdQFindingDisposition.ENVIRONMENT_TEST_ISSUE_EVIDENCE_REQUIRED,
        )
    if exit_code == 0:
        return (
            exit_code,
            tail,
            SqliteReproductionClassification.TEST_ISOLATION_OR_HARNESS,
            "",
            "",
            True,
            False,
            True,
            ProdQFindingDisposition.IN_SCOPE_BLOCKER,
        )
    classification, exc_type, location, sqlite_boot, strict_mat = _parse_sqlite_reproduction_tail(
        tail
    )
    return (
        exit_code,
        tail,
        classification,
        exc_type,
        location,
        sqlite_boot,
        strict_mat,
        True,
        ProdQFindingDisposition.IN_SCOPE_BLOCKER,
    )


@lru_cache(maxsize=1)
def build_r1_sqlite_disposition() -> R1SqliteEnv01Disposition:
    probe = probe_default_relational_db_header()
    command = (
        f"set {_R1_HTTP_REPRODUCTION_ENV}=1 && uv run --with cryptography pytest "
        f"{_R1_GCA_NODE} -p no:xdist -q"
    )
    bootstrap_summary, bootstrap_ok = _isolated_sqlite_bootstrap_evidence()
    if r1_http_reproduction_requested():
        (
            exit_code,
            summary,
            classification,
            exc_type,
            location,
            sqlite_boot,
            strict_mat,
            executed,
            finding_disposition,
        ) = _execute_r1_sqlite_reproduction()
        combined_summary = summary[:500] if summary else ""
    else:
        executed = False
        exit_code = None
        classification = SqliteReproductionClassification.NOT_EXECUTED
        exc_type = ""
        location = ""
        sqlite_boot = bootstrap_ok
        strict_mat = False
        finding_disposition = ProdQFindingDisposition.IN_SCOPE_BLOCKER
        combined_summary = (
            f"{bootstrap_summary}; http_reproduction=skipped "
            f"(set {_R1_HTTP_REPRODUCTION_ENV}=1 to execute bounded nested pytest)"
        )[:500]
    return R1SqliteEnv01Disposition(
        finding_id="R1-SQLITE-ENV-01",
        default_db_path="build/intergrax.db",
        default_db_header_probe=probe,
        isolated_env_var="INTERGRAX_RELATIONAL_DB",
        isolated_db_path=str(_isolated_db_path()),
        reproduction_executed=executed,
        reproduction_exit_code=exit_code if executed else None,
        reproduction_classification=classification,
        failing_exception_type=exc_type,
        failing_trace_location=location,
        sqlite_bootstrap_observed=sqlite_boot,
        strict_dependency_materialization_observed=strict_mat,
        reproduction_summary=combined_summary,
        evidence_state=ProductionReadinessEvidenceState.BLOCKER,
        finding_disposition=finding_disposition,
        recommended_child="PROD-Q-R1",
        reproduction_command=command,
    )


def _bootstrap_registers_provider(marker: str) -> bool:
    bootstrap = repo_root() / "intergrax" / "integrations" / "registry" / "bootstrap_m6_p6.py"
    if not bootstrap.is_file():
        return False
    try:
        text = bootstrap.read_text(encoding="utf-8")
    except OSError:
        return False
    return marker in text


def _registration_callable_exists(provider_id: str, register_marker: str) -> bool:
    slug = provider_id.lower()
    register_py = (
        repo_root()
        / "intergrax"
        / "integrations"
        / "providers"
        / "sandbox_host"
        / slug
        / "register.py"
    )
    if not register_py.is_file():
        return False
    try:
        text = register_py.read_text(encoding="utf-8")
    except OSError:
        return False
    return f"def {register_marker}" in text


@lru_cache(maxsize=1)
def build_provider_dispositions() -> tuple[ProviderQualificationDisposition, ...]:
    root = repo_root()
    out: list[ProviderQualificationDisposition] = []
    for provider_id, rel, register_marker in _PROVIDER_REGISTER_MARKERS:
        implementation_exists = (root / rel).is_file()
        registration_callable_exists = _registration_callable_exists(provider_id, register_marker)
        composition_reachable = _bootstrap_registers_provider(register_marker)
        registration_refs: tuple[str, ...] = ()
        registered = (
            implementation_exists
            and registration_callable_exists
            and composition_reachable
        )
        if registered:
            registration_refs = (
                rel,
                f"intergrax/integrations/providers/sandbox_host/{provider_id.lower()}/register.py",
                "intergrax/integrations/registry/bootstrap_m6_p6.py",
                register_marker,
            )
        out.append(
            ProviderQualificationDisposition(
                provider_id=provider_id,
                implementation_exists=implementation_exists,
                registration_callable_exists=registration_callable_exists,
                sanctioned_composition_reachable=composition_reachable,
                registered=registered,
                configured=False,
                effective=False,
                qualified=False,
                production_approved=False,
                evidence_state=ProductionReadinessEvidenceState.EXTERNAL_PROVIDER_EVIDENCE_REQUIRED,
                finding_disposition=ProdQFindingDisposition.IN_SCOPE_BLOCKER,
                registration_evidence=registration_refs,
                notes=(
                    "implementation → registration callable → sanctioned bootstrap composition "
                    "are separate evidence layers; live qualification not executed"
                ),
            )
        )
    return tuple(out)


@lru_cache(maxsize=1)
def build_tenant_isolation_audit() -> TenantIsolationAuditProdQ:
    from tests.qualification.prod_q._p0_classification import classified_inventory

    head = current_git_head()
    tenant_x_ok = _git_is_ancestor("5ccb72fccd4e3820ebdc9c18008afe6f7d59f558", head)
    contradictions: list[str] = []
    for record in classified_inventory():
        if not record.tenant_relevance:
            continue
        if record.finding_disposition == ProdQFindingDisposition.IN_SCOPE_BLOCKER:
            contradictions.append(
                f"{record.candidate_id}: tenant-relevant surface is IN-SCOPE BLOCKER"
            )
    if not tenant_x_ok:
        contradictions.append("TENANT-X accepted SHA not an ancestor of current HEAD")
    verdict = "BLOCKED" if contradictions else "DERIVED_PASS_REUSE_TENANT_X"
    return TenantIsolationAuditProdQ(
        tenant_scope_applicable=True,
        verdict=verdict,
        reuse_tenant_x_sha="5ccb72fccd4e3820ebdc9c18008afe6f7d59f558",
        production_path_contradictions=tuple(contradictions),
        evidence_refs=(
            "tests/qualification/tenant_x/",
            "classified_inventory tenant_relevance scan",
            "TENANT-X FRZ-TEN-01..12 PASS — not reopened by PROD-Q-P0",
        ),
    )


@lru_cache(maxsize=1)
def build_prod_q_p0_certification_report() -> ProdQP0CertificationReport:
    frz = build_frz_reconciliation()
    return ProdQP0CertificationReport(
        audited_head=current_git_head(),
        matrix=build_p0_matrix(len(frz)),
        known_findings=known_prod_q_findings(),
        discovery_report=build_closed_world_discovery_report(),
        family_coverage=build_family_coverage(),
        ownership=build_ownership_matrix(),
        ownership_reconciliation=build_ownership_reconciliation(),
        frz_records=frz,
        provider_dispositions=build_provider_dispositions(),
        r1_sqlite=build_r1_sqlite_disposition(),
        tenant_audit=build_tenant_isolation_audit(),
        recommended_children=_RECOMMENDED_CHILDREN,
    )
