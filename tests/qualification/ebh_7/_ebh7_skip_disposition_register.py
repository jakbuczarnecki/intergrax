# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E501

"""EBH-7 L3 skipped-test disposition register (one individual record per skipped node).

SSOT for the nine ``pytest.skip`` outcomes observed in GHA L3 run ``41231430749`` (JUnit: 590 tests,
581 passed, 9 skipped). Every record is an approved ``N/A - WITH EVIDENCE``: the skipped parametrization
routes to an executed-equivalent node that must be PASS in the *same* JUnit. The workflow verifies this
against the real ``junit.xml`` (``scripts/qualification/ebh7_junit_evidence.py skip-disposition``); nothing
here hardcodes JUnit totals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

_W6 = "tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py"
_W7 = "tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py"
_W9 = "tests/qualification/ebh_7/test_ebh_7_w9_dossier_gates.py"
_TEN_PLATFORM = "tests/qualification/ebh_7/tenant_platform_e2e/test_ebh7_platform_tenant_boundary_proofs.py"
_XDM_PLATFORM = "tests/qualification/ebh_7/cross_domain_e2e/test_ebh7_cross_domain_platform_proofs.py"

# Last commit touching the skip-bearing test file at the pre-reconciliation TEST_EXECUTION_SHA 496016c0c.
_SRC_W6_W7 = "8bbda35dd1686b15bd6959b732ade3ba0f268632"
_SRC_W9 = "496016c0cf6cc5cf1f9a29d885677605dd0415d3"

_ROUTED = "SKIP_ROUTED_TO_EQUIVALENT"
_LOCAL = "SKIP_LOCAL_STAGING_ONLY"


class SkipDisposition(StrEnum):
    ACCEPTED_NA = "ACCEPTED_NA"
    BLOCKED = "BLOCKED"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class SkipAuditStatus(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class SkipDispositionRecord:
    node_id: str
    skip_reason: str
    skip_source: str  # pytest.skip call site (file:line) — mechanism: in-test ``pytest.skip``
    requirement: str
    mandatory: bool
    disposition: SkipDisposition
    evidence_nodes: tuple[str, ...]  # must be PASS in the same JUnit
    evidence_documents: tuple[str, ...]
    source_sha: str
    justification: str
    audit_status: SkipAuditStatus
    scenario_id: str | None = None
    environment_dependent: bool = False


def _routed(
    *,
    param_file: str,
    test_fn: str,
    scenario_id: str,
    reason: str,
    line: int,
    requirement: str,
    evidence_nodes: tuple[str, ...],
    justification: str,
) -> SkipDispositionRecord:
    return SkipDispositionRecord(
        node_id=f"{param_file}::{test_fn}[{scenario_id}]",
        skip_reason=reason,
        skip_source=f"{param_file}:{line} pytest.skip ({_ROUTED})",
        requirement=requirement,
        mandatory=True,
        disposition=SkipDisposition.ACCEPTED_NA,
        evidence_nodes=evidence_nodes,
        evidence_documents=(
            "docs/project/maintainers/qualification/ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md",
        ),
        source_sha=_SRC_W6_W7,
        justification=justification,
        audit_status=SkipAuditStatus.READY,
        scenario_id=scenario_id,
    )


_W6_TEST = "test_ebh7_w6_tenant_e2e_executes_registered_proof"
_W7_TEST = "test_ebh7_w7_cross_domain_executes_registered_proof"
_ASYNC_REASON = "async/governed_execution proofs run in tenant_platform_e2e module"
_TMP_W6_REASON = "proof requires tmp_path — run in dedicated fixture test"
_TMP_W7_REASON = "proof requires tmp_path"

_ROUTE_NOTE = (
    "The skipped parametrization would call the registered proof callable through "
    "`execute_proof()` (plain function call, no pytest fixture/async support). The identical callable "
    "is collected and executed natively by pytest in the evidence node(s), which must be PASS in the "
    "same JUnit. No obligation is dropped; the skipped parameter is a duplicate, fixture-incapable route."
)

SKIP_DISPOSITION_REGISTER: Final[tuple[SkipDispositionRecord, ...]] = (
    _routed(
        param_file=_W6,
        test_fn=_W6_TEST,
        scenario_id="EBH7-TEN-E2E-04",
        reason=_ASYNC_REASON,
        line=27,
        requirement="EBH7-TEN-E2E-04 (Exec→Tool, DENY cross-tenant tool scope) — tenant E2E matrix 26/26 EXACT",
        evidence_nodes=(f"{_TEN_PLATFORM}::test_ebh7_ten_e2e_04_exec_tool_tenant_boundary_tool_runtime",),
        justification=(
            "Registered proof `test_ebh7_ten_e2e_04_exec_tool_tenant_boundary_tool_runtime` is `async` and "
            "cannot run via synchronous `execute_proof`; it is executed natively (pytest-asyncio) in the "
            "tenant_platform_e2e module. " + _ROUTE_NOTE
        ),
    ),
    _routed(
        param_file=_W6,
        test_fn=_W6_TEST,
        scenario_id="EBH7-TEN-E2E-07",
        reason=_TMP_W6_REASON,
        line=29,
        requirement="EBH7-TEN-E2E-07 (Persistence→Cache, DENY cross-tenant durable load) — tenant E2E matrix 26/26 EXACT",
        evidence_nodes=(f"{_W6}::test_ebh7_w6_tenant_e2e_scenario_07_durable_cross_tenant",),
        justification=(
            "Proof `test_tx_r3_q07_adversarial_durable_load_cross_tenant` needs `tmp_path`. The dedicated "
            "fixture test `test_ebh7_w6_tenant_e2e_scenario_07_durable_cross_tenant` runs the same registered "
            "callable with a real `tmp_path` (also recorded as `executed_equivalent_node_id` in the semantic "
            "disposition SSOT). " + _ROUTE_NOTE
        ),
    ),
    _routed(
        param_file=_W6,
        test_fn=_W6_TEST,
        scenario_id="EBH7-TEN-E2E-11",
        reason=_ASYNC_REASON,
        line=27,
        requirement="EBH7-TEN-E2E-11 (Trace→Evidence, REJECT cross export) — tenant E2E matrix 26/26 EXACT",
        evidence_nodes=(f"{_TEN_PLATFORM}::test_ebh7_ten_e2e_11_trace_attribution_cross_tenant_rejected",),
        justification=(
            "Proof requires the `governed_execution` pytest fixture; it is executed natively with that fixture "
            "in the tenant_platform_e2e module (also recorded as `executed_equivalent_node_id`). " + _ROUTE_NOTE
        ),
    ),
    _routed(
        param_file=_W6,
        test_fn=_W6_TEST,
        scenario_id="EBH7-TEN-E2E-14",
        reason=_TMP_W6_REASON,
        line=29,
        requirement="EBH7-TEN-E2E-14 (Provider→Persistence, PRESERVE tenant on backend switch) — tenant E2E matrix 26/26 EXACT",
        evidence_nodes=(f"{_TEN_PLATFORM}::test_ebh7_ten_e2e_14_backend_switch_preserves_tenant",),
        justification=(
            "Proof `test_ebh7_ten_e2e_14_backend_switch_preserves_tenant(tmp_path)` is executed natively by "
            "pytest with a real `tmp_path` (sqlite/duckdb dual slug, distinct tenant binding). " + _ROUTE_NOTE
        ),
    ),
    _routed(
        param_file=_W7,
        test_fn=_W7_TEST,
        scenario_id="EBH7-XDM-010",
        reason=_TMP_W7_REASON,
        line=25,
        requirement="EBH7-XDM-010 (Backend x Persistence, REJECT sqlite bleed) — cross-domain matrix 34/34 EXACT",
        evidence_nodes=(
            f"{_W7}::test_ebh7_w7_cross_domain_xdm_010_sqlite_behavioral",
            f"{_XDM_PLATFORM}::test_ebh7_xdm_010_backend_persistence_sqlite_behavioral",
        ),
        justification=(
            "Proof needs `tmp_path`; executed natively in the cross-domain platform module and again by the "
            "dedicated W7 fixture test (also `executed_equivalent_node_id`). " + _ROUTE_NOTE
        ),
    ),
    _routed(
        param_file=_W7,
        test_fn=_W7_TEST,
        scenario_id="EBH7-XDM-016",
        reason=_TMP_W7_REASON,
        line=25,
        requirement="EBH7-XDM-016 (VBN x Composition, REJECT hardcoded slug) — cross-domain matrix 34/34 EXACT",
        evidence_nodes=(f"{_XDM_PLATFORM}::test_ebh7_xdm_016_vbn_composition_factory_behavioral",),
        justification=(
            "Proof `test_ebh7_xdm_016_vbn_composition_factory_behavioral(tmp_path)` is executed natively by "
            "pytest with a real `tmp_path` (Package C v2 relational dual slug + unregistered fail-closed). "
            + _ROUTE_NOTE
        ),
    ),
    _routed(
        param_file=_W7,
        test_fn=_W7_TEST,
        scenario_id="EBH7-XDM-017",
        reason=_TMP_W7_REASON,
        line=25,
        requirement="EBH7-XDM-017 (Failure x Diagnostics, REJECT tenant leak) — cross-domain matrix 34/34 EXACT",
        evidence_nodes=(f"{_XDM_PLATFORM}::test_ebh7_xdm_017_failure_diagnostics_tenant_leak",),
        justification=(
            "Proof `test_ebh7_xdm_017_failure_diagnostics_tenant_leak` (Trace P5 configuration/profile tenant "
            "mismatch fail-closed) is executed natively by pytest in the cross-domain platform module. " + _ROUTE_NOTE
        ),
    ),
    _routed(
        param_file=_W7,
        test_fn=_W7_TEST,
        scenario_id="EBH7-XDM-029",
        reason=_TMP_W7_REASON,
        line=25,
        requirement="EBH7-XDM-029 (Integration x External, DENY DI bypass) — cross-domain matrix 34/34 EXACT",
        evidence_nodes=(
            f"{_W7}::test_ebh7_w7_cross_domain_xdm_029_integration_external_di_boundary",
            f"{_XDM_PLATFORM}::test_ebh7_xdm_029_integration_external_di_boundary",
        ),
        justification=(
            "Proof needs `tmp_path`; executed natively in the cross-domain platform module and again by the "
            "dedicated W7 fixture test. " + _ROUTE_NOTE
        ),
    ),
    SkipDispositionRecord(
        node_id=f"{_W9}::test_ebh7_w9_proof_staging_dir_exists",
        skip_reason="gitignored local proof staging (.tmp/) — absent in a clean checkout; not replayable evidence",
        skip_source=f"{_W9}:26 pytest.skip ({_LOCAL})",
        requirement=(
            "EBH-7 W9 evidence dossier index lists `.tmp/proof/ebh-7-materialization/MANIFEST.md` as a staged, "
            "gitignored local artifact (EBH_7_EVIDENCE_DOSSIER_INDEX.md) — not a replay-time obligation"
        ),
        mandatory=False,
        disposition=SkipDisposition.ACCEPTED_NA,
        evidence_nodes=(f"{_W9}::test_ebh7_w9_dossier_index_present",),
        evidence_documents=(
            "docs/project/maintainers/qualification/EBH_7_EVIDENCE_DOSSIER_INDEX.md",
            "docs/project/maintainers/qualification/ebh_7_p03_independent_replay/ARTIFACT_DIGESTS.json",
            "docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json",
        ),
        source_sha=_SRC_W9,
        justification=(
            "The gate inspects an operator-local, gitignored staging directory that by definition does not "
            "exist in a clean checkout and is not replayable evidence. Replayable W9 evidence is the committed "
            "dossier index (node `test_ebh7_w9_dossier_index_present`, PASS) plus B5 immutable digest "
            "verification of the committed dossier artifacts and MANIFEST (git objects @ replay HEAD). Skipping "
            "does not hide an environment or dependency defect: it is an explicit, message-bearing "
            "`pytest.skip` on a path-existence check, and the staging directory is never required by any "
            "other gate. Path B: no live-provider test is implied."
        ),
        audit_status=SkipAuditStatus.READY,
        environment_dependent=True,
    ),
)


def skip_disposition_by_node() -> dict[str, SkipDispositionRecord]:
    return {r.node_id: r for r in SKIP_DISPOSITION_REGISTER}
