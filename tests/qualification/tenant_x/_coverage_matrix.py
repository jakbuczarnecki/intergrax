# © Artur Czarnecki. All rights reserved.

"""FRZ-TEN-12 adversarial proof coverage matrix (P0 planning only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ProofCoverageRow:
    concern: str
    existing_executable_evidence: str
    historical_evidence_sha: str
    current_head_replay_needed: bool
    missing_proof: bool
    notes: str


TENANT_PROOF_COVERAGE_MATRIX: Final[tuple[ProofCoverageRow, ...]] = (
    ProofCoverageRow(
        concern="execution",
        existing_executable_evidence="tests/qualification/governance/gov_x2/",
        historical_evidence_sha="e2d7a40b291c658677cc6d981183e20957c7cf0e",
        current_head_replay_needed=True,
        missing_proof=False,
        notes="GOV-X2 local tenant audit — global TENANT-X replay required",
    ),
    ProofCoverageRow(
        concern="state",
        existing_executable_evidence="tests/qualification/state_x/",
        historical_evidence_sha="d5979a531e41f9a45a3a00b0b150765ee34bb0f8",
        current_head_replay_needed=True,
        missing_proof=True,
        notes="Not every checkpoint/restore path historically covered",
    ),
    ProofCoverageRow(
        concern="provider/config",
        existing_executable_evidence="tests/qualification/config_x/",
        historical_evidence_sha="fac9a43f3b128fde63541d2b56ebd8f4a751e8fb",
        current_head_replay_needed=True,
        missing_proof=False,
        notes="CONFIG-X local tenant audit",
    ),
    ProofCoverageRow(
        concern="credentials/secrets",
        existing_executable_evidence="sparse — CredentialBroker architecture only",
        historical_evidence_sha="",
        current_head_replay_needed=True,
        missing_proof=True,
        notes="FRZ-TEN-06 primary TENANT-X child candidate",
    ),
    ProofCoverageRow(
        concern="evidence/trace",
        existing_executable_evidence="tests/qualification/trace_x/",
        historical_evidence_sha="69d8b111e8d2f6717b75d866d3738a0106d00304",
        current_head_replay_needed=True,
        missing_proof=False,
        notes="TRACE-X local tenant isolation; P0 historically PARTIAL before remediation",
    ),
    ProofCoverageRow(
        concern="recovery",
        existing_executable_evidence="tests/qualification/state_x/_r6_recovery_branching_qualification_tests.py",
        historical_evidence_sha="61faf8f317125b996526ceceaf5754b9c28073d6",
        current_head_replay_needed=True,
        missing_proof=True,
        notes="STATE-X-R4 scoped; full recovery continuity open",
    ),
    ProofCoverageRow(
        concern="extensions",
        existing_executable_evidence="tests/qualification/external_contract_compatibility/",
        historical_evidence_sha="234d3c03dce5708766f8e068afbd2ea496132002",
        current_head_replay_needed=True,
        missing_proof=False,
        notes="Provider extension adversarial tenant matrix",
    ),
    ProofCoverageRow(
        concern="missing tenant",
        existing_executable_evidence="tests/qualification/tenant_x/test_tenant_x_p0_fast.py",
        historical_evidence_sha="",
        current_head_replay_needed=False,
        missing_proof=True,
        notes="P0 synthetic probes only — platform-wide replay pending",
    ),
    ProofCoverageRow(
        concern="cross-tenant denial",
        existing_executable_evidence="tests/qualification/config_x/test_config_x_adversarial_cx_gates.py",
        historical_evidence_sha="a59744517b92847f55def1db22826d17d89ee155",
        current_head_replay_needed=True,
        missing_proof=False,
        notes="INT-CONFIG-REAL-X-CERT scoped negatives",
    ),
)
