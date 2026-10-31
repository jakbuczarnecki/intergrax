# © Artur Czarnecki. All rights reserved.

"""B6-F10 certification model — obligations, evidence, attestation, computed results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EcapGate(StrEnum):
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"


class EvidenceForm(StrEnum):
    EXECUTED = "EXECUTED"
    HISTORICAL_WITH_DELTA = "HISTORICAL_WITH_DELTA"
    NA_WITH_EVIDENCE = "N/A_WITH_EVIDENCE"


class AttestationMode(StrEnum):
    IN_PROCESS = "IN_PROCESS"
    PROFILE_BOUND = "PROFILE_BOUND"


class ResultStatus(StrEnum):
    PASS = "PASS"
    NA_WITH_EVIDENCE = "N/A_WITH_EVIDENCE"
    FAIL = "FAIL"


class GateVerdict(StrEnum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"


class ChangeClass(StrEnum):
    DOCUMENTATION_ONLY = "documentation-only"
    QUALIFICATION_INFRASTRUCTURE = "qualification-infrastructure"
    CONTRACT = "contract"
    OWNER_AUTHORITY = "owner-authority"
    COMPOSITION_CONFIGURATION = "composition-configuration"
    RUNTIME_BEHAVIOR = "runtime-behavior"
    SECURITY_GOVERNANCE_TENANT = "security-governance-tenant"
    STATE_RECOVERY = "state-recovery"
    VENDOR_BACKEND = "vendor-backend"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class NormativeObligation:
    """One member of a closed normative set derived from a canonical source."""

    obligation_id: str
    gate: EcapGate
    subject: str
    canonical_source: str


@dataclass(frozen=True, slots=True)
class EvidenceBinding:
    obligation_id: str
    owner: str
    classification: str
    form: EvidenceForm
    evidence_nodes: tuple[str, ...]
    in_process_checks: tuple[str, ...] = ()
    rationale: str = ""
    historical_acceptance_sha: str = ""
    delta_check: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionAttestation:
    evidence_id: str
    mode: AttestationMode
    executed_sha: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class QualificationResult:
    obligation: NormativeObligation
    binding: EvidenceBinding | None
    status: ResultStatus
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoverageVerdict:
    gate: EcapGate
    verdict: GateVerdict
    obligations: int
    passed: int
    not_applicable: int
    failed: tuple[str, ...]
