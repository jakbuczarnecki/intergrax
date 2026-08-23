# © Artur Czarnecki. All rights reserved.

"""Repository-wide Intergrax proof suite contracts (PUBLIC-PROOF-GATE-1).

Intentionally separate from:
- ``intergrax.contracts.governed_proof`` (runtime governed side-effect profiles)
- ``intergrax.proofs.receipts.contracts.ProofReceipt`` (persisted workload receipts)
- ``applications.local_workspace_application.serving.proof_summary`` (HTTP metadata)
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUITE_RECEIPT_SCHEMA_VERSION = "intergrax.proof_suite_receipt.v1"
PROOF_MANIFEST_SCHEMA_VERSION = "intergrax.proof_manifest.v2"


class ProofProfile(StrEnum):
    QUICK = "quick"
    FULL = "full"
    LIVE = "live"


class ProofStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED_ENVIRONMENT = "BLOCKED_ENVIRONMENT"
    BLOCKED_CONFIGURATION = "BLOCKED_CONFIGURATION"
    SKIPPED_PLATFORM = "SKIPPED_PLATFORM"
    SKIPPED_PROFILE = "SKIPPED_PROFILE"


class SuiteOverallStatus(StrEnum):
    PASS = "PASS"
    PASS_WITH_BLOCKED = "PASS_WITH_BLOCKED"
    DRY_RUN = "DRY_RUN"
    FAIL = "FAIL"
    FAIL_MANIFEST = "FAIL_MANIFEST"


class ProofSafetyClass(StrEnum):
    LOCAL_READ_ONLY = "LOCAL_READ_ONLY"
    LOCAL_MUTATING = "LOCAL_MUTATING"
    EXTERNAL_READ_ONLY = "EXTERNAL_READ_ONLY"
    EXTERNAL_MUTATING = "EXTERNAL_MUTATING"


class EnvRequirementKind(StrEnum):
    ENV_PRESENT = "ENV_PRESENT"
    COMMAND_AVAILABLE = "COMMAND_AVAILABLE"
    DOCKER_AVAILABLE = "DOCKER_AVAILABLE"


class EnvRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: EnvRequirementKind
    name: str = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("name must be non-empty")
        return normalized


class ProofArgvCommand(BaseModel):
    """Structured subprocess invocation — never a shell string."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    executable: str = Field(min_length=1)
    argv: tuple[str, ...]

    @field_validator("executable")
    @classmethod
    def _strip_executable(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("executable must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _argv_non_empty(self) -> ProofArgvCommand:
        if not self.argv:
            raise ValueError("argv must be non-empty")
        return self


class ProofManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proof_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    profiles: frozenset[ProofProfile]
    proof_kind: str = Field(min_length=1)
    command: ProofArgvCommand
    platform_requirements: frozenset[str] = frozenset()
    environment_requirements: tuple[EnvRequirement, ...] = ()
    external_provider: str | None = None
    timeout_seconds: int = Field(default=600, ge=1)
    safety_class: ProofSafetyClass
    public_evidence_eligible: bool = False

    @field_validator("proof_id", "title", "proof_kind")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _validate_profiles(self) -> ProofManifestEntry:
        if not self.profiles:
            raise ValueError("profiles must be non-empty")
        return self


class IntergraxProofManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["intergrax.proof_manifest.v2"] = PROOF_MANIFEST_SCHEMA_VERSION
    entries: tuple[ProofManifestEntry, ...]

    @model_validator(mode="after")
    def _unique_proof_ids(self) -> IntergraxProofManifest:
        seen: set[str] = set()
        for entry in self.entries:
            if entry.proof_id in seen:
                raise ValueError(f"duplicate proof_id: {entry.proof_id}")
            seen.add(entry.proof_id)
        return self


class EnvRequirementResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: EnvRequirementKind
    name: str
    satisfied: bool


class EvidenceVerificationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    MISSING = "MISSING"
    INVALID = "INVALID"


class ArtifactVerificationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


class ProofRunResult(BaseModel):
    """Durable-safe per-proof result — no untrusted child process output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proof_id: str
    status: ProofStatus
    duration_seconds: float = Field(ge=0.0)
    exit_code: int | None = None
    diagnostic_summary: str = ""
    environment_requirements: tuple[EnvRequirementResult, ...] = ()
    evidence_verification_status: EvidenceVerificationStatus | None = None
    evidence_path: str | None = None
    artifact_verification_status: ArtifactVerificationStatus | None = None
    artifact_diagnostic: str = ""


class SuiteReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["intergrax.proof_suite_receipt.v1"] = (
        SUITE_RECEIPT_SCHEMA_VERSION
    )
    suite_run_id: str
    started_at: datetime
    completed_at: datetime
    git_commit_sha: str
    git_dirty: bool
    profile: ProofProfile
    platform: str
    python_version: str
    overall_status: SuiteOverallStatus
    results: tuple[ProofRunResult, ...]
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
