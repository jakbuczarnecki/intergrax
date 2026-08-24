# © Artur Czarnecki. All rights reserved.

"""Deterministic serialization helpers for Platform Proof evidence (PP-REPORT-2)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.proof.intergrax_platform_proof_evidence import (
    PLATFORM_PROOF_EVIDENCE_SCHEMA_VERSION,
    PlatformProofEvidence,
    ProvenanceEvidence,
)

EVIDENCE_FILENAME = "evidence.json"


def evidence_payload_dict(evidence: PlatformProofEvidence) -> dict[str, object]:
    return evidence.model_dump(mode="json")


def serialize_evidence_deterministic(evidence: PlatformProofEvidence) -> str:
    payload = evidence_payload_dict(evidence)
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def compute_evidence_checksum(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def with_evidence_checksum(evidence: PlatformProofEvidence) -> PlatformProofEvidence:
    base_payload = evidence_payload_dict(evidence)
    provenance = dict(base_payload["provenance"])
    provenance["evidence_checksum"] = None
    base_payload["provenance"] = provenance
    checksum = compute_evidence_checksum(base_payload)
    updated_provenance = evidence.provenance.model_copy(
        update={"evidence_checksum": checksum}
    )
    return evidence.model_copy(update={"provenance": updated_provenance})


def resolve_proof_evidence_directory(
    *,
    proof_id: str,
    run_id: str,
    artifact_root: Path | None = None,
    suite_run_id: str | None = None,
) -> Path:
    """Resolve per-proof evidence directory.

    When ``suite_run_id`` is provided, use hierarchical suite layout:
    ``<root>/<suite_run_id>/proofs/<proof_id>/``.

    Otherwise align with current proof-local layout:
    ``<root>/<proof_id>/<run_id>/``.
    """
    root = (artifact_root or Path.cwd() / ".artifacts" / "proof").expanduser().resolve()
    if suite_run_id is not None:
        return root / suite_run_id / "proofs" / proof_id
    return root / proof_id / run_id


def write_evidence_json(
    evidence: PlatformProofEvidence,
    *,
    proof_directory: Path,
    relative_path: str = EVIDENCE_FILENAME,
    include_checksum: bool = True,
) -> Path:
    proof_directory.mkdir(parents=True, exist_ok=True)
    resolved = with_evidence_checksum(evidence) if include_checksum else evidence
    path = proof_directory / relative_path
    path.write_text(serialize_evidence_deterministic(resolved), encoding="utf-8")
    return path


def build_artifact_identity(
    *,
    proof_id: str,
    execution_id: str,
    generated_at: datetime | None = None,
) -> str:
    timestamp = (generated_at or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{PLATFORM_PROOF_EVIDENCE_SCHEMA_VERSION}:{proof_id}:{execution_id}:{timestamp}"
