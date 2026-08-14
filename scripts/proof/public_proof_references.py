# © Artur Czarnecki. All rights reserved.

"""Public documentation proof reference conventions (PUBLIC-PROOF-GATE-2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from scripts.proof.intergrax_proof_contracts import IntergraxProofManifest
from scripts.proof.intergrax_proof_manifest import load_manifest

PUBLIC_GATEWAY_DOCUMENTS: tuple[str, ...] = (
    "README.md",
    "docs/project/proofs/PROOFS.md",
    "applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md",
    "docs/project/capabilities/token_optimization/README.md",
    "applications/local_workspace_application/docs/product/QUICKSTART.md",
    "applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md",
)

PROOF_REFERENCE_LINE_RE = re.compile(r"^\*\*Proofs?:\*\*\s*(.+)$", re.MULTILINE)
PROOF_ID_RE = re.compile(r"`([A-Z][A-Z0-9-]*)`")


@dataclass(frozen=True)
class ProofReference:
    document: str
    line_number: int
    proof_id: str


@dataclass(frozen=True)
class ProofReferenceIssue:
    document: str
    line_number: int
    message: str


@dataclass(frozen=True)
class PublicProofReferenceReport:
    references: tuple[ProofReference, ...]
    issues: tuple[ProofReferenceIssue, ...]
    duplicate_references: tuple[ProofReference, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def extract_proof_references(*, document: str, text: str) -> tuple[ProofReference, ...]:
    references: list[ProofReference] = []
    for match in PROOF_REFERENCE_LINE_RE.finditer(text):
        line_number = text.count("\n", 0, match.start()) + 1
        payload = match.group(1).strip()
        if not payload:
            continue
        ids = PROOF_ID_RE.findall(payload)
        if not ids:
            continue
        for proof_id in ids:
            references.append(
                ProofReference(
                    document=document,
                    line_number=line_number,
                    proof_id=proof_id,
                )
            )
    return tuple(references)


def _find_duplicate_references(
    references: tuple[ProofReference, ...],
) -> tuple[ProofReference, ...]:
    seen: set[tuple[str, str]] = set()
    duplicates: list[ProofReference] = []
    for reference in references:
        key = (reference.document, reference.proof_id)
        if key in seen:
            duplicates.append(reference)
            continue
        seen.add(key)
    return tuple(duplicates)


def validate_public_proof_references(
    *,
    repo_root: Path,
    manifest: IntergraxProofManifest | None = None,
) -> PublicProofReferenceReport:
    loaded_manifest = manifest or load_manifest(repo_root=repo_root)
    entries_by_id = {entry.proof_id: entry for entry in loaded_manifest.entries}

    references: list[ProofReference] = []
    issues: list[ProofReferenceIssue] = []

    for relative_path in PUBLIC_GATEWAY_DOCUMENTS:
        document_path = repo_root / relative_path
        if not document_path.is_file():
            issues.append(
                ProofReferenceIssue(
                    document=relative_path,
                    line_number=0,
                    message="configured public document is missing",
                )
            )
            continue

        text = document_path.read_text(encoding="utf-8")
        for match in PROOF_REFERENCE_LINE_RE.finditer(text):
            line_number = text.count("\n", 0, match.start()) + 1
            payload = match.group(1).strip()
            ids = PROOF_ID_RE.findall(payload)
            if not ids:
                issues.append(
                    ProofReferenceIssue(
                        document=relative_path,
                        line_number=line_number,
                        message="malformed proof reference line",
                    )
                )
                continue
            for proof_id in ids:
                references.append(
                    ProofReference(
                        document=relative_path,
                        line_number=line_number,
                        proof_id=proof_id,
                    )
                )
                entry = entries_by_id.get(proof_id)
                if entry is None:
                    issues.append(
                        ProofReferenceIssue(
                            document=relative_path,
                            line_number=line_number,
                            message=f"unknown proof_id: {proof_id}",
                        )
                    )
                    continue
                if not entry.public_evidence_eligible:
                    issues.append(
                        ProofReferenceIssue(
                            document=relative_path,
                            line_number=line_number,
                            message=(
                                f"proof_id not public-evidence eligible: {proof_id}"
                            ),
                        )
                    )

    ordered_references = tuple(references)
    return PublicProofReferenceReport(
        references=ordered_references,
        issues=tuple(issues),
        duplicate_references=_find_duplicate_references(ordered_references),
    )


def render_report(report: PublicProofReferenceReport) -> str:
    lines: list[str] = []
    if report.ok:
        lines.append("OK public proof references")
    else:
        lines.append("FAIL public proof references")

    lines.append(f"references={len(report.references)}")
    unique_ids = sorted({reference.proof_id for reference in report.references})
    lines.append(f"unique_proof_ids={len(unique_ids)}")

    if report.duplicate_references:
        lines.append("duplicate_references:")
        for duplicate in report.duplicate_references:
            lines.append(
                f"  {duplicate.document}:{duplicate.line_number} {duplicate.proof_id}"
            )

    if report.issues:
        lines.append("issues:")
        for issue in report.issues:
            location = (
                f"{issue.document}:{issue.line_number}"
                if issue.line_number
                else issue.document
            )
            lines.append(f"  {location} {issue.message}")

    return "\n".join(lines)
