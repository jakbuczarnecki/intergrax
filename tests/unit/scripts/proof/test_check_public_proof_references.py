# Â© Artur Czarnecki. All rights reserved.

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.proof.intergrax_proof_contracts import (
    IntergraxProofManifest,
    ProofArgvCommand,
    ProofManifestEntry,
    ProofProfile,
    ProofSafetyClass,
)
from scripts.proof.public_proof_references import (
    PUBLIC_GATEWAY_DOCUMENTS,
    ProofReference,
    ProofReferenceIssue,
    PublicProofReferenceReport,
    extract_proof_references,
    render_report,
    validate_public_proof_references,
)


def _write_gateway_documents(
    root: Path,
    *,
    contents: dict[str, str] | None = None,
) -> None:
    payload = contents or {}
    for relative_path in PUBLIC_GATEWAY_DOCUMENTS:
        path = root / relative_path
        if relative_path in payload:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload[relative_path], encoding="utf-8")
            continue
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("No proof references here.\n", encoding="utf-8")


def _entry(
    proof_id: str,
    *,
    public_evidence_eligible: bool = False,
) -> ProofManifestEntry:
    return ProofManifestEntry(
        proof_id=proof_id,
        title=proof_id,
        profiles=frozenset({ProofProfile.QUICK}),
        proof_kind="test",
        command=ProofArgvCommand(executable="python", argv=("-c", "print('ok')")),
        safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
        public_evidence_eligible=public_evidence_eligible,
    )


def test_extract_valid_single_reference() -> None:
    references = extract_proof_references(
        document="README.md",
        text="Intro\n\n**Proof:** `RUNTIME-TOKEN-OPTIMIZATION-OFFLINE`\n",
    )
    assert references == (
        ProofReference(
            document="README.md",
            line_number=3,
            proof_id="RUNTIME-TOKEN-OPTIMIZATION-OFFLINE",
        ),
    )


def test_extract_multiple_proof_references() -> None:
    references = extract_proof_references(
        document="README.md",
        text=(
            "**Proofs:** `LKW-CORE-PLATFORM-WINDOWS`, "
            "`LKW-CORE-PLATFORM-LINUX`, `LKW-CORE-PLATFORM-MACOS`\n"
        ),
    )
    assert [reference.proof_id for reference in references] == [
        "LKW-CORE-PLATFORM-WINDOWS",
        "LKW-CORE-PLATFORM-LINUX",
        "LKW-CORE-PLATFORM-MACOS",
    ]


def test_unknown_proof_id_fails(tmp_path: Path) -> None:
    _write_gateway_documents(
        tmp_path,
        contents={"README.md": "**Proof:** `UNKNOWN-PROOF`\n"},
    )
    manifest = IntergraxProofManifest(entries=(_entry("KNOWN-PROOF"),))
    report = validate_public_proof_references(
        repo_root=tmp_path,
        manifest=manifest,
    )
    assert not report.ok
    assert any("unknown proof_id" in issue.message for issue in report.issues)


def test_malformed_reference_fails(tmp_path: Path) -> None:
    _write_gateway_documents(
        tmp_path,
        contents={"README.md": "**Proof:** not-backticks\n"},
    )
    manifest = IntergraxProofManifest(entries=(_entry("KNOWN-PROOF"),))
    report = validate_public_proof_references(
        repo_root=tmp_path,
        manifest=manifest,
    )
    assert not report.ok
    assert any("malformed proof reference line" in issue.message for issue in report.issues)


def test_non_public_evidence_proof_fails(tmp_path: Path) -> None:
    _write_gateway_documents(
        tmp_path,
        contents={"README.md": "**Proof:** `PRIVATE-PROOF`\n"},
    )
    manifest = IntergraxProofManifest(
        entries=(_entry("PRIVATE-PROOF", public_evidence_eligible=False),)
    )
    report = validate_public_proof_references(
        repo_root=tmp_path,
        manifest=manifest,
    )
    assert not report.ok
    assert any("not public-evidence eligible" in issue.message for issue in report.issues)


def test_duplicate_use_across_docs_allowed(tmp_path: Path) -> None:
    _write_gateway_documents(
        tmp_path,
        contents={
            "README.md": "**Proof:** `SHARED-PROOF`\n",
            "docs/project/proofs/PROOFS.md": "**Proof:** `SHARED-PROOF`\n",
        },
    )

    manifest = IntergraxProofManifest(
        entries=(_entry("SHARED-PROOF", public_evidence_eligible=True),)
    )
    report = validate_public_proof_references(
        repo_root=tmp_path,
        manifest=manifest,
    )
    assert report.ok
    assert len(report.references) == 2
    assert report.duplicate_references == ()


def test_duplicate_in_same_document_reported_not_failed(tmp_path: Path) -> None:
    _write_gateway_documents(
        tmp_path,
        contents={
            "README.md": "**Proof:** `SHARED-PROOF`\n\n**Proof:** `SHARED-PROOF`\n",
        },
    )
    manifest = IntergraxProofManifest(
        entries=(_entry("SHARED-PROOF", public_evidence_eligible=True),)
    )
    report = validate_public_proof_references(
        repo_root=tmp_path,
        manifest=manifest,
    )
    assert report.ok
    assert len(report.duplicate_references) == 1


def test_missing_configured_document_fails(tmp_path: Path) -> None:
    report = validate_public_proof_references(
        repo_root=tmp_path,
        manifest=IntergraxProofManifest(entries=()),
    )
    assert not report.ok
    assert any("configured public document is missing" in issue.message for issue in report.issues)


def test_render_report_is_deterministic() -> None:
    report = PublicProofReferenceReport(
        references=(
            ProofReference(
                document="README.md",
                line_number=1,
                proof_id="RUNTIME-TOKEN-OPTIMIZATION-OFFLINE",
            ),
        ),
        issues=(
            ProofReferenceIssue(
                document="README.md",
                line_number=2,
                message="unknown proof_id: MISSING",
            ),
        ),
        duplicate_references=(),
    )
    first = render_report(report)
    second = render_report(report)
    assert first == second
    assert "FAIL public proof references" in first
    assert "unique_proof_ids=1" in first


def test_validator_does_not_process_secrets(tmp_path: Path) -> None:
    secret_name = "INTERGRAX_SLACK_BOT_TOKEN"
    _write_gateway_documents(
        tmp_path,
        contents={
            "README.md": (
                f"**Proof:** `TOKEN-PROOF`\nSecret mention: {secret_name}\n"
            ),
        },
    )
    manifest = IntergraxProofManifest(
        entries=(_entry("TOKEN-PROOF", public_evidence_eligible=True),)
    )
    report = validate_public_proof_references(
        repo_root=tmp_path,
        manifest=manifest,
    )
    rendered = render_report(report)
    assert secret_name in (tmp_path / "README.md").read_text(encoding="utf-8")
    assert secret_name not in rendered


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


_PRODUCT_QUICKSTART_PROOF_IDS = (
    "LKW-PRODUCT-QUICKSTART-WINDOWS",
    "LKW-PRODUCT-QUICKSTART-LINUX",
    "LKW-PRODUCT-QUICKSTART-MACOS",
)
_CORE_PLATFORM_PROOF_IDS = (
    "LKW-CORE-PLATFORM-WINDOWS",
    "LKW-CORE-PLATFORM-LINUX",
    "LKW-CORE-PLATFORM-MACOS",
)


def _proof_ids_on_line(text: str, *, label: str) -> set[str]:
    match = re.search(
        rf"^\*\*{re.escape(label)}:\*\*\s*(.+)$",
        text,
        re.MULTILINE,
    )
    assert match is not None, f"Missing {label} line"
    return set(re.findall(r"`([A-Z][A-Z0-9-]*)`", match.group(1)))


def test_public_claim_proof_mappings(repo_root: Path) -> None:
    quickstart = (repo_root / "applications/local_workspace_application/docs/product/QUICKSTART.md").read_text(
        encoding="utf-8"
    )
    tour = (repo_root / "applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md").read_text(
        encoding="utf-8"
    )
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    proofs = (repo_root / "docs/project/proofs/PROOFS.md").read_text(encoding="utf-8")

    quickstart_ids = _proof_ids_on_line(quickstart, label="Proofs")
    assert quickstart_ids == set(_PRODUCT_QUICKSTART_PROOF_IDS)
    assert not quickstart_ids.intersection(_CORE_PLATFORM_PROOF_IDS)

    tour_ids = _proof_ids_on_line(tour, label="Proofs")
    assert tour_ids == set(_PRODUCT_QUICKSTART_PROOF_IDS)
    assert not tour_ids.intersection(_CORE_PLATFORM_PROOF_IDS)

    assert "LKW-HYBRID-ASK-INDEXED" in readme
    assert "LKW-HYBRID-ASK-INDEXED" in proofs
    assert "LKW-WEB-URL-INDEXED-ASK" in proofs

    hybrid_section = readme.split("### What is boundedly proven today", 1)[1]
    assert "LKW-HYBRID-ASK-INDEXED" in hybrid_section
    assert not any(
        core_id in hybrid_section.split("### LKW routes", 1)[0]
        for core_id in _CORE_PLATFORM_PROOF_IDS
    )


def test_real_public_documents_validate(repo_root: Path) -> None:
    report = validate_public_proof_references(repo_root=repo_root)
    assert report.ok, render_report(report)
