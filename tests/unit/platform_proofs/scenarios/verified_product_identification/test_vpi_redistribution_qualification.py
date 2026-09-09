"""VPI Data Pack redistribution qualification gate tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.proof_data.descriptor import PublicationStatus
from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageDescriptorBuildError,
    VpiRedistributionQualificationError,
)
from platform_proofs.scenarios.verified_product_identification.data_package.publication import (
    assert_publication_permitted_for_qualification,
    effective_publication_status,
    is_public_redistribution_permitted,
)
from platform_proofs.scenarios.verified_product_identification.data_package.redistribution_qualification import (
    AssetRedistributionDecision,
    EmbeddingGateStatus,
    EmbeddingRedistributionGates,
    RedistributionAssetStatus,
    RedistributionAssetType,
    RedistributionQualificationRecord,
    SourceEvidenceRecord,
    assert_public_redistribution_permitted,
    can_publish,
    dump_redistribution_qualification,
    load_default_redistribution_qualification,
    load_redistribution_qualification,
    resolve_publication_status,
)

pytestmark = pytest.mark.unit

_REVIEW_PATH = (
    Path(__file__).resolve().parents[5]
    / "platform_proofs"
    / "scenarios"
    / "verified_product_identification"
    / "data_package"
    / "v1"
    / "redistribution-review.json"
)


def _baseline_qualification() -> RedistributionQualificationRecord:
    return load_redistribution_qualification(_REVIEW_PATH)


def _with_asset_status(
    qualification: RedistributionQualificationRecord,
    asset_type: RedistributionAssetType,
    status: RedistributionAssetStatus,
) -> RedistributionQualificationRecord:
    updated: list[AssetRedistributionDecision] = []
    for decision in qualification.asset_decisions:
        if decision.asset_type is asset_type:
            updated.append(
                decision.model_copy(update={"status": status, "blocking_uncertainty": None})
            )
        else:
            updated.append(decision)
    return qualification.model_copy(update={"asset_decisions": tuple(updated)})


def _all_components_approved(
    qualification: RedistributionQualificationRecord,
) -> RedistributionQualificationRecord:
    current = qualification
    for asset_type in (
        RedistributionAssetType.VPI_RELATIONAL_PARQUET,
        RedistributionAssetType.VPI_EMBEDDING_PARQUET,
        RedistributionAssetType.PACKAGE_METADATA,
        RedistributionAssetType.VPI_COMBINED_DATA_PACK,
    ):
        current = _with_asset_status(
            current,
            asset_type,
            RedistributionAssetStatus.PUBLIC_REDISTRIBUTION_APPROVED,
        )
    return current


def test_unresolved_wdc_rights_blocks_public_package() -> None:
    qualification = _baseline_qualification()
    assert not can_publish(RedistributionAssetType.VPI_COMBINED_DATA_PACK, qualification)
    assert resolve_publication_status(qualification) is PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED


def test_model_mit_alone_does_not_approve_combined_package() -> None:
    qualification = _baseline_qualification()
    assert qualification.embedding_gates.model_license_gate is EmbeddingGateStatus.PASSED
    assert qualification.embedding_gates.input_data_rights_gate is EmbeddingGateStatus.FAILED
    assert not can_publish(RedistributionAssetType.VPI_COMBINED_DATA_PACK, qualification)


def test_relational_blocked_embedding_approved_does_not_approve_combined() -> None:
    qualification = _baseline_qualification()
    qualification = _with_asset_status(
        qualification,
        RedistributionAssetType.VPI_RELATIONAL_PARQUET,
        RedistributionAssetStatus.PUBLIC_REDISTRIBUTION_BLOCKED,
    )
    qualification = _with_asset_status(
        qualification,
        RedistributionAssetType.VPI_EMBEDDING_PARQUET,
        RedistributionAssetStatus.PUBLIC_REDISTRIBUTION_APPROVED,
    )
    assert can_publish(RedistributionAssetType.VPI_EMBEDDING_PARQUET, qualification)
    assert not can_publish(RedistributionAssetType.VPI_COMBINED_DATA_PACK, qualification)


def test_all_required_asset_gates_approved_allows_combined_package() -> None:
    qualification = _all_components_approved(_baseline_qualification())
    assert can_publish(RedistributionAssetType.VPI_COMBINED_DATA_PACK, qualification)
    assert resolve_publication_status(qualification) is PublicationStatus.PUBLICATION_APPROVED


def test_missing_authoritative_source_marks_unresolved(tmp_path: Path) -> None:
    payload = json.loads(_REVIEW_PATH.read_text(encoding="utf-8"))
    payload["evidence_records"] = []
    path = tmp_path / "review.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VpiRedistributionQualificationError):
        load_redistribution_qualification(path)


def test_tier3_only_evidence_insufficient_for_approval() -> None:
    qualification = _baseline_qualification()
    assert not any(
        "blog" in record.url or "stackoverflow" in record.url
        for record in qualification.evidence_records
    )
    assert not can_publish(RedistributionAssetType.VPI_RELATIONAL_PARQUET, qualification)


def test_publication_gate_fails_closed() -> None:
    qualification = _baseline_qualification()
    with pytest.raises(VpiDataPackageDescriptorBuildError):
        assert_public_redistribution_permitted(qualification)
    with pytest.raises(VpiDataPackageDescriptorBuildError):
        assert_publication_permitted_for_qualification(qualification)
    assert effective_publication_status(qualification) is PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED
    assert not is_public_redistribution_permitted(qualification)


def test_attribution_requirements_propagated() -> None:
    qualification = _baseline_qualification()
    relational = qualification.decision_for(RedistributionAssetType.VPI_RELATIONAL_PARQUET)
    assert relational.attribution is not None
    assert relational.attribution.source_dataset == "offers_corpus_all_v2_non_norm"
    assert qualification.recommended_attribution is not None
    assert qualification.recommended_attribution.model_name == "BAAI/bge-m3"
    assert qualification.recommended_attribution.legally_required is False


def test_package_location_does_not_affect_legal_status(tmp_path: Path) -> None:
    qualification = load_redistribution_qualification(_REVIEW_PATH)
    copied = tmp_path / "mirror" / "redistribution-review.json"
    dump_redistribution_qualification(qualification, copied)
    reloaded = load_redistribution_qualification(copied)
    assert (
        resolve_publication_status(qualification)
        is resolve_publication_status(reloaded)
    )


def test_model_weights_not_part_of_package() -> None:
    qualification = _baseline_qualification()
    for decision in qualification.asset_decisions:
        assert decision.model_weights_distributed is False
    embedding = qualification.decision_for(RedistributionAssetType.VPI_EMBEDDING_PARQUET)
    assert embedding.attribution is not None
    assert "model weights" in embedding.attribution.generation_notice.lower()


def test_machine_readable_qualification_round_trip_deterministic(tmp_path: Path) -> None:
    qualification = _baseline_qualification()
    output_path = tmp_path / "roundtrip.json"
    dump_redistribution_qualification(qualification, output_path)
    first = output_path.read_text(encoding="utf-8")
    reloaded = load_redistribution_qualification(output_path)
    dump_redistribution_qualification(reloaded, output_path)
    second = output_path.read_text(encoding="utf-8")
    assert first == second


def test_unknown_asset_type_rejected() -> None:
    qualification = RedistributionQualificationRecord(
        review_id="partial-review",
        reviewed_at="2026-09-09",
        reviewer="test",
        processed_dataset_sha256="fc1268a9c4b3e37325919cd127912a67db0a0b6d1943229a2026d9fedff1d998",
        source_dataset_identifier="offers_corpus_all_v2_non_norm",
        third_party_content_rights_resolved=False,
        evidence_records=(
            SourceEvidenceRecord(
                source_id="evidence-1",
                authority="test",
                title="test",
                url="https://example.test/evidence",
                accessed_at="2026-09-09",
                evidence_scope="test",
                short_finding="test",
            ),
        ),
        asset_decisions=(
            AssetRedistributionDecision(
                asset_type=RedistributionAssetType.RAW_WDC_CORPUS,
                status=RedistributionAssetStatus.INTERNAL_USE_ONLY,
                evidence_source_ids=("evidence-1",),
                recommended_publication_action="exclude",
            ),
        ),
        embedding_gates=EmbeddingRedistributionGates(
            model_license_gate=EmbeddingGateStatus.PASSED,
            input_data_rights_gate=EmbeddingGateStatus.FAILED,
            output_restriction_gate=EmbeddingGateStatus.UNCLEAR,
        ),
        embeddings_only_fallback_viable="UNRESOLVED",
        legal_escalation_required=True,
    )
    with pytest.raises(VpiRedistributionQualificationError):
        qualification.decision_for(RedistributionAssetType.VPI_EMBEDDING_PARQUET)


def test_invalid_status_rejected(tmp_path: Path) -> None:
    payload = json.loads(_REVIEW_PATH.read_text(encoding="utf-8"))
    payload["asset_decisions"][0]["status"] = "NOT_A_REAL_STATUS"
    path = tmp_path / "invalid-status.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VpiRedistributionQualificationError):
        load_redistribution_qualification(path)


def test_no_database_or_provider_dependency() -> None:
    qualification = load_default_redistribution_qualification()
    assert qualification.source_dataset_identifier == "offers_corpus_all_v2_non_norm"


def test_no_network_required_for_unit_tests() -> None:
    qualification = _baseline_qualification()
    assert qualification.evidence_records
    assert all(record.accessed_at == "2026-09-09" for record in qualification.evidence_records)


def test_raw_wdc_corpus_remains_excluded() -> None:
    qualification = _baseline_qualification()
    raw = qualification.decision_for(RedistributionAssetType.RAW_WDC_CORPUS)
    assert raw.status is RedistributionAssetStatus.INTERNAL_USE_ONLY
    assert not can_publish(RedistributionAssetType.RAW_WDC_CORPUS, qualification)


def test_default_review_matches_processed_dataset_checksum() -> None:
    qualification = load_default_redistribution_qualification()
    assert (
        qualification.processed_dataset_sha256
        == "fc1268a9c4b3e37325919cd127912a67db0a0b6d1943229a2026d9fedff1d998"
    )


def test_combined_explicit_approval_requires_component_alignment() -> None:
    qualification = _baseline_qualification()
    qualification = _with_asset_status(
        qualification,
        RedistributionAssetType.VPI_COMBINED_DATA_PACK,
        RedistributionAssetStatus.PUBLIC_REDISTRIBUTION_APPROVED,
    )
    assert qualification.resolved_combined_status() is RedistributionAssetStatus.REDISTRIBUTION_REVIEW_REQUIRED
