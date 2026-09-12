"""VPI-IMPLEMENTATION-5C4E3B — bounded representation retrieval quality qualification."""

from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    load_vpi_embedding_configuration,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.versioning import (
    ARENA_QUERY_BENCHMARK_VERSION,
    ARENA_SAMPLE_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.stage_scope import (
    build_stage_evaluation_scope,
)
from platform_proofs.scenarios.verified_product_identification.arena.sampling.arena_sample import (
    ArenaSampleRecord,
)
from platform_proofs.scenarios.verified_product_identification.arena.sampling.dataset_loader import (
    load_arena_sample_records,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.analyzer import (
    create_diagnostic_embedding_port,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_execution_profiles import (
    PRODUCTION_LOCAL_GPU_PROFILE_ID,
    apply_data_pack_build_execution_profile,
    resolve_data_pack_build_execution_profile,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    PROOF_50_RECORD_COUNT,
    VPI_CANONICAL_EMBEDDING_DIMENSION,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.config import (
    load_vpi_embedding_materialization_config,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.bootstrap import (
    ensure_embedding_provider_integrations_registered,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.model_identity import (
    resolve_embedding_model_identity,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_representation.contracts import (
    REPRESENTATION_VARIANT_ORDER,
    BoundedRepresentationQualityReport,
    RepresentationVariant,
    VariantQualityGateResult,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_representation.evaluation import (
    compare_query_rankings,
    evaluate_quality_gate,
    evaluate_variant_rankings,
    select_winning_candidate,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_representation.truncation import (
    ProductRepresentationVariantPort,
)

_BATCH_SIZE = 1
_ARENA_SCAN_ROW_LIMIT = 50_000
_PROOF_IDENTIFIER = (
    "platform_proofs/scenarios/verified_product_identification/arena/evaluation/"
    f"proof-50-vector-retrieval/{ARENA_QUERY_BENCHMARK_VERSION}"
)
_SESSION_ROOT = _REPO_ROOT / ".tmp" / "session" / "vpi-5c4e3b"


def _resolve_runtime_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            profile = resolve_data_pack_build_execution_profile(PRODUCTION_LOCAL_GPU_PROFILE_ID)
            apply_data_pack_build_execution_profile(profile)
            return profile.device
    except ImportError:
        pass
    os.environ["VPI_EMBEDDING_DEVICE"] = "cpu"
    os.environ["VPI_EMBEDDING_PROVIDER_BATCH_SIZE"] = str(_BATCH_SIZE)
    return "cpu"


def _load_proof_corpus_records() -> tuple[ArenaSampleRecord, ...]:
    config = load_vpi_embedding_materialization_config(
        max_records_override=_ARENA_SCAN_ROW_LIMIT,
        artifact_dir_override=_SESSION_ROOT / "artifact-root",
    )
    return load_arena_sample_records(
        config,
        scan_row_limit=_ARENA_SCAN_ROW_LIMIT,
        target_size=PROOF_50_RECORD_COUNT,
    )


def _build_variant_port(
    embedding_port: IntergraxEmbeddingBootstrapAdapter,
) -> ProductRepresentationVariantPort:
    provider = embedding_port._provider  # noqa: SLF001 — qualification-only tokenizer access
    provider._ensure_model()  # noqa: SLF001
    tokenizer = provider._model.tokenizer  # noqa: SLF001

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=True)

    def decode(token_ids: list[int]) -> str:
        return tokenizer.decode(token_ids, skip_special_tokens=False)

    def count_tokens(text: str) -> int:
        return len(encode(text))

    return ProductRepresentationVariantPort(
        encode=encode,
        decode=decode,
        count_tokens=count_tokens,
    )


def _embed_texts_sequential(
    embedding_port: IntergraxEmbeddingBootstrapAdapter,
    texts: tuple[str, ...],
    *,
    expected_dimension: int,
) -> np.ndarray:
    vectors: list[tuple[float, ...]] = []
    for text in texts:
        vectors.append(embedding_port.embed_batch((text,))[0])
        gc.collect()
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.shape != (len(texts), expected_dimension):
        msg = (
            f"embedding matrix shape {matrix.shape} != expected "
            f"({len(texts)}, {expected_dimension})"
        )
        raise RuntimeError(msg)
    return matrix


def run_bounded_representation_quality_qualification(
    *,
    output_dir: Path = _SESSION_ROOT,
) -> BoundedRepresentationQualityReport:
    ensure_embedding_provider_integrations_registered()
    _resolve_runtime_device()

    embedding_configuration = load_vpi_embedding_configuration()
    model_name = embedding_configuration.model
    if model_name is None:
        raise RuntimeError("embedding model is required")
    expected_dimension = embedding_configuration.expected_dimension
    if expected_dimension != VPI_CANONICAL_EMBEDDING_DIMENSION:
        msg = (
            f"expected dimension {VPI_CANONICAL_EMBEDDING_DIMENSION}, "
            f"resolved {expected_dimension}"
        )
        raise RuntimeError(msg)

    records = _load_proof_corpus_records()
    scope = build_stage_evaluation_scope(stage_name="proof-50-vector", records=records)
    canonical_corpus_texts = tuple(record.semantic_text for record in records)
    query_texts = tuple(case.query_text for case in scope.query_cases)

    model_identity = resolve_embedding_model_identity(embedding_configuration.provider, model_name)
    model_load_count = 0
    embedding_port = create_diagnostic_embedding_port(_BATCH_SIZE)
    model_load_count += 1

    variant_metrics: dict[str, object] = {}
    variant_gates: dict[str, object] = {}
    per_query_comparisons: dict[str, tuple[object, ...]] = {}
    gate_by_variant: dict[RepresentationVariant, object] = {}

    try:
        variant_port = _build_variant_port(embedding_port)
        full_ranked: tuple[tuple[int, ...], ...] | None = None
        control_metrics = None

        for variant in REPRESENTATION_VARIANT_ORDER:
            if variant is RepresentationVariant.FULL:
                corpus_texts = canonical_corpus_texts
            else:
                corpus_texts = tuple(
                    variant_port.apply(text, variant) for text in canonical_corpus_texts
                )

            corpus_embeddings = _embed_texts_sequential(
                embedding_port,
                corpus_texts,
                expected_dimension=expected_dimension,
            )
            query_embeddings = _embed_texts_sequential(
                embedding_port,
                query_texts,
                expected_dimension=expected_dimension,
            )
            metrics, ranked = evaluate_variant_rankings(
                scope,
                corpus_embeddings=corpus_embeddings,
                query_embeddings=query_embeddings,
            )
            variant_metrics[variant.value] = metrics

            del corpus_embeddings
            del query_embeddings
            gc.collect()

            if variant is RepresentationVariant.FULL:
                control_metrics = metrics
                full_ranked = ranked
                gate_by_variant[variant] = VariantQualityGateResult(
                    variant=variant,
                    metrics=metrics,
                    passed=True,
                    failure_reasons=(),
                )
                variant_gates[variant.value] = gate_by_variant[variant]
                continue

            if control_metrics is None or full_ranked is None:
                raise RuntimeError("FULL control must be evaluated before bounded variants")

            comparisons = compare_query_rankings(
                scope,
                full_ranked=full_ranked,
                candidate_ranked=ranked,
            )
            per_query_comparisons[variant.value] = comparisons
            gate = evaluate_quality_gate(
                control_metrics,
                metrics,
                variant=variant,
                comparisons=comparisons,
            )
            gate_by_variant[variant] = gate
            variant_gates[variant.value] = gate
    finally:
        embedding_port.close()

    if control_metrics is None:
        raise RuntimeError("FULL control metrics missing")

    winner = select_winning_candidate(gate_by_variant)
    bounded_passed = any(
        gate_by_variant[variant].passed
        for variant in (
            RepresentationVariant.TOKEN_LIMIT_1024,
            RepresentationVariant.TOKEN_LIMIT_768,
            RepresentationVariant.TOKEN_LIMIT_512,
        )
    )
    if not bounded_passed:
        status = "QUALITY_GATE_FAIL"
    else:
        status = "PASS"

    report = BoundedRepresentationQualityReport(
        proof_identifier=_PROOF_IDENTIFIER,
        corpus_record_count=len(records),
        query_count=len(scope.query_cases),
        model_provider=model_identity.provider,
        model_name=model_identity.model,
        model_revision=model_identity.revision,
        model_dimension=expected_dimension,
        model_load_count=model_load_count,
        control_metrics=control_metrics,
        variant_metrics=variant_metrics,
        variant_gates=variant_gates,
        per_query_comparisons=per_query_comparisons,
        winning_candidate=winner.value if winner is not None else "NONE",
        vector_quality_coverage_valid=True,
        status=status,
    )
    _write_report_artifacts(report, output_dir=output_dir)
    return report


def _write_report_artifacts(
    report: BoundedRepresentationQualityReport,
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "bounded-representation-quality-report.json"
    md_path = output_dir / "BOUNDED_REPRESENTATION_QUALITY_REPORT.md"

    payload = _report_to_json(report)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Bounded Representation Retrieval Quality Report",
        "",
        f"STATUS: {report.status}",
        "",
        "## Model",
        f"- provider: {report.model_provider}",
        f"- model: {report.model_name}",
        f"- revision: {report.model_revision}",
        f"- dimension: {report.model_dimension}",
        f"- model_load_count: {report.model_load_count}",
        "",
        "## Proof",
        f"- identifier: {report.proof_identifier}",
        f"- offers: {report.corpus_record_count}",
        f"- queries: {report.query_count}",
        f"- sample_version: {ARENA_SAMPLE_VERSION}",
        f"- benchmark_version: {ARENA_QUERY_BENCHMARK_VERSION}",
        "",
        "## Quality Results",
        "",
        "| metric | FULL | 1024 | 768 | 512 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    metric_names = (
        ("Recall@1", "recall_at_1"),
        ("Recall@5", "recall_at_5"),
        ("MRR@10", "mrr_at_10"),
        ("nDCG@10", "ndcg_at_10"),
    )
    for label, field_name in metric_names:
        row = [label]
        for variant in REPRESENTATION_VARIANT_ORDER:
            metrics = report.variant_metrics[variant.value]
            row.append(f"{getattr(metrics, field_name):.6f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Quality Gate",
            "",
            "| variant | result |",
            "| --- | --- |",
            "| FULL | CONTROL |",
        ]
    )
    for variant in (
        RepresentationVariant.TOKEN_LIMIT_1024,
        RepresentationVariant.TOKEN_LIMIT_768,
        RepresentationVariant.TOKEN_LIMIT_512,
    ):
        gate = report.variant_gates[variant.value]
        lines.append(f"| {variant.token_limit()} | {'PASS' if gate.passed else 'FAIL'} |")

    lines.extend(
        [
            "",
            f"## Winning Candidate: {report.winning_candidate}",
            "",
            "## Performance Evidence (5C4E3A)",
            "",
            "| variant | speedup |",
            "| --- | ---: |",
            "| FULL | 1.00x |",
            "| 1024 | 1.79x |",
            "| 768 | 1.86x |",
            "| 512 | 2.24x |",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_to_json(report: BoundedRepresentationQualityReport) -> dict[str, object]:
    def metrics_payload(metrics: object) -> dict[str, float | int]:
        return {
            "recall_at_1": metrics.recall_at_1,
            "recall_at_5": metrics.recall_at_5,
            "recall_at_10": metrics.recall_at_10,
            "mrr_at_10": metrics.mrr_at_10,
            "ndcg_at_10": metrics.ndcg_at_10,
            "query_count": metrics.query_count,
        }

    comparisons_payload: dict[str, list[dict[str, object]]] = {}
    for variant_key, comparisons in report.per_query_comparisons.items():
        comparisons_payload[variant_key] = [asdict(item) for item in comparisons]

    gates_payload = {
        key: {
            "passed": gate.passed,
            "failure_reasons": list(gate.failure_reasons),
            "metrics": metrics_payload(gate.metrics),
        }
        for key, gate in report.variant_gates.items()
    }

    return {
        "status": report.status,
        "proof_identifier": report.proof_identifier,
        "corpus_record_count": report.corpus_record_count,
        "query_count": report.query_count,
        "model": {
            "provider": report.model_provider,
            "model": report.model_name,
            "revision": report.model_revision,
            "dimension": report.model_dimension,
            "model_load_count": report.model_load_count,
        },
        "control_metrics": metrics_payload(report.control_metrics),
        "variant_metrics": {
            key: metrics_payload(value) for key, value in report.variant_metrics.items()
        },
        "variant_gates": gates_payload,
        "per_query_comparisons": comparisons_payload,
        "winning_candidate": report.winning_candidate,
        "vector_quality_coverage_valid": report.vector_quality_coverage_valid,
        "performance_evidence_5c4e3a": {
            "FULL": "1.00x",
            "1024": "1.79x",
            "768": "1.86x",
            "512": "2.24x",
        },
    }


def main() -> int:
    report = run_bounded_representation_quality_qualification()
    print(f"STATUS: {report.status}")
    print(f"WINNER: {report.winning_candidate}")
    print(f"ARTIFACTS: {_SESSION_ROOT}")
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
