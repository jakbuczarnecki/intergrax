"""Enterprise embedding diagnosis and optimization qualification layer."""

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.analyzer import (
    create_diagnostic_embedding_port,
    derive_semantic_texts,
    load_diagnostic_dataset_sample,
    measure_record_representations,
    resolve_token_counter,
    run_embedding_diagnostics,
    run_embedding_experiment,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.contracts import (
    BATCH_EXPERIMENT_SIZES,
    DIAGNOSTIC_MAX_RECORD_LIMIT,
    BatchExperimentResult,
    BatchLatencyMeasurement,
    CudaEnvironmentSnapshot,
    EmbeddingBottleneckCase,
    EmbeddingDiagnosticClassification,
    EmbeddingDiagnosticReport,
    EmbeddingPerformanceMetrics,
    RecordRepresentationMeasurement,
    TokenDistributionReport,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.metrics import (
    batch_sizes_for_experiments,
    build_embedding_performance_metrics,
    build_token_distribution_report,
    classify_embedding_bottleneck,
    percentile,
    validate_record_limit,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.sinks import (
    write_embedding_diagnostic_report,
)

__all__ = [
    "BATCH_EXPERIMENT_SIZES",
    "DIAGNOSTIC_MAX_RECORD_LIMIT",
    "BatchExperimentResult",
    "BatchLatencyMeasurement",
    "CudaEnvironmentSnapshot",
    "EmbeddingBottleneckCase",
    "EmbeddingDiagnosticClassification",
    "EmbeddingDiagnosticReport",
    "EmbeddingPerformanceMetrics",
    "RecordRepresentationMeasurement",
    "TokenDistributionReport",
    "batch_sizes_for_experiments",
    "build_embedding_performance_metrics",
    "build_token_distribution_report",
    "classify_embedding_bottleneck",
    "create_diagnostic_embedding_port",
    "derive_semantic_texts",
    "load_diagnostic_dataset_sample",
    "measure_record_representations",
    "percentile",
    "resolve_token_counter",
    "run_embedding_diagnostics",
    "run_embedding_experiment",
    "validate_record_limit",
    "write_embedding_diagnostic_report",
]
