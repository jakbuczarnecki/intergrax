"""Bounded semantic representation layer for Data Pack embedding input."""

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.builder import (
    SemanticRepresentationBuilder,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.contracts import (
    RepresentationReductionMetrics,
    RepresentationSectionKind,
    SemanticRepresentationPolicy,
    SemanticRepresentationResult,
    SemanticSection,
    TruncationStrategy,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.metrics import (
    RepresentationPolicyExperimentResult,
    aggregate_reduction_metrics,
    build_representation_reduction_metrics,
    reduction_metrics_for_result,
    run_all_representation_policy_experiments,
    run_representation_policy_experiment,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.policy import (
    BOUNDED_POLICY_COMPACT_CHARACTERS,
    BOUNDED_POLICY_V1_CHARACTERS,
    BOUNDED_POLICY_V1_TOKENS,
    DEFAULT_PRESERVED_FIELDS,
    RepresentationPolicyProfile,
    VPI_SEMANTIC_REPRESENTATION_ENV,
    bounded_policy_compact,
    bounded_policy_v1,
    load_semantic_representation_policy_from_env,
    resolve_semantic_representation_policy,
    token_limit_policy_v1,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.ports import (
    CharacterRatioTokenEstimator,
    TokenEstimatorPort,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.serialization import (
    build_policy_json_document,
    build_reduction_metrics_json_document,
    build_result_json_document,
    parse_policy_json,
    parse_reduction_metrics_json,
    serialize_policy_json,
    serialize_reduction_metrics_json,
    serialize_result_json,
)

__all__ = [
    "BOUNDED_POLICY_COMPACT_CHARACTERS",
    "BOUNDED_POLICY_V1_CHARACTERS",
    "BOUNDED_POLICY_V1_TOKENS",
    "DEFAULT_PRESERVED_FIELDS",
    "CharacterRatioTokenEstimator",
    "RepresentationPolicyExperimentResult",
    "RepresentationPolicyProfile",
    "RepresentationReductionMetrics",
    "RepresentationSectionKind",
    "SemanticRepresentationBuilder",
    "SemanticRepresentationPolicy",
    "SemanticRepresentationResult",
    "SemanticSection",
    "TokenEstimatorPort",
    "TruncationStrategy",
    "VPI_SEMANTIC_REPRESENTATION_ENV",
    "aggregate_reduction_metrics",
    "bounded_policy_compact",
    "bounded_policy_v1",
    "build_policy_json_document",
    "build_representation_reduction_metrics",
    "build_reduction_metrics_json_document",
    "build_result_json_document",
    "load_semantic_representation_policy_from_env",
    "parse_policy_json",
    "parse_reduction_metrics_json",
    "reduction_metrics_for_result",
    "resolve_semantic_representation_policy",
    "run_all_representation_policy_experiments",
    "run_representation_policy_experiment",
    "serialize_policy_json",
    "serialize_reduction_metrics_json",
    "serialize_result_json",
    "token_limit_policy_v1",
]
