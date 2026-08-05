# © Artur Czarnecki. All rights reserved.

"""Token Optimization proof infrastructure and live-proof runners."""

from intergrax.runtime.token_optimization.proofs.config import (
    load_universal_token_optimization_proof_config,
)
from intergrax.runtime.token_optimization.proofs.contracts import (
    ProofAdapterConfig,
    ProofArtifactError,
    ProofArtifactRef,
    ProofCaseInput,
    ProofCompositionError,
    ProofConfigurationError,
    ProofExecutionError,
    ProofMeasurement,
    ProofOutputConfig,
    ProofPipelineConfig,
    ProofPipelineEvidence,
    ProofPrefixIdentityEvidence,
    ProofProtectedRegionEvidence,
    ProofProviderUnavailableError,
    ProofRouterConfig,
    ProofRouterEvidence,
    UniversalProofArtifactManifest,
    UniversalProofCaseResult,
    UniversalProofEnvironmentSummary,
    UniversalProofRunResult,
    UniversalTokenOptimizationProofConfig,
)
from intergrax.runtime.token_optimization.proofs.runner import (
    UniversalTokenOptimizationProofRunner,
    write_universal_proof_artifacts,
)

from intergrax.runtime.token_optimization.proofs.vllm_prefix_cache_live import (
    VllmPrefixCacheLiveProofConfig,
    build_default_config,
    run_vllm_prefix_cache_live_proof,
)
from intergrax.runtime.token_optimization.proofs.vllm_prefix_cache_report import (
    VllmPrefixCacheLiveProofAggregateResult,
)

__all__ = [
    "ProofAdapterConfig",
    "ProofArtifactError",
    "ProofArtifactRef",
    "ProofCaseInput",
    "ProofCompositionError",
    "ProofConfigurationError",
    "ProofExecutionError",
    "ProofMeasurement",
    "ProofOutputConfig",
    "ProofPipelineConfig",
    "ProofPipelineEvidence",
    "ProofPrefixIdentityEvidence",
    "ProofProtectedRegionEvidence",
    "ProofProviderUnavailableError",
    "ProofRouterConfig",
    "ProofRouterEvidence",
    "VllmPrefixCacheLiveProofConfig",
    "VllmPrefixCacheLiveProofAggregateResult",
    "UniversalProofArtifactManifest",
    "UniversalProofCaseResult",
    "UniversalProofEnvironmentSummary",
    "UniversalProofRunResult",
    "UniversalTokenOptimizationProofConfig",
    "UniversalTokenOptimizationProofRunner",
    "build_default_config",
    "load_universal_token_optimization_proof_config",
    "run_vllm_prefix_cache_live_proof",
    "write_universal_proof_artifacts",
]
