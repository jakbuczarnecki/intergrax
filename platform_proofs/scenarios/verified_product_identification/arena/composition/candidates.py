"""Arena candidate registry — typed specifications at composition boundary."""

from __future__ import annotations

from intergrax.rag.embedding.registry.profile import EmbeddingProfile

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.input_policies import (
    BGE_M3_INPUT_POLICY_VERSION,
    E5_INSTRUCT_INPUT_POLICY_VERSION,
    NOMIC_V2_INPUT_POLICY_VERSION,
    QWEN3_EMBEDDING_INPUT_POLICY_VERSION,
    bge_m3_input_policy,
    e5_instruct_input_policy,
    nomic_v2_input_policy,
    qwen3_embedding_input_policy,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.candidates import (
    EmbeddingArenaCandidate,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    EmbeddingArenaCandidateEligibility,
    EmbeddingLicenseClassification,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.input_policy import (
    EmbeddingInputPolicyRef,
)

BASELINE_CANDIDATE_ID = "bge-m3"
QWEN_CANDIDATE_ID = "qwen3-0.6b"
NOMIC_CANDIDATE_ID = "nomic-v2-moe"
E5_CANDIDATE_ID = "e5-large-instruct"

DEFAULT_STAGE_A_RECORDS = 100
DEFAULT_STAGE_B_RECORDS = 500
DEFAULT_STAGE_C_RECORDS = 1000
DEFAULT_BATCH_CANDIDATES = (8, 16, 32, 64)
BASELINE_KNOWN_BATCH_SIZE = 16
BASELINE_KNOWN_THROUGHPUT_RPS = 10.4
FULL_DATASET_RECORD_COUNT = 3_770_377


def _policy_ref_from_transformation(policy_id: str) -> EmbeddingInputPolicyRef:
    if policy_id == "bge-m3":
        transformation = bge_m3_input_policy()
    elif policy_id == "qwen3-embedding-0.6b":
        transformation = qwen3_embedding_input_policy()
    elif policy_id == "nomic-embed-text-v2-moe":
        transformation = nomic_v2_input_policy()
    elif policy_id == "e5-large-instruct":
        transformation = e5_instruct_input_policy()
    else:
        msg = f"unsupported policy id: {policy_id}"
        raise ValueError(msg)
    return transformation.policy_ref


def build_default_arena_candidates(
    *,
    include_e5_control: bool = False,
) -> tuple[EmbeddingArenaCandidate, ...]:
    baseline_query = _policy_ref_from_transformation("bge-m3")
    baseline_document = _policy_ref_from_transformation("bge-m3")
    qwen_query = _policy_ref_from_transformation("qwen3-embedding-0.6b")
    qwen_document = _policy_ref_from_transformation("qwen3-embedding-0.6b")
    nomic_query = _policy_ref_from_transformation("nomic-embed-text-v2-moe")
    nomic_document = _policy_ref_from_transformation("nomic-embed-text-v2-moe")

    candidates: list[EmbeddingArenaCandidate] = [
        EmbeddingArenaCandidate(
            candidate_id=BASELINE_CANDIDATE_ID,
            provider="hf",
            model="BAAI/bge-m3",
            expected_dimension=1024,
            license_classification=EmbeddingLicenseClassification.ELIGIBLE_COMMERCIAL,
            license_identifier="MIT",
            license_reference="https://huggingface.co/BAAI/bge-m3",
            license_reason="Model card lists MIT license",
            query_instruction_policy=baseline_query,
            document_instruction_policy=baseline_document,
            semantic_input_policy_id=BGE_M3_INPUT_POLICY_VERSION,
            max_sequence_length=8192,
            trust_remote_code_required=False,
            normalization_expected=True,
            eligibility_status=EmbeddingArenaCandidateEligibility.ELIGIBLE,
            is_baseline=True,
            fixed_provider_batch_size=BASELINE_KNOWN_BATCH_SIZE,
        ),
        EmbeddingArenaCandidate(
            candidate_id=QWEN_CANDIDATE_ID,
            provider="hf",
            model="Qwen/Qwen3-Embedding-0.6B",
            expected_dimension=1024,
            license_classification=EmbeddingLicenseClassification.ELIGIBLE_COMMERCIAL,
            license_identifier="Apache-2.0",
            license_reference="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
            license_reason="Model card lists Apache-2.0 license",
            query_instruction_policy=qwen_query,
            document_instruction_policy=qwen_document,
            semantic_input_policy_id=QWEN3_EMBEDDING_INPUT_POLICY_VERSION,
            max_sequence_length=32768,
            trust_remote_code_required=False,
            normalization_expected=True,
            eligibility_status=EmbeddingArenaCandidateEligibility.ELIGIBLE,
            is_baseline=False,
            fixed_provider_batch_size=None,
        ),
        EmbeddingArenaCandidate(
            candidate_id=NOMIC_CANDIDATE_ID,
            provider="hf",
            model="nomic-ai/nomic-embed-text-v2-moe",
            expected_dimension=768,
            license_classification=EmbeddingLicenseClassification.ELIGIBLE_COMMERCIAL,
            license_identifier="Apache-2.0",
            license_reference="https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe",
            license_reason="Model card lists Apache-2.0 license",
            query_instruction_policy=nomic_query,
            document_instruction_policy=nomic_document,
            semantic_input_policy_id=NOMIC_V2_INPUT_POLICY_VERSION,
            max_sequence_length=512,
            trust_remote_code_required=True,
            normalization_expected=True,
            eligibility_status=EmbeddingArenaCandidateEligibility.ELIGIBLE,
            is_baseline=False,
            fixed_provider_batch_size=None,
        ),
    ]

    if include_e5_control:
        e5_query = _policy_ref_from_transformation("e5-large-instruct")
        e5_document = _policy_ref_from_transformation("e5-large-instruct")
        candidates.append(
            EmbeddingArenaCandidate(
                candidate_id=E5_CANDIDATE_ID,
                provider="hf",
                model="intfloat/multilingual-e5-large-instruct",
                expected_dimension=1024,
                license_classification=EmbeddingLicenseClassification.ELIGIBLE_COMMERCIAL,
                license_identifier="MIT",
                license_reference="https://huggingface.co/intfloat/multilingual-e5-large-instruct",
                license_reason="Model card lists MIT license",
                query_instruction_policy=e5_query,
                document_instruction_policy=e5_document,
                semantic_input_policy_id=E5_INSTRUCT_INPUT_POLICY_VERSION,
                max_sequence_length=512,
                trust_remote_code_required=False,
                normalization_expected=True,
                eligibility_status=EmbeddingArenaCandidateEligibility.OPTIONAL_CONTROL,
                is_baseline=False,
                fixed_provider_batch_size=None,
            )
        )
    return tuple(candidates)


def build_candidate_embedding_configuration(
    candidate: EmbeddingArenaCandidate,
) -> VpiEmbeddingConfiguration:
    profile = EmbeddingProfile(provider=candidate.provider, model=candidate.model)
    return VpiEmbeddingConfiguration(
        profile=profile,
        expected_dimension=candidate.expected_dimension,
    )
