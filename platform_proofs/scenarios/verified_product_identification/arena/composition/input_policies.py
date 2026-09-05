"""Arena candidate input policies — composition boundary only."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.arena.contracts.input_policy import (
    EmbeddingInputPolicyRef,
    EmbeddingInputRole,
    EmbeddingInputTransformation,
)

BGE_M3_INPUT_POLICY_VERSION = "bge-m3-v1"
QWEN3_EMBEDDING_INPUT_POLICY_VERSION = "qwen3-embedding-0.6b-v1"
NOMIC_V2_INPUT_POLICY_VERSION = "nomic-embed-text-v2-moe-v1"
E5_INSTRUCT_INPUT_POLICY_VERSION = "e5-large-instruct-v1"
IDENTITY_INPUT_POLICY_VERSION = "identity-v1"


@dataclass(frozen=True, slots=True)
class StaticEmbeddingInputTransformation:
    policy_ref: EmbeddingInputPolicyRef
    query_prefix: str
    document_prefix: str

    def transform(self, role: EmbeddingInputRole, canonical_text: str) -> str:
        if role is EmbeddingInputRole.QUERY:
            return f"{self.query_prefix}{canonical_text}"
        return f"{self.document_prefix}{canonical_text}"


def identity_input_policy(*, policy_id: str, policy_version: str) -> StaticEmbeddingInputTransformation:
    ref = EmbeddingInputPolicyRef(
        policy_id=policy_id,
        policy_version=policy_version,
        query_instruction_summary="no query instruction; canonical semantic_text only",
        document_instruction_summary="no document prefix; canonical semantic_text only",
    )
    return StaticEmbeddingInputTransformation(
        policy_ref=ref,
        query_prefix="",
        document_prefix="",
    )


def bge_m3_input_policy() -> StaticEmbeddingInputTransformation:
    return identity_input_policy(
        policy_id="bge-m3",
        policy_version=BGE_M3_INPUT_POLICY_VERSION,
    )


def qwen3_embedding_input_policy() -> StaticEmbeddingInputTransformation:
    ref = EmbeddingInputPolicyRef(
        policy_id="qwen3-embedding-0.6b",
        policy_version=QWEN3_EMBEDDING_INPUT_POLICY_VERSION,
        query_instruction_summary=(
            "Official Qwen3 retrieval query instruction: "
            "Instruct: Given a web search query, retrieve relevant passages\\nQuery:"
        ),
        document_instruction_summary="Official Qwen3 document encoding: canonical text without query instruction",
    )
    return StaticEmbeddingInputTransformation(
        policy_ref=ref,
        query_prefix="Instruct: Given a web search query, retrieve relevant passages\nQuery: ",
        document_prefix="",
    )


def nomic_v2_input_policy() -> StaticEmbeddingInputTransformation:
    ref = EmbeddingInputPolicyRef(
        policy_id="nomic-embed-text-v2-moe",
        policy_version=NOMIC_V2_INPUT_POLICY_VERSION,
        query_instruction_summary="Official Nomic retrieval prefix: search_query:",
        document_instruction_summary="Official Nomic retrieval prefix: search_document:",
    )
    return StaticEmbeddingInputTransformation(
        policy_ref=ref,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )


def e5_instruct_input_policy() -> StaticEmbeddingInputTransformation:
    ref = EmbeddingInputPolicyRef(
        policy_id="e5-large-instruct",
        policy_version=E5_INSTRUCT_INPUT_POLICY_VERSION,
        query_instruction_summary="Official E5 instruct query prefix: query:",
        document_instruction_summary="Official E5 instruct passage prefix: passage:",
    )
    return StaticEmbeddingInputTransformation(
        policy_ref=ref,
        query_prefix="query: ",
        document_prefix="passage: ",
    )


def resolve_input_transformation(policy_id: str) -> EmbeddingInputTransformation:
    if policy_id == "bge-m3":
        return bge_m3_input_policy()
    if policy_id == "qwen3-embedding-0.6b":
        return qwen3_embedding_input_policy()
    if policy_id == "nomic-embed-text-v2-moe":
        return nomic_v2_input_policy()
    if policy_id == "e5-large-instruct":
        return e5_instruct_input_policy()
    msg = f"unsupported input policy id: {policy_id}"
    raise ValueError(msg)
