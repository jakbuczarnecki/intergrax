"""Test support for resumable VPI data pack builder."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    DataPackBuildConfig,
    DataPackBuildReport,
    ShardBuildSeams,
    run_resumable_data_pack_build,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.token_budget_truncation import (
    truncate_to_token_limit,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding_input_policy import (
    VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
    VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.ports import (
    DataPackEmbeddingPort,
)


class FakeDataPackEmbeddingPort:
    def __init__(self, *, dimension: int = 1024) -> None:
        self.dimension = dimension
        self.embed_calls = 0
        self.texts_seen: list[str] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        self.texts_seen.extend(texts)
        return [[0.1] * self.dimension for _ in texts]

    def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class FakeTokenizerCodec:
    def encode_text_tokens(self, text: str) -> tuple[int, ...]:
        return tuple(ord(char) for char in text)

    def decode_text_tokens(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


@dataclass(frozen=True, slots=True)
class FakeDocumentEmbeddingInputPolicy:
    policy_version: str
    token_budget: int | None = None
    tokenizer_codec: FakeTokenizerCodec | None = None

    def apply_document(self, text: str) -> str:
        if self.token_budget is None or self.tokenizer_codec is None:
            return text
        return truncate_to_token_limit(
            text,
            token_limit=self.token_budget,
            encode=self.tokenizer_codec.encode_text_tokens,
            decode=self.tokenizer_codec.decode_text_tokens,
        )


def canonical_fake_document_embedding_input_policy(
    *,
    token_budget: int = VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
) -> FakeDocumentEmbeddingInputPolicy:
    return FakeDocumentEmbeddingInputPolicy(
        policy_version=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
        token_budget=token_budget,
        tokenizer_codec=FakeTokenizerCodec(),
    )


def run_resumable_data_pack_build_with_fake_policy(
    config: DataPackBuildConfig,
    *,
    embedding_port: DataPackEmbeddingPort | None = None,
    document_embedding_input_policy: FakeDocumentEmbeddingInputPolicy | None = None,
    build_seams: ShardBuildSeams | None = None,
) -> DataPackBuildReport:
    return run_resumable_data_pack_build(
        config,
        embedding_port=embedding_port or FakeDataPackEmbeddingPort(),
        document_embedding_input_policy=(
            document_embedding_input_policy or canonical_fake_document_embedding_input_policy()
        ),
        build_seams=build_seams,
    )


def write_tiny_selected_dataset(
    directory: Path,
    *,
    row_count: int,
) -> tuple[Path, Path]:
    return write_selected_dataset_with_manifest_count(
        directory,
        parquet_row_count=row_count,
        manifest_record_count=row_count,
    )


def write_selected_dataset_with_manifest_count(
    directory: Path,
    *,
    parquet_row_count: int,
    manifest_record_count: int,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    dataset_path = directory / "selected_offers.parquet"
    manifest_path = directory / "selected_offers_manifest.json"
    records = []
    for index in range(parquet_row_count):
        records.append(
            json.dumps(
                {
                    "id": f"offer-{index}",
                    "title": f"Relay module {index}",
                    "identifiers": [{"gtin": f"{1000000000000 + index}"}],
                    "keyValuePairs": {"voltage": "24V"},
                }
            )
        )
    table = pa.table({"record_json": records})
    pq.write_table(table, dataset_path)
    manifest_path.write_text(
        json.dumps(
            {
                "source_dataset_name": "offers_corpus_all_v2_non_norm",
                "output_path": str(dataset_path),
                "output_sha256": "a" * 64,
                "selected_record_count": manifest_record_count,
            }
        ),
        encoding="utf-8",
    )
    return dataset_path, manifest_path


def patch_canonical_model_identity(monkeypatch) -> None:
    from platform_proofs.scenarios.verified_product_identification.integrations.embedding.bootstrap import (
        ensure_embedding_provider_integrations_registered,
    )
    from platform_proofs.scenarios.verified_product_identification.integrations.embedding.model_identity import (
        EmbeddingModelArtifactIdentity,
    )

    ensure_embedding_provider_integrations_registered()
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder.resolve_embedding_model_identity",
        lambda provider, model: EmbeddingModelArtifactIdentity(
            provider=provider,
            model=model,
            revision="5617a9f61b028005a4858fdac845db406aefb181",
            artifact_fingerprint=None,
        ),
    )
