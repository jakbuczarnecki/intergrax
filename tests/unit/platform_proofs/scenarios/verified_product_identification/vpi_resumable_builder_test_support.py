"""Test support for resumable VPI data pack builder."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
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


def write_tiny_selected_dataset(
    directory: Path,
    *,
    row_count: int,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    dataset_path = directory / "selected_offers.parquet"
    manifest_path = directory / "selected_offers_manifest.json"
    records = []
    for index in range(row_count):
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
                "output_sha256": "testsha256" + "0" * 54,
                "selected_record_count": row_count,
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
