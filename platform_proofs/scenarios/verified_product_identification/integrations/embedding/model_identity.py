"""HF embedding model identity resolution for VPI data pack builds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    EmbeddingModelIdentityError,
)


@dataclass(frozen=True, slots=True)
class EmbeddingModelArtifactIdentity:
    provider: str
    model: str
    revision: str
    artifact_fingerprint: str | None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.revision.strip():
            raise ValueError("revision must be non-empty")


def _revision_from_hf_cache(model: str) -> str | None:
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id != model:
            continue
        main_revisions = [
            revision.commit_hash
            for revision in repo.revisions
            if "main" in revision.refs
        ]
        if main_revisions:
            return main_revisions[0]
        if repo.revisions:
            return next(iter(repo.revisions)).commit_hash
    return None


def _revision_from_hf_hub(model: str) -> str:
    from huggingface_hub import model_info

    info = model_info(model)
    if info.sha is None or not info.sha.strip():
        raise EmbeddingModelIdentityError(f"HF model_info returned empty sha for {model}")
    return info.sha


def _fingerprint_from_snapshot(snapshot_path: Path) -> str:
    identity_files = (
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "sentence_bert_config.json",
    )
    digest = hashlib.sha256()
    for file_name in identity_files:
        file_path = snapshot_path / file_name
        if file_path.is_file():
            digest.update(file_name.encode("utf-8"))
            digest.update(file_path.read_bytes())
    fingerprint = digest.hexdigest()
    if fingerprint == hashlib.sha256().hexdigest():
        raise EmbeddingModelIdentityError(
            f"unable to derive artifact fingerprint from snapshot: {snapshot_path}"
        )
    return fingerprint


def resolve_hf_embedding_model_identity(model: str) -> EmbeddingModelArtifactIdentity:
    revision = _revision_from_hf_cache(model)
    artifact_fingerprint: str | None = None
    if revision is None:
        revision = _revision_from_hf_hub(model)
    else:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id != model:
                continue
            for cached_revision in repo.revisions:
                if cached_revision.commit_hash == revision:
                    artifact_fingerprint = _fingerprint_from_snapshot(cached_revision.snapshot_path)
                    break
    if revision is None:
        raise EmbeddingModelIdentityError(f"unable to resolve HF revision for model {model}")
    return EmbeddingModelArtifactIdentity(
        provider="hf",
        model=model,
        revision=revision,
        artifact_fingerprint=artifact_fingerprint,
    )


def resolve_embedding_model_identity(provider: str, model: str) -> EmbeddingModelArtifactIdentity:
    if provider == "hf":
        resolved = resolve_hf_embedding_model_identity(model)
        if resolved.model != model:
            raise EmbeddingModelIdentityError("resolved HF model name mismatch")
        return resolved
    raise EmbeddingModelIdentityError(f"unsupported embedding provider for identity resolution: {provider}")
