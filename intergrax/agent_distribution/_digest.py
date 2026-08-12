# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic content-addressed digests for Agent Distribution contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from intergrax.runtime.attestation.canonical_json import stable_payload_hash


def content_digest_for_model(model: BaseModel) -> str:
    """Return a stable sha256 digest for a frozen distribution contract model."""
    payload: dict[str, Any] = model.model_dump(mode="json")
    return stable_payload_hash(payload)
