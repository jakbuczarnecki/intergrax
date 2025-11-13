# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import hashlib
from typing import Optional


def normalize_for_dedupe(text: Optional[str]) -> str:
    """
    Normalizes text before deduplication.

    Steps:
      - Treats None as empty string.
      - Strips leading and trailing whitespace.
      - Converts to lower case.
      - Collapses internal whitespace sequences to a single space.

    This is intentionally simple and fast; heavy normalization
    (e.g., stemming, punctuation removal) should be done elsewhere
    if needed.
    """
    if not text:
        return ""
    stripped = text.strip().lower()
    # Collapse whitespace: split/join to avoid regex dependency.
    return " ".join(stripped.split())


def simple_dedupe_key(text: Optional[str]) -> str:
    """
    Produces a stable SHA-256 based deduplication key for the given text.

    This is used to detect near-identical documents in the web search pipeline.
    For more advanced scenarios, this can later be replaced or extended with
    simhash/minhash, but the interface (text -> key) should remain stable.

    Returns:
      Hex-encoded SHA-256 digest of the normalized text.
    """
    normalized = normalize_for_dedupe(text)
    digest = hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()
    return digest
