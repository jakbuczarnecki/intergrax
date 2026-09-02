# © Artur Czarnecki. All rights reserved.

"""Generic URL identity normalization for source selection."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def normalize_url_identity(url: str) -> str:
    """Normalize scheme/host case, strip fragments, preserve path/query identity."""
    raw = url.strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    normalized = urlunparse(
        (
            scheme,
            host,
            path,
            "",
            parsed.query,
            "",
        ),
    )
    return normalized


__all__ = ["normalize_url_identity"]
