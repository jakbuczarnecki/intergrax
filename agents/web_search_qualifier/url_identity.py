# © Artur Czarnecki. All rights reserved.

"""Deterministic URL identity normalization for web-search qualification evidence."""

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


def artifact_ref_for_url(url: str) -> str:
    normalized = normalize_url_identity(url)
    return f"url:{normalized}"


def url_from_artifact_ref(artifact_ref: str) -> str:
    prefix = "url:"
    if artifact_ref.startswith(prefix):
        return artifact_ref[len(prefix) :]
    return artifact_ref


_CANONICAL_PYTHON_3120_RELEASE_PATH = "/downloads/release/python-3120"


def is_official_python_release_source(url: str) -> bool:
    normalized = normalize_url_identity(url)
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()
    if not host.endswith("python.org"):
        return False
    path = parsed.path.lower()
    return "python-312" in path or "python/3.12" in path


def is_expected_python_3120_release_source(url: str) -> bool:
    """True only for the canonical Python 3.12.0 final-release page."""
    normalized = normalize_url_identity(url)
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()
    if not host.endswith("python.org"):
        return False
    path = parsed.path.lower().rstrip("/") or "/"
    return path == _CANONICAL_PYTHON_3120_RELEASE_PATH


__all__ = [
    "artifact_ref_for_url",
    "is_expected_python_3120_release_source",
    "is_official_python_release_source",
    "normalize_url_identity",
    "url_from_artifact_ref",
]
