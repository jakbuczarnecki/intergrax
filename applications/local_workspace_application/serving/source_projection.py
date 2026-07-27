# © Artur Czarnecki. All rights reserved.

"""Safe serving-layer projection helpers for managed workspace sources."""

from __future__ import annotations

import re

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_MAX_LABEL_CHARS = 80

_LOCAL_FOLDER_FALLBACK = "Local folder"
_GENERIC_SOURCE_FALLBACK = "Source"

_TYPE_DISPLAY_FALLBACKS: dict[str, str] = {
    "local_folder": _LOCAL_FOLDER_FALLBACK,
    "object_storage": "Object storage",
    "sharepoint": "SharePoint",
    "google_drive": "Google Drive",
    "remote_repository": "Remote source",
    "remote_drive": "Remote source",
    "uploaded_file": "Uploaded file",
}


def sanitize_display_label(value: str, *, max_chars: int = _MAX_LABEL_CHARS) -> str:
    """Collapse whitespace, neutralize control characters, and bound display length."""
    cleaned = (value or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = _CONTROL_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return ""
    if len(cleaned) > max_chars:
        return cleaned[: max_chars - 1] + "…"
    return cleaned


def local_folder_safe_label(path: str | None) -> str:
    """
    Project a local-folder path to the final folder name only.

    Never returns parent directories, drive letters, or UNC prefixes.
    """
    raw = (path or "").replace("\\", "/")
    # Strip trailing separators so basename is meaningful.
    trimmed = raw.rstrip("/")
    if not trimmed or trimmed in {".", ".."}:
        return _LOCAL_FOLDER_FALLBACK

    # UNC: //host/share[/rest...] — drop host/share; use last segment of rest if any.
    if trimmed.startswith("//"):
        parts = [p for p in trimmed[2:].split("/") if p]
        if len(parts) <= 2:
            return _LOCAL_FOLDER_FALLBACK
        candidate = parts[-1]
    else:
        parts = [p for p in trimmed.split("/") if p]
        if not parts:
            return _LOCAL_FOLDER_FALLBACK
        # Drive-only (e.g. "C:") is not a usable folder name.
        if len(parts) == 1 and len(parts[0]) == 2 and parts[0][1] == ":":
            return _LOCAL_FOLDER_FALLBACK
        candidate = parts[-1]
        if len(candidate) == 2 and candidate[1] == ":":
            return _LOCAL_FOLDER_FALLBACK

    label = sanitize_display_label(candidate)
    if not label or label in {".", ".."}:
        return _LOCAL_FOLDER_FALLBACK
    return label


def safe_source_label(*, source_type: str, path: str | None = None) -> str:
    """Generate a provider-neutral safe label on the API projection side."""
    normalized_type = (source_type or "").strip().casefold()
    if normalized_type == "local_folder":
        return local_folder_safe_label(path)

    typed = _TYPE_DISPLAY_FALLBACKS.get(normalized_type)
    if typed:
        return typed

    if normalized_type:
        humanized = sanitize_display_label(normalized_type.replace("_", " "))
        if humanized:
            return humanized[:1].upper() + humanized[1:]
    return _GENERIC_SOURCE_FALLBACK
