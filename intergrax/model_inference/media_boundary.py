# © Artur Czarnecki. All rights reserved.

"""Canonical local media resolution and remote egress boundary for Plane C vision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from intergrax.model_inference.contracts import (
    AuthorizedLocalMedia,
    MediaAuthorizationError,
    VisionInferenceRequest,
)


@dataclass(frozen=True)
class RemoteMediaEgressPolicy:
    """Host/runtime authority for sending authorized local media to remote inference."""

    permitted: bool = False

REMOTE_EGRESS_VISION_ADAPTER_SLUGS: frozenset[str] = frozenset(
    {"vision_serving", "huggingface_inference"}
)


def adapter_requires_remote_egress(adapter_slug: str) -> bool:
    """Return whether the effective adapter sends authorized local bytes outside the host."""
    return adapter_slug in REMOTE_EGRESS_VISION_ADAPTER_SLUGS


def host_permits_remote_media_egress(policy: RemoteMediaEgressPolicy | None) -> bool:
    """Return trusted host/runtime permission for remote media egress."""
    return policy is not None and policy.permitted


def parse_local_media_candidate(media_uri: str) -> Path:
    stripped = media_uri.strip()
    if not stripped:
        raise MediaAuthorizationError("media_uri_empty")
    if stripped.startswith("file://"):
        parsed = urlparse(stripped)
        path_str = unquote(parsed.path)
        if path_str.startswith("/") and len(path_str) > 2 and path_str[2] == ":":
            path_str = path_str[1:]
        return Path(path_str)
    if "://" in stripped:
        raise MediaAuthorizationError("unsupported_media_uri_scheme")
    return Path(stripped)


def resolve_path_within_allowlist_roots(candidate: Path, roots: frozenset[str]) -> Path:
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        try:
            if candidate.is_absolute():
                resolved = candidate.expanduser().resolve()
            else:
                resolved = (root_path / candidate).resolve()
            resolved.relative_to(root_path)
            return resolved
        except (ValueError, OSError):
            continue
    raise MediaAuthorizationError("media_not_in_allowlist")


def resolve_authorized_local_media(
    media_uri: str,
    *,
    roots: frozenset[str] | None,
    remote_egress_permitted: bool = False,
) -> AuthorizedLocalMedia:
    """Resolve caller media against read roots; egress must be minted separately by host policy."""
    if not roots:
        raise MediaAuthorizationError("read_allowlist_not_configured")
    candidate = parse_local_media_candidate(media_uri)
    resolved = resolve_path_within_allowlist_roots(candidate, roots)
    return AuthorizedLocalMedia(
        resolved_path=resolved,
        remote_egress_permitted=remote_egress_permitted,
    )


def local_media_path_from_request(request: VisionInferenceRequest) -> Path:
    """Return an authorized path or fall back to lab-only direct adapter resolution."""
    if request.authorized_local_media is not None:
        return request.authorized_local_media.resolved_path
    return parse_local_media_candidate(request.media_uri).expanduser().resolve()


def require_remote_egress_bytes(request: VisionInferenceRequest) -> bytes:
    media = request.authorized_local_media
    if media is None or not media.remote_egress_permitted:
        raise MediaAuthorizationError("remote_media_egress_not_authorized")
    return media.resolved_path.read_bytes()
