# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Installed-tree digest of a plugin distribution (EX-D6-03, T-IGX-051-B Part D).

A content digest only, not a signature or trust authority: the publisher signature over the attested
``package_digest`` is verified by ``PluginProvenanceAdmission`` with the agent package trust mechanism.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from importlib.metadata import Distribution
from pathlib import Path


class InstalledDistributionDigestError(ValueError):
    """An installed file is missing, unreadable or differs from its ``RECORD`` hash (EX-D6-03)."""


# ``.dist-info`` files rewritten at install time; compiled caches are never part of the tree digest.
_INSTALL_TIME_METADATA_FILES = frozenset({"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"})


def _excluded_from_tree_digest(relative: str) -> bool:
    parts = relative.split("/")
    if "__pycache__" in parts:
        return True
    return len(parts) >= 2 and parts[-2].endswith(".dist-info") and parts[-1] in _INSTALL_TIME_METADATA_FILES


def _is_metadata_dir(name: str) -> bool:
    return name.endswith((".dist-info", ".data", ".egg-info"))


def _unlisted_importable_files(root: Path, listed: frozenset[str]) -> list[str]:
    """Files under the distribution's package directories, or importable siblings of its top-level
    modules, that RECORD does not list (``__pycache__`` excluded: see the module limit)."""
    from importlib.machinery import EXTENSION_SUFFIXES

    unlisted: list[str] = []
    package_dirs = sorted(
        {
            r.split("/")[0]
            for r in listed
            if "/" in r and not r.startswith("..") and not _is_metadata_dir(r.split("/")[0])
        }
    )
    for name in package_dirs:
        base = root / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if "__pycache__" in relative.split("/") or relative in listed:
                continue
            unlisted.append(relative)
    for relative in sorted(r for r in listed if "/" not in r and r.endswith(".py")):
        stem = relative[: -len(".py")]
        for suffix in (*EXTENSION_SUFFIXES, ".pyc"):
            sibling = f"{stem}{suffix}"
            if sibling not in listed and (root / sibling).is_file():
                unlisted.append(sibling)
    return unlisted


def installed_tree_digest(distribution: str, *, search_path: Sequence[str] | None = None) -> str | None:
    """Deterministic digest of an installed distribution's files, re-hashed from disk (EX-D6-03).

    Every file listed in the distribution's ``RECORD`` (except the install-time metadata files and
    compiled caches) is read from disk, its sha256 is compared with the ``RECORD`` hash when one is
    recorded, and the sorted ``relative path`` / ``sha256`` pairs are hashed into one
    ``sha256:<hex>`` value. This is the packaging convention for the attested ``package_digest`` of a
    plugin distribution (Q-SUP-04). Returns ``None`` when the distribution is not installed; raises
    :class:`InstalledDistributionDigestError` for a missing, unreadable or altered file, and for a file
    that RECORD does not list inside the distribution's package directories or an unlisted importable
    sibling (extension module, sourceless ``.pyc``) of one of its top-level modules.

    Security limit (open, EX-D6-03): ``__pycache__`` bytecode is an install-time artefact and is not
    covered; a planted cache file whose header matches the source is loaded instead of the verified
    source. Closing it needs checked-hash ``.pyc`` or no bytecode cache for plugin roots.
    """
    if search_path is None:
        found = list(Distribution.discover(name=distribution))
    else:
        found = list(Distribution.discover(name=distribution, path=list(search_path)))
    if not found:
        return None
    dist = found[0]
    files = dist.files
    if files is None:
        raise InstalledDistributionDigestError(f"distribution {distribution!r} has no RECORD")
    lines: list[str] = []
    for item in files:
        relative = item.as_posix()
        if _excluded_from_tree_digest(relative):
            continue
        try:
            content = Path(str(item.locate())).read_bytes()
        except OSError as exc:
            raise InstalledDistributionDigestError(
                f"installed file of {distribution!r} is missing or unreadable: {relative}"
            ) from exc
        recorded = item.hash
        if recorded is not None:
            if recorded.mode != "sha256":
                raise InstalledDistributionDigestError(
                    f"installed file of {distribution!r} has an unsupported RECORD hash: {relative}"
                )
            actual = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
            if actual != recorded.value:
                raise InstalledDistributionDigestError(
                    f"installed file of {distribution!r} differs from its RECORD hash: {relative}"
                )
        lines.append(f"{relative}\0{hashlib.sha256(content).hexdigest()}\n")
    listed = frozenset(item.as_posix() for item in files)
    unlisted = _unlisted_importable_files(Path(str(dist.locate_file(""))), listed)
    if unlisted:
        raise InstalledDistributionDigestError(
            f"distribution {distribution!r} has files that RECORD does not list: {', '.join(unlisted[:5])}"
        )
    lines.sort()
    return "sha256:" + hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
