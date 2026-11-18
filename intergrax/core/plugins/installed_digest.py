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
    """Entries that RECORD does not list and that could run instead of the verified files.

    In a regular package directory of the distribution (one with a listed ``__init__``) every file and
    every non-cache sub-directory must be listed; files written there at runtime are therefore refused
    too. In the site root and in a shared PEP 420 namespace directory only shadowing entries count: an
    unlisted file or directory with the name of a listed module or package (any import suffix, or a
    directory holding files), and a planted ``__init__`` in a namespace directory. Other distributions'
    entries in a shared namespace are not refused. ``__pycache__`` is excluded (see the module limit).
    """
    from importlib.machinery import EXTENSION_SUFFIXES

    suffixes = (*EXTENSION_SUFFIXES, ".py", ".pyc")
    paths = {r for r in listed if not r.startswith("..") and not _is_metadata_dir(r.split("/")[0])}
    levels: set[str] = {""}
    for relative in paths:
        parts = relative.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            levels.add("/".join(parts[:depth]))
    regular = {level for level in levels if level and any(_join(level, f"__init__{s}") in listed for s in suffixes)}

    unlisted: list[str] = []
    for level in sorted(levels):
        base = root / level if level else root
        if not base.is_dir():
            continue
        if level in regular:
            # A regular package directory belongs to this distribution: every entry must be listed.
            for child in sorted(base.iterdir()):
                relative = _join(level, child.name)
                if child.is_file():
                    if relative not in listed:
                        unlisted.append(relative)
                elif child.name != "__pycache__" and relative not in levels and _holds_files(child):
                    unlisted.append(relative + "/")
            continue
        # The site root or a namespace directory is shared: check only entries that would shadow a
        # listed module or package name, and (for a namespace directory) a planted ``__init__``.
        if level:
            unlisted.extend(
                _join(level, f"__init__{s}") for s in suffixes if _unlisted_file(root, level, f"__init__{s}", listed)
            )
        names = {_module_stem(name, suffixes) for name in _listed_children(level, paths)} - {None}
        names |= {child.rsplit("/", 1)[-1] for child in levels if child and _parent(child) == level}
        for name in sorted(n for n in names if n is not None):
            unlisted.extend(
                _join(level, f"{name}{s}") for s in suffixes if _unlisted_file(root, level, f"{name}{s}", listed)
            )
            directory = _join(level, name)
            if directory not in levels and (root / directory).is_dir() and _holds_files(root / directory):
                unlisted.append(directory + "/")
    return unlisted


def _join(level: str, name: str) -> str:
    return f"{level}/{name}" if level else name


def _parent(relative: str) -> str:
    return relative.rsplit("/", 1)[0] if "/" in relative else ""


def _listed_children(level: str, paths: set[str]) -> list[str]:
    return [p.rsplit("/", 1)[-1] for p in paths if _parent(p) == level]


def _module_stem(name: str, suffixes: tuple[str, ...]) -> str | None:
    for suffix in sorted(suffixes, key=len, reverse=True):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return None


def _unlisted_file(root: Path, level: str, name: str, listed: frozenset[str]) -> bool:
    relative = _join(level, name)
    return relative not in listed and (root / relative).is_file()


def _holds_files(directory: Path) -> bool:
    return any(p.is_file() and "__pycache__" not in p.parts for p in directory.rglob("*"))


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
