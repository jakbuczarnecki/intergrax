# © Artur Czarnecki. All rights reserved.

"""Package-prefix allowlist for loaders that import a module named by serialized or configured data.

Decision DP-21 (follow-on of EQOA-A7-021, A4-007). The loaders (``wiring.load_callable``,
``agent_resolution.resolve_agent_type``, ``migration_wiring._load_manifest_for_package``,
``mcp_import_guard.load_mcp_server_builder``) call :func:`require_import_allowed` before
``importlib.import_module``. This module holds no platform constant: the permitted packages are
supplied by host composition, derived from one registered source (for example the Tier-2 agent
packages or the application packages of the product manifest registry).

* With an allowlist the check is enforced in every posture.
* Without an allowlist the check is a no-op, unless the caller declares ``fail_closed``: then a
  missing allowlist is itself refused.

The two errors are deliberately not ``ImportError`` subclasses, so a loader that swallows
``ImportError`` can never turn a refusal into "module not found".
"""

from __future__ import annotations

from dataclasses import dataclass


class ImportAllowlistError(Exception):
    """Base class of import allowlist refusals (not an ``ImportError``)."""


class ImportPathNotAllowedError(ImportAllowlistError):
    """The module path is outside every permitted package prefix."""


class ImportAllowlistRequiredError(ImportAllowlistError):
    """A fail-closed caller supplied no allowlist."""


def _is_dotted_identifier(path: str) -> bool:
    parts = path.split(".")
    return bool(path) and all(part.isidentifier() for part in parts)


@dataclass(frozen=True)
class ImportPrefixAllowlist:
    """Whole-package prefixes (no trailing dot); ``permits`` matches the package or a submodule."""

    prefixes: tuple[str, ...]

    def __post_init__(self) -> None:
        for prefix in self.prefixes:
            if not _is_dotted_identifier(prefix):
                raise ValueError(f"invalid import prefix {prefix!r}: expected a dotted package name")

    def permits(self, path: str) -> bool:
        """True when ``path`` is a well-formed dotted name equal to or below a permitted prefix."""
        if not _is_dotted_identifier(path):
            return False
        return any(path == prefix or path.startswith(prefix + ".") for prefix in self.prefixes)


def require_import_allowed(
    path: str,
    allowlist: ImportPrefixAllowlist | None,
    *,
    fail_closed: bool,
    purpose: str,
) -> None:
    """Raise a typed error unless importing ``path`` is permitted for ``purpose``."""
    if allowlist is None:
        if fail_closed:
            raise ImportAllowlistRequiredError(
                f"{purpose}: no import allowlist supplied; refusing to import {path!r}",
            )
        return
    if not allowlist.permits(path):
        raise ImportPathNotAllowedError(f"{purpose}: import of {path!r} is outside the allowlist")
