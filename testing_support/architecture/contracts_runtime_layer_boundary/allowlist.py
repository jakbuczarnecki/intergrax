# © Artur Czarnecki. All rights reserved.

"""HARDENING-3 — documented contract→runtime import deferrals (SSOT)."""

from __future__ import annotations

# Documented deferred migrations — see docs/project/maintainers/architecture/LAYER_BOUNDARY_HARDENING.md
# Empty @ current HEAD: prior deferred contract→runtime couplings were migrated out.
CONTRACTS_RUNTIME_IMPORT_ALLOWLIST: frozenset[str] = frozenset()
