# © Artur Czarnecki. All rights reserved.

"""COMPAT-X shim discovery / parallel-authority scope reconciliation (P0-R3)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from tests.qualification.compat_x._compat_x_ast_signals import (
    build_compatibility_candidate_context,
    is_compatibility_adapter_candidate,
    module_exhibits_parallel_authority,
    parse_module,
)
from tests.qualification.compat_x._compat_x_closed_world import build_closed_world_report
from tests.qualification.compat_x._compat_x_types import ShimAuthorityScopeReconciliation, ShimClass

_REPO_ROOT = Path(__file__).resolve().parents[3]


def inspect_parallel_authority_for_module(module_path: str, source: str) -> bool | None:
    """Return parallel-authority verdict when candidate; ``None`` when not a compatibility candidate."""
    tree = parse_module(module_path, source)
    context = build_compatibility_candidate_context(module_path, source, tree)
    if not is_compatibility_adapter_candidate(context):
        return None
    return module_exhibits_parallel_authority(tree, context)


@lru_cache(maxsize=1)
def build_shim_authority_scope_reconciliation() -> ShimAuthorityScopeReconciliation:
    report = build_closed_world_report()
    shim_paths = sorted({c.path for c in report.raw_candidates if c.discovery_kind == "compat.shim"})
    inspected = 0
    uninspected = 0
    parallel = 0
    for module_path in shim_paths:
        full = _REPO_ROOT / module_path
        if not full.is_file():
            uninspected += 1
            continue
        source = full.read_text(encoding="utf-8")
        verdict = inspect_parallel_authority_for_module(module_path, source)
        if verdict is None:
            uninspected += 1
            continue
        inspected += 1
        if verdict:
            parallel += 1
    return ShimAuthorityScopeReconciliation(
        total_compatibility_candidates=len(shim_paths),
        authority_inspected_compatibility_candidates=inspected,
        uninspected_compatibility_candidates=uninspected,
        production_parallel_authority_count=parallel,
    )


def production_inventory_parallel_authority_count() -> int:
    from tests.qualification.compat_x._compat_x_inventory import COMPAT_X_INVENTORY

    return sum(1 for row in COMPAT_X_INVENTORY if row.shim_class == ShimClass.PARALLEL_AUTHORITY)
