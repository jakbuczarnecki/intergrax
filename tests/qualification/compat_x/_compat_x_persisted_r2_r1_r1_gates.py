# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-R1-R1 idempotency ledger universal schema-gate evidence."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from tests.qualification.compat_x._compat_x_persisted_r2_r1_gates import (
    persisted_r2_r1_inventory_gate_passes,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SQLITE_PATH = _REPO_ROOT / "intergrax/runtime/tools/sqlite_idempotency_store.py"
_REDIS_PATH = _REPO_ROOT / "intergrax/distributed/providers/redis_idempotency_store.py"
_CANONICAL_SCHEMA = "tool_idempotency_ledger.v1"

_MUTATION_METHODS = (
    "complete_with_claim",
    "mark_uncertain_with_claim",
    "abandon_pre_effect_with_claim",
    "admit_external_effect_may_have_started_with_claim",
)

_LUA_SCRIPTS_ATTR = (
    "_complete_with_claim_script",
    "_mark_uncertain_with_claim_script",
    "_abandon_pre_effect_with_claim_script",
    "_admit_external_effect_script",
    "_reconcile_abandoned_pre_effect_script",
)


@dataclass(frozen=True, slots=True)
class IdempotencySchemaGateReport:
    sqlite_mutation_methods_gated: bool
    redis_lua_scripts_gated: bool
    redis_external_effect_read_gated: bool
    r2_r1_inventory_gate: bool


def _sqlite_mutation_methods_reference_schema_predicate() -> bool:
    source = _SQLITE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in _MUTATION_METHODS:
            continue
        segment = ast.get_source_segment(source, node) or ""
        if "schema_version" not in segment:
            return False
        if "SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1" not in segment and (
            "_require_ledger_schema" not in segment
        ):
            return False
    return True


def _redis_lua_scripts_gate_schema() -> bool:
    source = _REDIS_PATH.read_text(encoding="utf-8")
    for attr in _LUA_SCRIPTS_ATTR:
        pattern = rf'{attr} = self\._redis\.register_script\(\s*"""(.*?)"""'
        match = re.search(pattern, source, re.DOTALL)
        if not match:
            return False
        body = match.group(1)
        if 'HGET", KEYS[1], "schema_version"' not in body:
            return False
        if f'~= "{_CANONICAL_SCHEMA}"' not in body and (
            f'== "{_CANONICAL_SCHEMA}"' not in body
        ):
            return False
    return True


def _redis_external_effect_read_uses_hgetall_and_schema() -> bool:
    source = _REDIS_PATH.read_text(encoding="utf-8")
    fn_match = re.search(
        r"def external_effect_may_have_started\(.*?\n(?:.*\n)*?        return",
        source,
    )
    if not fn_match:
        return False
    body = fn_match.group(0)
    return "hgetall" in body and "_require_ledger_schema_hash" in body


def build_idempotency_schema_gate_report() -> IdempotencySchemaGateReport:
    return IdempotencySchemaGateReport(
        sqlite_mutation_methods_gated=_sqlite_mutation_methods_reference_schema_predicate(),
        redis_lua_scripts_gated=_redis_lua_scripts_gate_schema(),
        redis_external_effect_read_gated=_redis_external_effect_read_uses_hgetall_and_schema(),
        r2_r1_inventory_gate=persisted_r2_r1_inventory_gate_passes(),
    )


def persisted_r2_r1_r1_schema_gate_passes() -> bool:
    report = build_idempotency_schema_gate_report()
    return (
        report.r2_r1_inventory_gate
        and report.sqlite_mutation_methods_gated
        and report.redis_lua_scripts_gated
        and report.redis_external_effect_read_gated
    )
