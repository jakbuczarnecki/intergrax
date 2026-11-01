# © Artur Czarnecki. All rights reserved.

"""W6-01: real-population measurement of EVOL-M-01..11 (no stubs; every reading derives from data)."""

from __future__ import annotations

import ast
import subprocess
import time
from pathlib import Path

from testing_support.change_safety.assess.consumers import critical_consumer_closure
from testing_support.change_safety.assess.contract_diff import assess_persisted_schema_change, diff_schema_record
from testing_support.change_safety.assess.triggers import (
    build_baseline_manifest,
    compute_actual_digests,
    evaluate_freshness,
    trigger_table,
)
from testing_support.change_safety.contracts.canonical_json import stable_digest_hex
from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult as C
from testing_support.change_safety.contracts.enums import ImpactStatus, SelectionStatus
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.diagnostics.dossier import build_change_dossier
from testing_support.change_safety.discovery.closed_world import (
    ClosedWorldInventory,
    accepted_exclusion_patterns,
    owners_for_path,
    path_matches_patterns,
)
from testing_support.change_safety.discovery.extractors import module_id_for_path, suite_id_for_path
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.intake.git_change_set import GitChangeSetPort
from testing_support.change_safety.metrics.runner import MetricReading, make_reading
from testing_support.change_safety.selection.oracle import (
    FAULT_INJECTION_MATRIX,
    SelectionFaultClass,
    inject_selection_fault,
)
from testing_support.change_safety.selection.selector import select_tests
from tests.qualification.evol_x.adapters.selection_inputs import build_selection_inputs
from tests.qualification.evol_x.experiments._harness import load_oracle
from tests.qualification.evol_x.experiments.arms import s1, s2, s5, s6, s7, s8

_REPO = Path(__file__).resolve().parents[4]
_CODE_ROOTS = ("intergrax/", "agents/", "applications/", "platform_proofs/", "testing_support/change_safety/")
_SCOPE_ROOTS = (*_CODE_ROOTS, "tests/")
DRILL_SLA_SECONDS = 120.0
DRILL_SEEDS = 3


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=_REPO, check=True, capture_output=True, text=True).stdout


def _head_files() -> frozenset[str]:
    return frozenset(_git("ls-files").splitlines())


def _in_scope(path: str) -> bool:
    return path.endswith(".py") and path.startswith(_SCOPE_ROOTS)


# ----------------------------------------------------------------------------------------------- M-01 (P-H)
def p_h_commits(head: str) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    cfg = load_evol_x_data_document("metric_populations.v1.json")["p_h_window"]
    shas = _git("rev-list", "--no-merges", f"--max-count={cfg['max_scan_commits']}", cfg["anchor_sha"]).split()
    out: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for sha in shas:
        change = GitChangeSetPort().load(base_sha=f"{sha}^", head_sha=sha)
        scoped = tuple((p, s) for p, s in change.statuses if _in_scope(p))
        if scoped:
            out.append((sha, scoped))
        if len(out) >= cfg["in_scope_commits"]:
            break
    return tuple(out)


def _path_classified(
    path: str, status: str, inventory: ClosedWorldInventory, owners: tuple[dict, ...], exclusions: tuple[str, ...]
) -> bool:
    if status == "D" or path not in _head_files():
        # removed (then or later): no node at HEAD by design -> classified by an owner rule / test-tree membership
        return bool(owners_for_path(path, owners)) or path.startswith("tests/")
    if module_id_for_path(path) in inventory.module_ids or suite_id_for_path(path) in {
        n.node_id for n in inventory.nodes if n.category == "suite"
    }:
        return True
    return path_matches_patterns(path, exclusions)


def measure_m01(inventory: ClosedWorldInventory, head: str) -> MetricReading:
    owners = tuple(load_evol_x_data_document("owner_rules.v1.json")["owners"])
    exclusions = accepted_exclusion_patterns(tuple(load_evol_x_data_document("exclusion_register.v1.json")["entries"]))
    commits = p_h_commits(head)
    complete = 0
    for _sha, scoped in commits:
        impact_ok = all(_path_classified(p, s, inventory, owners, exclusions) for p, s in scoped)
        live = frozenset(p for p, s in scoped if s != "D" and p in _head_files())
        impact = assess_path_change_impact(inventory, changed_paths=live) if live else None
        if impact_ok and (impact is None or impact.status in {ImpactStatus.COMPLETE, ImpactStatus.PARTIAL}):
            complete += 1
    digest = stable_digest_hex([sha for sha, _ in commits])
    return make_reading(
        "EVOL-M-01", complete, len(commits), population_digest=digest, note=f"P-H last {len(commits)} in-scope commits"
    )


# ----------------------------------------------------------------------------------------------- M-02 / M-02C
def measure_m02(inventory: ClosedWorldInventory) -> tuple[MetricReading, MetricReading]:
    closure = critical_consumer_closure(inventory)
    covered = closure.text_consumer_pairs - closure.orphan_count
    digest = inventory.inventory_digest
    return (
        make_reading("EVOL-M-02", covered, closure.text_consumer_pairs, population_digest=digest, note="CW-06 pairs"),
        make_reading("EVOL-M-02C", closure.orphan_count, closure.text_consumer_pairs, population_digest=digest),
    )


# ----------------------------------------------------------------------------------------------- M-03 family
_IMPORT_INDEX: dict[str, frozenset[str]] | None = None
_SELF_SURFACE = ("testing_support/change_safety/", "tests/qualification/evol_x/")


def _dotted(path: str) -> str:
    stem = path[:-3] if path.endswith(".py") else path
    parts = stem.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _file_imports(path: str, tree: ast.AST) -> frozenset[str]:
    own = _dotted(path).split(".")
    pkg = own if path.endswith("__init__.py") else own[:-1]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = ".".join(pkg[: len(pkg) - node.level + 1]) if node.level else ""
            module = ".".join(x for x in (base, node.module or "") if x)
            if module:
                found.add(module)
                found.update(f"{module}.{alias.name}" for alias in node.names)
    return frozenset(found)


def _import_index() -> dict[str, frozenset[str]]:
    """Independent (ast, not the closed-world extractor) module-import index over git-tracked .py files."""
    global _IMPORT_INDEX
    if _IMPORT_INDEX is None:
        index: dict[str, frozenset[str]] = {}
        for path in _git("ls-files", "*.py").splitlines():
            if not path.startswith(_SCOPE_ROOTS):
                continue
            try:
                tree = ast.parse((_REPO / path).read_bytes())
            except (OSError, SyntaxError, ValueError):
                continue
            index[path] = _file_imports(path, tree)
        _IMPORT_INDEX = index
    return _IMPORT_INDEX


def _reverse_closure(changed: frozenset[str]) -> frozenset[str]:
    index = _import_index()
    by_dotted = {_dotted(p): p for p in index}
    importers: dict[str, set[str]] = {}
    for path, imports in index.items():
        for name in imports:
            if name in by_dotted:
                importers.setdefault(name, set()).add(path)
    seen = set(changed)
    frontier = [_dotted(p) for p in changed]
    while frontier:
        for importer in importers.get(frontier.pop(), ()):
            if importer not in seen:
                seen.add(importer)
                frontier.append(_dotted(importer))
    return frozenset(seen)


def measure_m03(inventory: ClosedWorldInventory) -> tuple[MetricReading, ...]:
    """O-X oracle R*: mandatory + self-surface policy + files in the ast reverse-import closure of the change."""
    arcs = load_evol_x_data_document("metric_populations.v1.json")["p_s_arcs"]
    sel_total = hit = oracle_total = mand_total = mand_hit = 0
    for arc in arcs:
        changed = frozenset(arc["changed_paths"])
        impact = assess_path_change_impact(inventory, changed_paths=changed)
        base = build_selection_inputs(impact, expected_head_sha=impact.source_commit_sha)
        decision = select_tests(base)
        if decision.status is not SelectionStatus.SELECTED:
            sel_total += 1  # a non-SELECTED decision on a clean arc counts as an irrelevant selection (fail)
            continue
        selected = {n.node_id for n in decision.selected}
        closure = _reverse_closure(changed)
        touches_self = any(p.startswith(_SELF_SURFACE) for p in closure)
        universe = base.mandatory_node_ids | base.pop_a_node_ids | base.evol_manifest_node_ids | base.adjacent_node_ids
        relevant_star = {
            n
            for n in universe
            if n in base.mandatory_node_ids
            or n.split("::", 1)[0] in closure
            or (touches_self and n in base.evol_manifest_node_ids)
        }
        sel_total += len(selected)
        hit += len(selected & relevant_star)
        oracle_total += len(relevant_star)
        mand_total += len(base.mandatory_node_ids)
        mand_hit += len(base.mandatory_node_ids & selected)
    digest = stable_digest_hex(arcs)
    return (
        make_reading("EVOL-M-03", hit, sel_total, population_digest=digest, note="P-S arcs; O-X ast-closure oracle R*"),
        make_reading("EVOL-M-03R", hit, oracle_total, population_digest=digest),
        make_reading("EVOL-M-03C", mand_hit, mand_total, population_digest=digest),
        make_reading("EVOL-M-03E", oracle_total - hit, oracle_total, population_digest=digest),
    )


# ----------------------------------------------------------------------------------------------- M-04 / M-05
def measure_m04_m05(inventory: ClosedWorldInventory) -> tuple[MetricReading, MetricReading]:
    changed = frozenset(load_oracle("s6_negative_regression")["changed_paths"])
    impact = assess_path_change_impact(inventory, changed_paths=changed)
    base = build_selection_inputs(impact, expected_head_sha=impact.source_commit_sha)
    safe = 0
    for row in FAULT_INJECTION_MATRIX:
        decision = select_tests(inject_selection_fault(row.fault, base))
        selected = {n.node_id for n in decision.selected}
        if decision.status is SelectionStatus.REFUSED_INTEGRITY or (
            decision.status is SelectionStatus.CONSERVATIVE_FULL and base.mandatory_node_ids <= selected
        ):
            safe += 1
    positive, negative = s6.run(inventory)
    detected = int(positive.passed) + int(negative.passed)
    digest = stable_digest_hex([r.fault.value for r in FAULT_INJECTION_MATRIX])
    return (
        make_reading("EVOL-M-04", safe, len(SelectionFaultClass), population_digest=digest, note="8 fault classes"),
        make_reading("EVOL-M-05", detected, 2, population_digest=digest, note="S6 positive+negative arms"),
    )


# ----------------------------------------------------------------------------------------------- M-06..M-08, M-10
def measure_m06() -> MetricReading:
    oracle = load_oracle("s1_local_agent")["scenarios"]
    expected = sum(len(s["expected_escapes"]) for s in oracle.values())
    positive, negative = s1.run()
    missed = 0 if (positive.passed and negative.passed) else expected
    return make_reading("EVOL-M-06", missed, expected, population_digest=stable_digest_hex(oracle), note="S1 O-F")


def measure_m07() -> MetricReading:
    s2_oracle, s5_oracle = load_oracle("s2_shared_contract"), load_oracle("s5_state_migration")
    cases = (
        diff_schema_record(s2_oracle["schema_before"], s2_oracle["schema_after_breaking"]).result is C.INCOMPATIBLE,
        assess_persisted_schema_change(
            s5_oracle["schema_before"], s5_oracle["schema_after_breaking"], migration_registered=False
        ).result
        is C.INCOMPATIBLE,
        assess_persisted_schema_change(
            s5_oracle["schema_before"], s5_oracle["schema_after_breaking"], migration_registered=True
        ).result
        is C.MIGRATION_REQUIRED,
        all(a.passed for a in (*s2.run(), *s5.run())),
    )
    return make_reading(
        "EVOL-M-07", sum(cases), len(cases), population_digest=stable_digest_hex([s2_oracle, s5_oracle]), note="S2+S5"
    )


def measure_m08() -> MetricReading:
    rows = load_oracle("s7_rollback")["rows"]
    positive, negative = s7.run()
    hits = int(positive.passed) + int(negative.passed)
    return make_reading(
        "EVOL-M-08",
        hits,
        2,
        population_digest=stable_digest_hex(rows),
        note=f"S7 DM matrix {len(rows)} rows + DM-04 snapshots",
    )


def measure_m10() -> MetricReading:
    positive, negative = s8.run()
    undetected = int(not negative.passed) + int(not positive.passed)
    return make_reading(
        "EVOL-M-10", undetected, 2, population_digest=stable_digest_hex(load_oracle("s8_new_application"))
    )


# ----------------------------------------------------------------------------------------------- M-09, M-11
def measure_m09(inventory: ClosedWorldInventory, head: str) -> MetricReading:
    changed = frozenset({"intergrax/contracts/execution_identity.py"})
    started = time.monotonic()
    digests = []
    for _ in range(DRILL_SEEDS):
        impact = assess_path_change_impact(inventory, changed_paths=changed)
        dossier = build_change_dossier(
            graph_digest=impact.graph_digest,
            inventory_digest=inventory.inventory_digest,
            impact=impact,
            created_at_utc="1970-01-01T00:00:00+00:00",
        )
        digests.append(dossier.content_digest)
    elapsed = time.monotonic() - started
    matches = sum(1 for d in digests if d == digests[0])
    return make_reading(
        "EVOL-M-09",
        matches,
        DRILL_SEEDS,
        population_digest=digests[0],
        note=(
            f"tool_seconds={elapsed:.2f} (SLA<={DRILL_SLA_SECONDS:.0f}s); operator attribution median<=15min "
            "is recorded by the independent B6 drill (W7), not by this tool reading"
        ),
        extra_condition=elapsed <= DRILL_SLA_SECONDS,
    )


def measure_m11(inventory: ClosedWorldInventory, head: str) -> MetricReading:
    actual = compute_actual_digests(inventory)
    baseline = build_baseline_manifest(accepted_at_sha=head, digests=actual)
    fired = 0
    rows = trigger_table()
    for row in rows:
        verdict = evaluate_freshness(baseline, dict(actual, **{row["digest_key"]: "tampered"}))
        fired += int(verdict.stale_triggers == (row["trigger_id"],))
    fresh_ok = evaluate_freshness(baseline, actual).allowed
    return make_reading(
        "EVOL-M-11",
        fired,
        len(rows),
        population_digest=stable_digest_hex(list(rows)),
        note="each trigger fires on its own digest flip; baseline fresh",
        extra_condition=fresh_ok,
    )


def collect_all(inventory: ClosedWorldInventory, head: str) -> tuple[MetricReading, ...]:
    m02, m02c = measure_m02(inventory)
    m04, m05 = measure_m04_m05(inventory)
    readings = (
        measure_m01(inventory, head),
        m02,
        m02c,
        *measure_m03(inventory),
        m04,
        m05,
        measure_m06(),
        measure_m07(),
        measure_m08(),
        measure_m09(inventory, head),
        measure_m10(),
        measure_m11(inventory, head),
    )
    return tuple(sorted(readings, key=lambda r: r.metric_id))
