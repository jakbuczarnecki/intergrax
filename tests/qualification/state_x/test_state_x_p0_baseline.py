# © Artur Czarnecki. All rights reserved.

"""STATE-X-P0 mechanical baseline gates (SX-P0-Q01..Q24)."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from tests.qualification.state_x.inventory import (
    AuthorityRole,
    ContractReference,
    IdentityRole,
    MANDATORY_FAMILY_IDS,
    ProjectionOrTruth,
    SemanticOwnershipRole,
    STATE_X_FAMILY_INVENTORY,
    STATE_X_KNOWN_BLOCKERS,
    STATE_X_P0_ALLOWLIST_PATHS,
    STATE_X_P0_AUDITED_HEAD,
    STATE_X_R1_ALLOWLIST_PATHS,
    STATE_X_R2_ALLOWLIST_PATHS,
    STATE_X_R2_PRE_AUDIT_HEAD,
    STATE_X_R2_R1_ALLOWLIST_PATHS,
    STATE_X_R3_R1_ALLOWLIST_PATHS,
    STATE_X_R3_R1_PRE_AUDIT_HEAD,
    STATE_X_R3_R2_ALLOWLIST_PATHS,
    STATE_X_R3_R2_PRE_AUDIT_HEAD,
    STATE_X_R3_R3_ALLOWLIST_PATHS,
    STATE_X_R3_R3_PRE_AUDIT_HEAD,
    StateFamilyInventoryEntry,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_FORBIDDEN_CHANGE_PREFIXES = (
    "intergrax/",
    "applications/",
    "scripts/",
)

_NON_CANONICAL_TRUTH_CLASSES = frozenset(
    {
        ProjectionOrTruth.DERIVED_PROJECTION,
        ProjectionOrTruth.READ_MODEL,
        ProjectionOrTruth.COORDINATION_ONLY,
    },
)

_REQUIRES_CONTRACT_REFERENCES = frozenset(
    {
        ProjectionOrTruth.CANONICAL_TRUTH,
        ProjectionOrTruth.DURABLE_COMPONENT_OF_CANONICAL_TRUTH,
        ProjectionOrTruth.READ_MODEL,
    },
)

_KNOWN_INVALID_PATH_FRAGMENTS = (
    "intergrax/runtime/long_running/in_memory_checkpoint_store.py",
    "intergrax/runtime/events/sqlite_event_store.py",
    "intergrax/runtime/human/sqlite_human_decision_store.py",
)

_COMPOUND_PSEUDO_CONTRACT_FRAGMENTS = (
    "DecisionCheckpointPersistence / DecisionCheckpointState",
    "RunBudgetPersistence / ExecutionBudgetLedger",
    "ScheduledResumePersistence / SchedulerLedger",
    "RunTraceStore / RunTraceReader",
    "ExecutionTerminalStore / ExecutionTerminalRecord",
    "RuntimeCheckpoint.execution_tree / ExecutionTreeSnapshot",
)

_SYMBOL_CACHE: dict[str, frozenset[str]] = {}


def _inventory_by_id() -> dict[str, StateFamilyInventoryEntry]:
    return {entry.family_id: entry for entry in STATE_X_FAMILY_INVENTORY}


def _defined_symbols_in_file(rel_path: str) -> frozenset[str]:
    if rel_path in _SYMBOL_CACHE:
        return _SYMBOL_CACHE[rel_path]
    full = _REPO_ROOT / rel_path
    text = full.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=rel_path)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    _SYMBOL_CACHE[rel_path] = frozenset(names)
    return _SYMBOL_CACHE[rel_path]


def _implementation_resolution_paths(entry: StateFamilyInventoryEntry) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for rel in entry.production_paths:
        if rel not in seen:
            seen.add(rel)
            ordered.append(rel)
    for ref in entry.contract_references:
        if ref.path not in seen:
            seen.add(ref.path)
            ordered.append(ref.path)
    return tuple(ordered)


def _symbol_resolves(symbol: str, rel_paths: tuple[str, ...]) -> bool:
    for rel in rel_paths:
        if symbol in _defined_symbols_in_file(rel):
            return True
    return False


def _symbol_resolves_in_exact_path(symbol: str, rel_path: str) -> bool:
    return symbol in _defined_symbols_in_file(rel_path)


def _all_contract_references() -> tuple[ContractReference, ...]:
    refs: list[ContractReference] = []
    for entry in STATE_X_FAMILY_INVENTORY:
        refs.extend(entry.contract_references)
    return tuple(refs)


def test_sx_p0_q01_mandatory_family_ids_exactly_once() -> None:
    ids = [entry.family_id for entry in STATE_X_FAMILY_INVENTORY]
    assert ids == list(MANDATORY_FAMILY_IDS)
    assert len(ids) == len(set(ids))


def test_sx_p0_q02_ownership_role_and_contract_requirements() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        assert entry.composition_owner.strip()
        if entry.projection_or_truth in _REQUIRES_CONTRACT_REFERENCES:
            assert entry.contract_references
        elif entry.projection_or_truth == ProjectionOrTruth.COORDINATION_ONLY:
            assert entry.contract_references == ()
        if entry.projection_or_truth == ProjectionOrTruth.CANONICAL_TRUTH:
            assert entry.semantic_ownership_role == SemanticOwnershipRole.CANONICAL_OWNER
            assert entry.semantic_owner.strip()
        elif entry.projection_or_truth == ProjectionOrTruth.DURABLE_COMPONENT_OF_CANONICAL_TRUTH:
            assert entry.semantic_ownership_role == SemanticOwnershipRole.COMPONENT_OWNER
            assert entry.semantic_owner.strip()
        elif entry.projection_or_truth == ProjectionOrTruth.COORDINATION_ONLY:
            assert entry.semantic_ownership_role == SemanticOwnershipRole.NO_TRUTH_OWNERSHIP
            assert entry.underlying_family_ids
        elif entry.projection_or_truth in _NON_CANONICAL_TRUTH_CLASSES:
            assert entry.semantic_ownership_role != SemanticOwnershipRole.CANONICAL_OWNER


def test_sx_p0_q03_no_unknown_semantic_owner_tokens() -> None:
    forbidden = {"UNKNOWN", "TBD", "unknown", "tbd"}
    for entry in STATE_X_FAMILY_INVENTORY:
        if entry.semantic_ownership_role == SemanticOwnershipRole.NO_TRUTH_OWNERSHIP:
            continue
        owner = entry.semantic_owner.upper()
        for token in forbidden:
            assert token not in owner


def test_sx_p0_q04_contract_reference_paths_exist() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        for ref in entry.contract_references:
            path = _REPO_ROOT / ref.path
            assert path.is_file(), f"{entry.family_id}: missing contract path {ref.path}"


def test_sx_p0_q05_contract_symbols_resolve_in_exact_paths() -> None:
    checked = 0
    for entry in STATE_X_FAMILY_INVENTORY:
        for ref in entry.contract_references:
            checked += 1
            assert _symbol_resolves_in_exact_path(ref.symbol, ref.path), (
                f"{entry.family_id}: {ref.symbol} not in {ref.path}"
            )
    assert checked == len(_all_contract_references())


def test_sx_p0_q06_contract_reference_shape_and_no_duplicate_pairs() -> None:
    mixed = 0
    for entry in STATE_X_FAMILY_INVENTORY:
        pairs: list[tuple[str, str]] = []
        for ref in entry.contract_references:
            if "/" in ref.symbol or ref.symbol.endswith(".py"):
                mixed += 1
            if "/" not in ref.path or not ref.path.endswith(".py"):
                mixed += 1
            pairs.append((ref.symbol, ref.path))
        assert len(pairs) == len(set(pairs)), f"{entry.family_id}: duplicate contract ref"
    assert mixed == 0


def test_sx_p0_q07_no_compound_pseudo_contract_strings_in_inventory_source() -> None:
    inventory_text = (_REPO_ROOT / "tests/qualification/state_x/inventory.py").read_text(
        encoding="utf-8"
    )
    for fragment in _COMPOUND_PSEUDO_CONTRACT_FRAGMENTS:
        assert fragment not in inventory_text


def test_sx_p0_q08_production_paths_exist_and_shape() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        for rel in entry.production_paths:
            assert "/" in rel
            assert rel.endswith(".py")
            assert (_REPO_ROOT / rel).is_file(), f"missing production path: {rel}"


def test_sx_p0_q09_implementation_symbols_shape_and_resolve() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        paths = _implementation_resolution_paths(entry)
        for symbol in entry.implementation_symbols:
            assert symbol.strip()
            assert "/" not in symbol
            assert not symbol.endswith(".py")
            assert _symbol_resolves(symbol, paths), (
                f"{entry.family_id}: unresolved symbol {symbol}"
            )


def test_sx_p0_q10_projection_or_truth_classified() -> None:
    allowed = frozenset(ProjectionOrTruth)
    for entry in STATE_X_FAMILY_INVENTORY:
        assert entry.projection_or_truth in allowed


def test_sx_p0_q11_projections_coordination_not_mint_authority() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        if entry.projection_or_truth in _NON_CANONICAL_TRUTH_CLASSES:
            assert entry.authority_role != AuthorityRole.MINT_AUTHORITY
            assert entry.identity_role != IdentityRole.MINT_IDENTITY
        if entry.family_id == "SX-F07":
            assert entry.authority_role != AuthorityRole.MINT_AUTHORITY


def test_sx_p0_q12_checkpoint_not_mint_authority_or_identity() -> None:
    checkpoint = _inventory_by_id()["SX-F01"]
    assert checkpoint.authority_role != AuthorityRole.MINT_AUTHORITY
    assert checkpoint.identity_role != IdentityRole.MINT_IDENTITY


def test_sx_p0_q13_known_blockers_exact_set() -> None:
    ids = {blocker.blocker_id for blocker in STATE_X_KNOWN_BLOCKERS}
    assert ids == {"SX-B01", "SX-B02", "SX-B03"}


def test_sx_p0_q14_applicable_frz_mapping_present() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        assert entry.applicable_frz
        for code in entry.applicable_frz:
            assert code.startswith("FRZ-")


def test_sx_p0_q15_backup_restore_responsibility_classified() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        assert entry.backup_restore_responsibility.value
        assert "TBD" not in entry.backup_restore_responsibility.value.upper()
        assert "UNKNOWN" not in entry.backup_restore_responsibility.value.upper()


def test_sx_p0_q16_tenant_audit_disposition_present() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        assert entry.tenant_audit_disposition.value


def test_sx_p0_q17_f15_coordination_only_invariants() -> None:
    f15 = _inventory_by_id()["SX-F15"]
    assert f15.projection_or_truth == ProjectionOrTruth.COORDINATION_ONLY
    assert f15.semantic_ownership_role == SemanticOwnershipRole.NO_TRUTH_OWNERSHIP
    assert f15.authority_role != AuthorityRole.MINT_AUTHORITY
    assert f15.identity_role != IdentityRole.MINT_IDENTITY
    assert f15.contract_references == ()
    assert f15.underlying_family_ids
    assert "SX-F15" not in f15.underlying_family_ids


def test_sx_p0_q18_underlying_family_references_valid() -> None:
    valid = set(MANDATORY_FAMILY_IDS)
    for entry in STATE_X_FAMILY_INVENTORY:
        assert len(entry.underlying_family_ids) == len(set(entry.underlying_family_ids))
        for ref in entry.underlying_family_ids:
            assert ref in valid
            assert ref != entry.family_id


def test_sx_p0_q19_duplicated_truth_gate() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        if entry.projection_or_truth == ProjectionOrTruth.CANONICAL_TRUTH:
            assert entry.semantic_ownership_role == SemanticOwnershipRole.CANONICAL_OWNER
        if entry.projection_or_truth in _NON_CANONICAL_TRUTH_CLASSES:
            assert entry.semantic_ownership_role != SemanticOwnershipRole.CANONICAL_OWNER


def test_sx_p0_q20_no_mixed_production_path_symbol_representation() -> None:
    mixed = 0
    for entry in STATE_X_FAMILY_INVENTORY:
        for value in entry.production_paths:
            if "/" not in value or not value.endswith(".py"):
                mixed += 1
        for value in entry.implementation_symbols:
            if "/" in value or value.endswith(".py"):
                mixed += 1
    assert mixed == 0


def test_sx_p0_q21_production_implementations_field_absent() -> None:
    assert not hasattr(StateFamilyInventoryEntry, "production_implementations")


def test_sx_p0_q22_legacy_canonical_contract_fields_absent() -> None:
    assert not hasattr(StateFamilyInventoryEntry, "canonical_contract")
    assert not hasattr(StateFamilyInventoryEntry, "contract_path")


def test_sx_p0_q23_regression_known_invalid_paths_absent() -> None:
    inventory_text = (_REPO_ROOT / "tests/qualification/state_x/inventory.py").read_text(
        encoding="utf-8"
    )
    for fragment in _KNOWN_INVALID_PATH_FRAGMENTS:
        assert fragment not in inventory_text


def test_sx_p0_q24_no_production_file_changed_since_audited_head() -> None:
    result = subprocess.run(
        ["git", "diff", "--name-only", STATE_X_R3_R3_PRE_AUDIT_HEAD],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not changed:
        return
    scope_allowlist = (
        STATE_X_P0_ALLOWLIST_PATHS
        | STATE_X_R1_ALLOWLIST_PATHS
        | STATE_X_R2_ALLOWLIST_PATHS
        | STATE_X_R2_R1_ALLOWLIST_PATHS
        | STATE_X_R3_R1_ALLOWLIST_PATHS
        | STATE_X_R3_R2_ALLOWLIST_PATHS
        | STATE_X_R3_R3_ALLOWLIST_PATHS
    )
    scoped_child_allowlist = (
        STATE_X_R1_ALLOWLIST_PATHS
        | STATE_X_R2_ALLOWLIST_PATHS
        | STATE_X_R2_R1_ALLOWLIST_PATHS
        | STATE_X_R3_R1_ALLOWLIST_PATHS
        | STATE_X_R3_R2_ALLOWLIST_PATHS
        | STATE_X_R3_R3_ALLOWLIST_PATHS
    )
    for path in changed:
        normalized = path.replace("\\", "/")
        assert normalized in scope_allowlist, (
            f"STATE-X scope violation since R2 pre-audit head: {normalized}"
        )
        if normalized not in scoped_child_allowlist:
            for prefix in _FORBIDDEN_CHANGE_PREFIXES:
                assert not normalized.startswith(prefix), normalized


def test_sx_p0_no_family_mints_authority() -> None:
    mint_count = sum(
        1
        for entry in STATE_X_FAMILY_INVENTORY
        if entry.authority_role == AuthorityRole.MINT_AUTHORITY
    )
    assert mint_count == 0


def test_sx_p0_f07_sqlite_runtime_event_store_path() -> None:
    f07 = _inventory_by_id()["SX-F07"]
    expected = "intergrax/runtime/events/stores/sqlite_runtime_event_store.py"
    assert expected in f07.production_paths
    assert (_REPO_ROOT / expected).is_file()


def test_sx_p0_f12_human_decision_store_path_and_symbol() -> None:
    f12 = _inventory_by_id()["SX-F12"]
    store_path = "intergrax/runtime/human/store.py"
    assert store_path in f12.production_paths
    assert "SQLiteHumanDecisionStore" in f12.implementation_symbols
    assert (_REPO_ROOT / store_path).is_file()


def test_sx_p0_f01_checkpoint_store_path_and_symbol() -> None:
    f01 = _inventory_by_id()["SX-F01"]
    store_path = "intergrax/runtime/long_running/store.py"
    assert store_path in f01.production_paths
    assert "SQLiteTaskCheckpointStore" in f01.implementation_symbols
    assert (_REPO_ROOT / store_path).is_file()


def _assert_family_contract_refs(
    family_id: str,
    expected: tuple[tuple[str, str], ...],
) -> None:
    entry = _inventory_by_id()[family_id]
    actual = tuple((ref.symbol, ref.path) for ref in entry.contract_references)
    assert actual == expected


def test_sx_p0_multi_contract_f02_exact_pairs() -> None:
    _assert_family_contract_refs(
        "SX-F02",
        (
            (
                "RuntimeCheckpoint",
                "intergrax/runtime/long_running/runtime_checkpoint.py",
            ),
            (
                "ExecutionTreeSnapshot",
                "intergrax/runtime/long_running/execution_tree_checkpoint.py",
            ),
        ),
    )


def test_sx_p0_multi_contract_f03_exact_pairs() -> None:
    _assert_family_contract_refs(
        "SX-F03",
        (
            (
                "DecisionCheckpointPersistence",
                "intergrax/runtime/execution/decision_checkpoint_persistence.py",
            ),
            (
                "DecisionCheckpointState",
                "intergrax/contracts/decision_checkpoint.py",
            ),
        ),
    )


def test_sx_p0_multi_contract_f05_exact_pairs() -> None:
    _assert_family_contract_refs(
        "SX-F05",
        (
            (
                "ExecutionTerminalStore",
                "intergrax/contracts/execution_terminal.py",
            ),
            (
                "ExecutionTerminalRecord",
                "intergrax/contracts/execution_terminal.py",
            ),
        ),
    )


def test_sx_p0_multi_contract_f08_exact_pairs() -> None:
    _assert_family_contract_refs(
        "SX-F08",
        (
            ("RunTraceStore", "intergrax/contracts/run_trace_store.py"),
            ("RunTraceReader", "intergrax/contracts/run_trace_store.py"),
        ),
    )


def test_sx_p0_multi_contract_f09_exact_pairs() -> None:
    _assert_family_contract_refs(
        "SX-F09",
        (
            (
                "RunBudgetPersistence",
                "intergrax/runtime/execution/budget/persistence.py",
            ),
            (
                "ExecutionBudgetLedger",
                "intergrax/runtime/execution/budget/ledger.py",
            ),
        ),
    )


def test_sx_p0_multi_contract_f13_exact_pairs() -> None:
    _assert_family_contract_refs(
        "SX-F13",
        (
            (
                "ScheduledResumePersistence",
                "intergrax/runtime/long_running/scheduled_resume.py",
            ),
            (
                "SchedulerLedger",
                "intergrax/runtime/long_running/persistence_contract.py",
            ),
        ),
    )


@pytest.mark.parametrize("blocker_id", ["SX-B01", "SX-B02", "SX-B03"])
def test_sx_p0_blockers_have_owner(blocker_id: str) -> None:
    blocker = next(b for b in STATE_X_KNOWN_BLOCKERS if b.blocker_id == blocker_id)
    assert blocker.owner_child.startswith("STATE-X")
