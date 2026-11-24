# © Artur Czarnecki. All rights reserved.

"""D1/D5/D8 closed-world inventory builder (git-tree universe + owner/exclusion reconciliation)."""

from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

from testing_support.change_safety.contracts.canonical_json import stable_digest_hex
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.discovery.extractors import (
    DISCOVERY_ROOTS,
    EXTRACTORS,
    TEST_ROOT,
    ExtractionContext,
    GraphEdge,
    GraphNode,
    directory_files_with_blobs,
    extractor_manifest,
    facts_for_file,
    flush_fact_cache,
    git_ls_files_with_blobs,
    module_id_for_path,
    resolve_import_targets,
)

_REPO = Path(__file__).resolve().parents[3]
_EXCLUDED_TOPS = frozenset({"docs", "exports", "build", "dist"})
CONSUMPTION_RELS: tuple[str, ...] = (
    "IMPORTS",
    "CONSUMES_CONTRACT",
    "IMPLEMENTS",
    "REGISTERED_IN",
    "BINDS",
    "COMPOSES",
    "SHARES",
    "PERSISTS",
    "READS_PERSISTED",
)


def path_matches_patterns(rel: str, patterns: tuple[str, ...] | list[str]) -> bool:
    normalized = rel.replace("\\", "/")
    for pattern in patterns:
        if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch("/" + normalized, pattern):
            return True
    return False


def accepted_exclusion_patterns(entries: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    """Patterns that may exclude *tracked* paths.

    ``GENERATED_UNTRACKED`` predicates (X-01/02/05) can by definition only exclude paths absent from
    ``git ls-files``; a tracked path is never "generated untracked", so those rows never shrink the tracked universe.
    """
    out: list[str] = []
    for entry in entries:
        if entry.get("status") == "ACCEPTED" and entry.get("predicate") != "GENERATED_UNTRACKED":
            out.extend(entry.get("glob_patterns", ()))
    return tuple(out)


def owners_for_path(path: str, owners: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    normalized = path.replace("\\", "/")
    matched: list[str] = []
    for owner in owners:
        for prefix in owner.get("paths_prefix", ()):
            p = str(prefix).rstrip("/")
            if normalized == p or normalized.startswith(f"{p}/"):
                matched.append(str(owner["owner_id"]))
    return tuple(sorted(set(matched)))


@dataclass(frozen=True)
class ClosedWorldInventory:
    schema_version: str
    ruleset_version: str
    source_commit_sha: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    source_tree_id: str = ""
    universe_paths: tuple[str, ...] = ()
    excluded_paths: tuple[str, ...] = ()
    test_paths: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    unresolved_imports: tuple[tuple[str, str], ...] = ()
    test_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()
    contract_text_refs: tuple[tuple[str, tuple[str, ...]], ...] = ()
    contract_string_refs: tuple[tuple[str, tuple[str, ...]], ...] = ()
    import_stats: tuple[tuple[str, int], ...] = ()
    k_to_x_edges: tuple[tuple[str, str], ...] = ()
    extractor_digest: str = ""
    roots: tuple[str, ...] = DISCOVERY_ROOTS
    raw_imports: tuple[tuple[str, tuple[str, ...]], ...] = field(default=(), repr=False)

    @cached_property
    def inventory_digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "ruleset_version": self.ruleset_version,
            "source_commit_sha": self.source_commit_sha,
            "source_tree_id": self.source_tree_id,
            "extractor_digest": self.extractor_digest,
            "roots": list(self.roots),
            "nodes": [[n.node_id, n.category, n.path] for n in self.nodes],
            "edges": [[e.source_id, e.target_id, e.rel] for e in self.edges],
            "universe": list(self.universe_paths),
            "excluded": list(self.excluded_paths),
            "tests": list(self.test_paths),
            "unknowns": list(self.unknowns),
        }
        return stable_digest_hex(payload)

    @cached_property
    def module_ids(self) -> frozenset[str]:
        return frozenset(n.node_id for n in self.nodes if n.category == "module")

    @cached_property
    def consumers_reverse(self) -> dict[str, tuple[tuple[str, str], ...]]:
        """target -> ((source, rel), ...) over consumption relations (reverse impact adjacency)."""
        acc: dict[str, list[tuple[str, str]]] = {}
        for e in self.edges:
            if e.rel in CONSUMPTION_RELS:
                acc.setdefault(e.target_id, []).append((e.source_id, e.rel))
        return {k: tuple(sorted(v)) for k, v in acc.items()}

    @cached_property
    def tested_by(self) -> dict[str, tuple[str, ...]]:
        acc: dict[str, list[str]] = {}
        for e in self.edges:
            if e.rel == "TESTED_BY":
                acc.setdefault(e.source_id, []).append(e.target_id)
        return {k: tuple(sorted(set(v))) for k, v in acc.items()}

    def edge_rels(self) -> frozenset[str]:
        return frozenset(e.rel for e in self.edges)

    def with_edges(self, edges: tuple[GraphEdge, ...]) -> ClosedWorldInventory:
        from dataclasses import replace

        return replace(self, edges=edges)


def _git_tree_id(sha: str) -> str:
    if not sha:
        return ""
    result = subprocess.run(
        ["git", "rev-parse", f"{sha}^{{tree}}"], cwd=_REPO, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def build_closed_world_inventory(
    *,
    source_commit_sha: str,
    ruleset_version: str = "ruleset.v1",
    roots: tuple[str, ...] = DISCOVERY_ROOTS,
    include_tests: bool = True,
    use_cache: bool = True,
    owner_rules: dict[str, Any] | None = None,
    exclusion_register: dict[str, Any] | None = None,
    fixture_root: Path | None = None,
) -> ClosedWorldInventory:
    """Build the inventory from git (default) or from a plain directory tree (``fixture_root``)."""
    owner_doc = owner_rules if owner_rules is not None else load_evol_x_data_document("owner_rules.v1.json")
    exclusion_doc = (
        exclusion_register
        if exclusion_register is not None
        else load_evol_x_data_document("exclusion_register.v1.json")
    )
    owners = tuple(owner_doc.get("owners", ()))
    exclusion_patterns = accepted_exclusion_patterns(tuple(exclusion_doc.get("entries", ())))

    if fixture_root is not None:
        use_cache = False
        code_files = list(directory_files_with_blobs(fixture_root, roots))
        test_files = list(directory_files_with_blobs(fixture_root, (TEST_ROOT,))) if include_tests else []
    else:
        code_files = [t for t in git_ls_files_with_blobs(roots)]
        test_files = list(git_ls_files_with_blobs((TEST_ROOT,))) if include_tests else []
    universe = tuple(t.path for t in code_files)
    classified_files = [t for t in code_files if not path_matches_patterns(t.path, exclusion_patterns)]
    excluded = tuple(t.path for t in code_files if path_matches_patterns(t.path, exclusion_patterns))
    test_files = [t for t in test_files if not path_matches_patterns(t.path, exclusion_patterns)]
    known_modules = frozenset(module_id_for_path(t.path) for t in classified_files)
    known_suites = frozenset(f"suite:{t.path}" for t in test_files)

    nodes: dict[str, GraphNode] = {}
    edges: set[GraphEdge] = set()
    unknowns: set[str] = set()
    unresolved_imports: list[tuple[str, str]] = []
    k_to_x: list[tuple[str, str]] = []
    test_ids: list[tuple[str, tuple[str, ...]]] = []
    contract_text: list[tuple[str, tuple[str, ...]]] = []
    contract_str: list[tuple[str, tuple[str, ...]]] = []
    n_internal = 0
    n_total = 0

    def owner_for(path: str) -> tuple[str, ...]:
        return owners_for_path(path, owners)

    for owner in owners:
        nodes[f"owner:{owner['owner_id']}"] = GraphNode(f"owner:{owner['owner_id']}", "owner")

    for is_test, files in ((False, classified_files), (True, test_files)):
        for tracked in files:
            facts = facts_for_file(tracked, use_cache=use_cache, repo_root=fixture_root)
            mid = module_id_for_path(tracked.path)
            if not is_test:
                nodes[mid] = GraphNode(mid, "module", tracked.path)
            resolved = tuple(
                (rec, *resolve_import_targets(tracked.path, rec, known_modules)) for rec in facts["imports"]
            )
            n_total += len(resolved)
            if not is_test:
                for rec, _t, unres in resolved:
                    dotted = rec[0]
                    if rec[1] or not dotted:
                        continue
                    if unres or dotted.split(".")[0] in _EXCLUDED_TOPS:
                        rel = dotted.replace(".", "/")
                        if path_matches_patterns(rel + ".py", exclusion_patterns) or path_matches_patterns(
                            rel + "/__init__.py", exclusion_patterns
                        ):
                            k_to_x.append((mid, dotted))
            n_internal += sum(1 for _r, targets, unres in resolved if targets or unres)
            ctx = ExtractionContext(
                path=tracked.path,
                module_id=mid,
                facts=facts,
                known_modules=known_modules,
                known_suites=known_suites,
                owner_for_path=owner_for,
                is_test=is_test,
                resolved=resolved,
            )
            for extractor in EXTRACTORS:
                result = extractor.fn(ctx)
                for node in result.nodes:
                    nodes.setdefault(node.node_id, node)
                edges.update(result.edges)
                unknowns.update(result.unknowns)
            if not is_test:
                for _rec, _targets, unres in resolved:
                    for dotted in unres:
                        unresolved_imports.append((mid, dotted))
                if facts["test_ids"]:
                    pass
                if facts["text_contract_refs"]:
                    contract_text.append((tracked.path, tuple(facts["text_contract_refs"])))
                strs = [s for s in facts["str_modules"] if s.startswith("intergrax.contracts")]
                if strs:
                    contract_str.append((tracked.path, tuple(strs)))
            else:
                test_ids.append((tracked.path, tuple(facts["test_ids"])))
    for src, dotted in sorted(set(unresolved_imports)):
        uid = f"unknown:import:{dotted}"
        nodes.setdefault(uid, GraphNode(uid, "unknown"))
        edges.add(GraphEdge(src, uid, "DEPENDS_UNKNOWN"))
        unknowns.add(uid)
    flush_fact_cache() if use_cache else None
    return ClosedWorldInventory(
        schema_version="evol_closed_world_inventory.v1",
        ruleset_version=ruleset_version,
        source_commit_sha=source_commit_sha,
        source_tree_id="" if fixture_root is not None else _git_tree_id(source_commit_sha),
        nodes=tuple(sorted(nodes.values(), key=lambda n: n.node_id)),
        edges=tuple(sorted(edges, key=lambda e: (e.source_id, e.target_id, e.rel))),
        universe_paths=universe,
        excluded_paths=excluded,
        test_paths=tuple(t.path for t in test_files),
        unknowns=tuple(sorted(unknowns)),
        unresolved_imports=tuple(sorted(set(unresolved_imports))),
        test_ids=tuple(sorted(test_ids)),
        contract_text_refs=tuple(sorted(contract_text)),
        contract_string_refs=tuple(sorted(contract_str)),
        k_to_x_edges=tuple(sorted(set(k_to_x))),
        import_stats=(("internal_import_records", n_internal), ("total_import_records", n_total)),
        extractor_digest=str(extractor_manifest()["digest"]),
        roots=roots,
    )


TIER_BOUNDARY_RULESET = "tier-boundary.v1"

# Longest-prefix wins; mirrors AGENTS.md tier roots (no allowlist growth for violations).
_PACKAGE_TIER_RULES: tuple[tuple[str, str], ...] = (
    ("intergrax/applications/", "tier3"),
    ("intergrax/runtime/", "tier1"),
    ("applications/", "tier3"),
    ("agents/", "tier2"),
    ("platform_proofs/", "tier3"),
    ("testing_support/change_safety/", "tooling"),
    ("intergrax/", "tier0"),
)

_TIER_ENFORCEMENT_PREFIXES: tuple[str, ...] = ("intergrax/", "agents/", "applications/")
_UNCLASSIFIED_PRODUCT_PREFIXES: tuple[str, ...] = ("intergrax/unscoped/",)


@dataclass(frozen=True)
class TierEnforcementFinding:
    kind: str
    location: str
    detail: str


@dataclass(frozen=True)
class TierEnforcementReport:
    ruleset_version: str
    findings: tuple[TierEnforcementFinding, ...]

    @cached_property
    def enforcement_digest(self) -> str:
        return stable_digest_hex(
            {
                "ruleset_version": self.ruleset_version,
                "findings": [[f.kind, f.location, f.detail] for f in self.findings],
            }
        )


def classify_package_tier(rel_path: str) -> str | None:
    normalized = rel_path.replace("\\", "/")
    for prefix in _UNCLASSIFIED_PRODUCT_PREFIXES:
        if normalized.startswith(prefix):
            return None
    for prefix, tier in _PACKAGE_TIER_RULES:
        stem = prefix.rstrip("/")
        if normalized == stem or normalized.startswith(prefix):
            return tier
    return None


def is_in_universe_dotted(dotted: str) -> bool:
    root = dotted.split(".", 1)[0]
    return root in {"intergrax", "agents", "applications"}


def _dotted_import_target(node_id: str) -> str:
    if node_id.startswith("module:"):
        return node_id.removeprefix("module:")
    if node_id.startswith("unknown:import:"):
        return node_id.removeprefix("unknown:import:")
    return node_id


def forbidden_tier_import_rule(source_tier: str, target_dotted: str) -> str | None:
    if source_tier in {"", "tooling"}:
        return None
    top = target_dotted.split(".", 1)[0]
    if source_tier in ("tier0", "tier1"):
        if top == "applications" or target_dotted.startswith("intergrax.applications"):
            return "tier-boundary:applications-import"
        if top == "agents" or target_dotted.startswith("intergrax.agents"):
            return "tier-boundary:agents-import"
    if source_tier == "tier2":
        if top == "applications" or target_dotted.startswith("intergrax.applications"):
            return "tier-boundary:applications-import"
    return None


def _module_path_by_id(inventory: ClosedWorldInventory) -> dict[str, str]:
    return {n.node_id: n.path for n in inventory.nodes if n.category == "module" and n.path}


@lru_cache(maxsize=1)
def baseline_accepted_exclusion_ids() -> frozenset[str]:
    doc = load_evol_x_data_document("exclusion_register.v1.json")
    return frozenset(
        str(entry["exclusion_id"]) for entry in doc.get("entries", ()) if entry.get("status") == "ACCEPTED"
    )


def detect_forbidden_exclusion_expansion(exclusion_register: dict[str, Any]) -> tuple[str, ...]:
    baseline = baseline_accepted_exclusion_ids()
    expanded: list[str] = []
    for entry in exclusion_register.get("entries", ()):
        if entry.get("status") != "ACCEPTED":
            continue
        exc_id = str(entry["exclusion_id"])
        if exc_id not in baseline:
            expanded.append(exc_id)
    return tuple(sorted(expanded))


def collect_unclassified_dependencies(inventory: ClosedWorldInventory) -> tuple[str, ...]:
    unknown_edges = {(e.source_id, e.target_id) for e in inventory.edges if e.rel == "DEPENDS_UNKNOWN"}
    silent: list[str] = []
    for src, dotted in inventory.unresolved_imports:
        if not is_in_universe_dotted(dotted):
            continue
        uid = f"unknown:import:{dotted}"
        if (src, uid) not in unknown_edges:
            silent.append(f"{src}->{dotted}")
    return tuple(sorted(silent))


def build_tier_enforcement_report(inventory: ClosedWorldInventory) -> TierEnforcementReport:
    findings: list[TierEnforcementFinding] = []
    paths_by_id = _module_path_by_id(inventory)
    import_edges = [e for e in inventory.edges if e.rel == "IMPORTS"]

    for node in inventory.nodes:
        if node.category != "module" or not node.path:
            continue
        if not any(node.path.replace("\\", "/").startswith(p) for p in _TIER_ENFORCEMENT_PREFIXES):
            continue
        tier = classify_package_tier(node.path)
        if tier is None:
            findings.append(
                TierEnforcementFinding(
                    "UNCLASSIFIED_MODULE",
                    node.path,
                    "production-path module has no package→tier classification",
                )
            )

    for edge in import_edges:
        src_path = paths_by_id.get(edge.source_id)
        if not src_path:
            continue
        if not any(src_path.replace("\\", "/").startswith(p) for p in _TIER_ENFORCEMENT_PREFIXES):
            continue
        source_tier = classify_package_tier(src_path)
        if source_tier is None:
            continue
        target_dotted = _dotted_import_target(edge.target_id)
        rule = forbidden_tier_import_rule(source_tier, target_dotted)
        if rule:
            findings.append(
                TierEnforcementFinding(
                    "FORBIDDEN_TIER_IMPORT",
                    src_path,
                    f"{rule}: {edge.source_id} -> {target_dotted}",
                )
            )

    for item in collect_unclassified_dependencies(inventory):
        findings.append(
            TierEnforcementFinding(
                "UNCLASSIFIED_DEPENDENCY",
                item,
                "in-universe import lacks explicit unknown:import classification",
            )
        )

    ordered = tuple(sorted(findings, key=lambda f: (f.kind, f.location, f.detail)))
    return TierEnforcementReport(ruleset_version=TIER_BOUNDARY_RULESET, findings=ordered)
