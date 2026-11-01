# © Artur Czarnecki. All rights reserved.

"""D1/D5/D8 closed-world inventory builder (git-tree universe + owner/exclusion reconciliation)."""

from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass, field
from functools import cached_property
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
) -> ClosedWorldInventory:
    owner_doc = owner_rules if owner_rules is not None else load_evol_x_data_document("owner_rules.v1.json")
    exclusion_doc = (
        exclusion_register
        if exclusion_register is not None
        else load_evol_x_data_document("exclusion_register.v1.json")
    )
    owners = tuple(owner_doc.get("owners", ()))
    exclusion_patterns = accepted_exclusion_patterns(tuple(exclusion_doc.get("entries", ())))

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
            facts = facts_for_file(tracked, use_cache=use_cache)
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
        source_tree_id=_git_tree_id(source_commit_sha),
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
