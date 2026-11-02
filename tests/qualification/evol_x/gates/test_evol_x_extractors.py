# © Artur Czarnecki. All rights reserved.

"""W2-04: 19 REL extractors (E-01..E-19), dynamic -> UNKNOWN, relative imports, import-only graph is gone."""

from __future__ import annotations

import pytest
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.discovery.extractors import (
    EXTRACTORS,
    MANDATORY_RELS,
    ExtractionContext,
    extractor_manifest,
    module_id_for_path,
    resolve_import_targets,
    scan_source,
)

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_KNOWN = frozenset(
    {
        "module:intergrax.contracts.x",
        "module:intergrax.core.store_impl",
        "module:intergrax.core.shared.util",
        "module:agents.demo.a",
        "module:agents.demo.b",
        "module:agents.demo",
    }
)


def _ctx(path: str, source: str, *, is_test: bool = False) -> ExtractionContext:
    facts = scan_source(source)
    resolved = tuple((rec, *resolve_import_targets(path, rec, _KNOWN)) for rec in facts["imports"])
    return ExtractionContext(
        path=path,
        module_id=module_id_for_path(path),
        facts=facts,
        known_modules=_KNOWN,
        known_suites=frozenset(),
        owner_for_path=lambda p: ("EVOL-X-A",),
        is_test=is_test,
        resolved=resolved,
    )


def _edges(path: str, source: str, *, is_test: bool = False) -> set[tuple[str, str, str]]:
    ctx = _ctx(path, source, is_test=is_test)
    out: set[tuple[str, str, str]] = set()
    for extractor in EXTRACTORS:
        for edge in extractor.fn(ctx).edges:
            out.add((edge.source_id, edge.target_id, edge.rel))
    return out


def test_evol_x_extractor_registry_is_19_and_covers_mandatory_rels() -> None:
    assert [e.extractor_id for e in EXTRACTORS] == [f"E-{i:02d}" for i in range(1, 20)]
    produced = {e.rel for e in EXTRACTORS}
    assert set(MANDATORY_RELS) <= produced
    manifest = extractor_manifest()
    assert manifest["digest"] and len(manifest["extractors"]) == 19


def test_evol_x_imports_relative_and_absolute_resolved() -> None:
    edges = _edges("agents/demo/a.py", "from . import b\nfrom .b import thing\nimport intergrax.contracts.x\n")
    assert ("module:agents.demo.a", "module:agents.demo.b", "IMPORTS") in edges
    assert ("module:agents.demo.a", "module:intergrax.contracts.x", "IMPORTS") in edges
    assert ("module:agents.demo.a", "module:intergrax.contracts.x", "CONSUMES_CONTRACT") in edges


def test_evol_x_dynamic_import_non_literal_becomes_unknown_not_dropped() -> None:
    ctx = _ctx("agents/demo/a.py", "import importlib\nname = 'x'\nimportlib.import_module(name)\n__import__(name)\n")
    unknown_edges = [e for x in EXTRACTORS for e in x.fn(ctx).edges if e.rel == "DEPENDS_UNKNOWN"]
    assert len(unknown_edges) == 2
    assert all(e.target_id.startswith("unknown:dynamic:") for e in unknown_edges)


def test_evol_x_dynamic_import_literal_resolved_as_import() -> None:
    edges = _edges("agents/demo/a.py", "import importlib\nimportlib.import_module('agents.demo.b')\n")
    assert ("module:agents.demo.a", "module:agents.demo.b", "IMPORTS") in edges


def test_evol_x_implements_registered_binds_events_persistence() -> None:
    src = (
        "from intergrax.contracts.x import Base\n"
        "from intergrax.core.store_impl import Store\n"
        "from intergrax.core.shared.util import helper\n"
        "import os, json\n"
        "class Impl(Base):\n"
        "    pass\n"
        "def on_ready():\n"
        "    pass\n"
        "register_thing(Store)\n"
        "emit('order.created')\n"
        "os.getenv('EVOL_KEY')\n"
        "p = 'conf/settings.json'\n"
        "open('state.json', 'w')\n"
    )
    edges = _edges("agents/demo/a.py", src)
    rels = {rel for _s, _t, rel in edges}
    assert {
        "IMPLEMENTS",
        "REGISTERED_IN",
        "SHARES",
        "BINDS",
        "EMITS_EVENT",
        "HANDLES_EVENT",
        "PERSISTS",
        "OWNED_BY",
    } <= rels
    assert ("module:agents.demo.a", "config:env:EVOL_KEY", "BINDS") in edges
    assert ("module:agents.demo.a", "config:conf/settings.json", "BINDS") in edges


def test_evol_x_tested_by_and_gated_by_for_test_modules() -> None:
    src = "import pytest\nfrom agents.demo.a import f\npytestmark = [pytest.mark.gate]\ndef test_f():\n    pass\n"
    edges = _edges("tests/unit/test_demo.py", src, is_test=True)
    assert ("module:agents.demo.a", "suite:tests/unit/test_demo.py", "TESTED_BY") in edges
    assert ("suite:tests/unit/test_demo.py", "ci:gate", "GATED_BY") in edges


def test_evol_x_docstring_refs_are_not_string_binds() -> None:
    src = '"""mentions intergrax.contracts.x only here"""\nvalue = 1\n'
    edges = _edges("agents/demo/a.py", src)
    assert not [e for e in edges if e[2] == "BINDS"]


def test_evol_x_real_graph_has_semantic_rels_beyond_imports(evol_inventory: ClosedWorldInventory) -> None:
    rels = {e.rel for e in evol_inventory.edges}
    assert set(MANDATORY_RELS) <= rels, set(MANDATORY_RELS) - rels
    assert len(rels) >= 16
    assert any(n.category.startswith("symbol_") for n in evol_inventory.nodes)
    assert any(n.category == "unknown" for n in evol_inventory.nodes) == bool(evol_inventory.unknowns)
