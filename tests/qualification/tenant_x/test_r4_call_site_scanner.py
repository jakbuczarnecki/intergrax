# © Artur Czarnecki. All rights reserved.

"""Unit tests for R4 fail-closed AST call-site scanner (synthetic sources)."""

from __future__ import annotations

import pytest

from tests.qualification.tenant_x._r4_call_site_scanner import (
    R4ProductionScanResult,
    discover_call_sites_in_module,
    discover_r4_calls_for_callee,
    r4_scan_accounting_holds,
    r4_unresolved_protected_call_sites,
    scan_r4_production_modules,
)

pytestmark = [pytest.mark.unit, pytest.mark.qualification]


def test_r4_scan_dotted_import_without_alias() -> None:
    source = """
import intergrax.runtime.events.llm_call_recording

def f():
    intergrax.runtime.events.llm_call_recording.record_llm_call_runtime_event(tenant_id=x)
"""
    sites = discover_call_sites_in_module("intergrax/synthetic/dotted.py", source)
    llm = [s for s in sites if s.callee == "record_llm_call_runtime_event"]
    assert len(llm) == 1
    assert llm[0].resolution == "resolved"


def test_r4_scan_module_alias_import() -> None:
    source = """
import intergrax.runtime.events.llm_call_recording as recording

def f():
    recording.record_llm_call_runtime_event(tenant_id=x)
"""
    sites = discover_call_sites_in_module("intergrax/synthetic/mod_alias.py", source)
    llm = [s for s in sites if s.callee == "record_llm_call_runtime_event"]
    assert len(llm) == 1
    assert llm[0].resolution == "resolved"


def test_r4_scan_direct_symbol_alias_import() -> None:
    source = """
from intergrax.runtime.events.llm_call_recording import (
    record_llm_call_runtime_event as record_event,
)

def f():
    record_event(tenant_id=x)
"""
    sites = discover_call_sites_in_module("intergrax/synthetic/sym_alias.py", source)
    llm = [s for s in sites if s.callee == "record_llm_call_runtime_event"]
    assert len(llm) == 1
    assert llm[0].resolution == "resolved"


def test_r4_scan_ambiguous_star_import_fail_closed() -> None:
    source = """
from intergrax.runtime.events.llm_call_recording import *

def f():
    record_llm_call_runtime_event(tenant_id=x)
"""
    sites = discover_call_sites_in_module("intergrax/synthetic/star.py", source)
    llm = [s for s in sites if s.callee == "record_llm_call_runtime_event"]
    assert len(llm) == 1
    assert llm[0].resolution == "ambiguous_star_import"
    unresolved = r4_unresolved_protected_call_sites(
        R4ProductionScanResult((), (), (), tuple(sites))
    )
    assert len(unresolved) == 1


def test_r4_scan_parse_failure_is_blocking() -> None:
    with pytest.raises(SyntaxError):
        discover_call_sites_in_module("intergrax/synthetic/broken.py", "def f(:\n")


def test_r4_scan_reconciliation_on_current_head() -> None:
    scan = scan_r4_production_modules()
    assert r4_scan_accounting_holds(scan)
    assert len(scan.failures) == 0
    assert len(scan.eligible_modules) == len(scan.successfully_scanned)


def test_r4_scan_current_head_four_production_call_sites() -> None:
    ctx = discover_r4_calls_for_callee("record_context_assembly")
    direct = discover_r4_calls_for_callee("record_llm_call_runtime_event")
    wrapper = discover_r4_calls_for_callee("maybe_record_llm_call_from_usage_end")
    resolved = [s for s in (*ctx, *direct, *wrapper) if s.resolution == "resolved"]
    assert len(resolved) == 4
    assert len(r4_unresolved_protected_call_sites(scan_r4_production_modules())) == 0
