# © Artur Czarnecki. All rights reserved.

"""PDC-X F-35 — the evidence-plane security audit trail is a classified COMPAT-X persistence site.

``EvidencePlaneSecurityAuditTrail.append`` builds a fresh canonical ``RuntimeEvent`` from live
arguments and appends it through the injected ``EvidencePersistencePort`` (the canonical
composition wraps that port in the validating port). The positive tests pin its classification;
the negative tests pin that the rule is narrow (path + enclosing function + callee) and that an
unclassified or suspicious site still fails the gates.
"""

from __future__ import annotations

from tests.qualification.compat_x._compat_x_event_evolution_inventory import (
    RuntimeEventCallSiteClassificationEvidence,
    RuntimeEventPersistenceCallSite,
    RuntimeEventWritePathDisposition,
    classify_runtime_event_append_call_site,
    collect_event_evolution_r3_p0_gate_violations,
    count_canonical_production_raw_store_bypasses,
    discover_runtime_event_persistence_call_sites,
)
from tests.qualification.compat_x._compat_x_event_evolution_r3_r1 import (
    RuntimeEventReaderClassificationEvidence,
    RuntimeEventReaderDisposition,
    RuntimeEventReaderSite,
    classify_runtime_event_constructor_site,
    collect_event_evolution_r3_r1_violations,
    discover_runtime_event_reader_sites,
    validate_constructor_positive_evidence_rule_integrity,
)

_TRAIL = "intergrax/runtime/security/security_audit_trail.py"
_CALLEE = "self._evidence.append"


def _trail_persistence_sites() -> tuple[RuntimeEventPersistenceCallSite, ...]:
    return tuple(s for s in discover_runtime_event_persistence_call_sites() if s.module_path == _TRAIL)


def _trail_constructor_sites() -> tuple[RuntimeEventReaderSite, ...]:
    return tuple(
        s
        for s in discover_runtime_event_reader_sites()
        if s.module_path == _TRAIL and s.operation == "RuntimeEvent(...)"
    )


# ---- positive: the real site is classified, with typed evidence --------------------------------


def test_f35_trail_append_site_is_discovered_and_canonical() -> None:
    sites = _trail_persistence_sites()
    assert [s.callee_expr for s in sites] == [_CALLEE]
    site = sites[0]
    assert site.disposition is RuntimeEventWritePathDisposition.CANONICAL_VALIDATING_PATH
    assert (
        site.classification_evidence
        is RuntimeEventCallSiteClassificationEvidence.SECURITY_AUDIT_TRAIL_EVIDENCE_PLANE_PORT
    )


def test_f35_trail_constructor_site_is_discovered_and_non_persisted() -> None:
    sites = _trail_constructor_sites()
    assert len(sites) == 1
    site = sites[0]
    assert site.disposition is RuntimeEventReaderDisposition.NON_PERSISTED_CONSTRUCTION
    assert site.classification_evidence is RuntimeEventReaderClassificationEvidence.SECURITY_AUDIT_TRAIL_EVENT_RECORDER


def test_f35_production_bypass_count_and_constructor_rule_integrity_are_zero() -> None:
    assert count_canonical_production_raw_store_bypasses() == 0
    assert validate_constructor_positive_evidence_rule_integrity() == ()


# ---- negative: the rule is narrow and unclassified sites still fail ----------------------------


def test_f35_append_rule_requires_exact_module_function_and_callee() -> None:
    unclassified = RuntimeEventWritePathDisposition.UNCLASSIFIED
    assert classify_runtime_event_append_call_site(_TRAIL, "append", _CALLEE).disposition is not unclassified
    # another function in the same module
    assert classify_runtime_event_append_call_site(_TRAIL, "list_entries", _CALLEE).disposition is unclassified
    # another callee (a raw backend handle) in the same function
    assert classify_runtime_event_append_call_site(_TRAIL, "append", "self._store.append").disposition is unclassified
    assert classify_runtime_event_append_call_site(_TRAIL, "append", "self._inner.append").disposition is unclassified
    # same function and callee in another module
    other = "intergrax/runtime/security/other_trail.py"
    assert classify_runtime_event_append_call_site(other, "append", _CALLEE).disposition is unclassified


def test_f35_constructor_rule_requires_exact_function_and_rejects_reconstruction() -> None:
    unclassified = RuntimeEventReaderDisposition.UNCLASSIFIED
    assert classify_runtime_event_constructor_site(_TRAIL, "append")[0] is (
        RuntimeEventReaderDisposition.NON_PERSISTED_CONSTRUCTION
    )
    assert classify_runtime_event_constructor_site(_TRAIL, "list_entries")[0] is unclassified
    assert classify_runtime_event_constructor_site("intergrax/runtime/security/other_trail.py", "append")[0] is (
        unclassified
    )
    # a persisted-reconstruction shape in the classified function is still unclassified
    assert classify_runtime_event_constructor_site(_TRAIL, "append", force_persisted_reconstruction=True)[0] is (
        unclassified
    )


def test_f35_unclassified_append_site_still_fails_the_p0_gate() -> None:
    assert collect_event_evolution_r3_p0_gate_violations() == ()
    site = RuntimeEventPersistenceCallSite(
        module_path=_TRAIL,
        lineno=1,
        qualname="list_entries",
        callee_expr=_CALLEE,
        disposition=RuntimeEventWritePathDisposition.UNCLASSIFIED,
    )
    violations = collect_event_evolution_r3_p0_gate_violations(extra_persistence_call_sites=(site,))
    assert any(v.startswith("unclassified persistence append") for v in violations)
    assert any("persistence bypass count" in v for v in violations)


def test_f35_unclassified_constructor_site_still_fails_the_r3_r1_gate() -> None:
    # The combined R3-R1 verdict also carries closed-world inventory findings that are red at the pin
    # (not F-35), so compare against the baseline instead of requiring an empty tuple.
    baseline = collect_event_evolution_r3_r1_violations()
    assert not [v for v in baseline if _TRAIL in v]
    site = RuntimeEventReaderSite(
        module_path=_TRAIL,
        line=1,
        qualname="list_entries",
        operation="RuntimeEvent(...)",
        disposition=RuntimeEventReaderDisposition.UNCLASSIFIED,
        classification_evidence=RuntimeEventReaderClassificationEvidence.UNSPECIFIED,
    )
    injected = collect_event_evolution_r3_r1_violations(extra_reader_sites=(site,))
    assert [v for v in injected if _TRAIL in v and "unclassified" in v]
