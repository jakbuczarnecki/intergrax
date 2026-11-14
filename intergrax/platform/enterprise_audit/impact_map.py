# © Artur Czarnecki. All rights reserved.

"""CHANGED-SINCE SSOT path → check_id impact map (C024/C029/C072)."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import ManifestCheck

_IMPACT_RESOURCE = "changed_since_impact_map.v1.json"

_FAIL_CLOSED_PREFIXES = (
    "intergrax/platform/enterprise_audit/",
    "scripts/enterprise_audit/",
)


# PDC product-domain prefix -> existing manifest check ids (B-P6-G, T-IGX-036-V). Longest prefix wins, so a more
# specific prefix repeats the ids of its parent area. Every id exists in the shipped manifest (85 checks); each
# rule names the entrypoint that already exercises the area.
PDC_DOMAIN_IMPACT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # ebh_3 dependency ownership gate + ebh_5 factory/plugin one-pass + hardening_3 layer boundary gate.
    (
        "intergrax/agent_distribution",
        (
            "ENT-A-BND-01",
            "ENT-A-OWN-01",
            "ENT-A-OWN-02",
            "ENT-A-OWN-03",
            "ENT-A-PLG-01",
            "ENT-A-PLG-02",
            "ENT-A-PLG-03",
        ),
    ),
    # ebh_2i contract surface purity + ebh_2a public contract boundary + immutable security audit trail script.
    (
        "intergrax/applications",
        (
            "ENT-A-BND-01",
            "ENT-A-CTR-01",
            "ENT-A-CTR-02",
            "ENT-A-CTR-03",
            "ENT-A-TYP-01",
            "ENT-A-SEC-01",
            "ENT-A-SEC-02",
            "VBN-CONTRACT-LEAKAGE",
        ),
    ),
    # state_x durable-state gates (content storage) + ebh_3 dependency ownership gate.
    ("intergrax/collaborative_work", ("ENT-A-OWN-01", "ENT-A-STA-01", "ENT-A-STA-02")),
    # tenant_x r4 context cross-tenant, context view mismatch and context-llm adversarial probes.
    ("intergrax/context", ("ENT-A-BND-01", "TEN-STATE", "TEN-ASYNC", "TEN-CROSS-TENANT")),
    # hardening_3 layer boundary gate + ebh_2i contract surface purity + ebh_2a public contract boundary.
    (
        "intergrax/contracts",
        ("ENT-A-BND-01", "ENT-A-BND-02", "ENT-A-CTR-01", "ENT-A-CTR-02", "ENT-A-CTR-03", "ENT-A-TYP-01"),
    ),
    # factory/plugin one-pass + immutable security audit trail script + config_x/tenant_x credential gates.
    (
        "intergrax/core",
        (
            "ENT-A-PLG-01",
            "ENT-A-PLG-02",
            "ENT-A-PLG-03",
            "ENT-A-SEC-01",
            "ENT-A-SEC-02",
            "TEN-CONFIG",
            "TEN-CREDENTIALS",
            "VBN-CORE-HARDCODING",
        ),
    ),
    # ebh_3 dependency ownership + ebh_2i contract surface purity + ebh_2a public contract boundary.
    ("intergrax/eval", ("ENT-A-OWN-01", "ENT-A-CTR-01", "ENT-A-TYP-01")),
    # ee_a1 execution engine ownership + harness_w5 observability transport gate + layer boundary gate.
    (
        "intergrax/hosting",
        (
            "ENT-A-BND-01",
            "ENT-A-OWN-01",
            "ENT-A-EXE-01",
            "ENT-A-EXE-02",
            "ENT-A-EXE-03",
            "ENT-A-OBS-01",
            "ENT-A-OBS-02",
        ),
    ),
    # ebh_5 replaceability matrix + tenant_x provider resolver mismatch probe + factory/plugin one-pass.
    ("intergrax/integrations", ("ENT-A-PLG-01", "TEN-PROVIDER", "VBN-REPLACEABILITY-MATRIX", "VBN-SILENT-FALLBACK")),
    # ebh_5 replaceability matrix + config_x shadow-selection gate + ebh_2i contract purity + tenant provider.
    (
        "intergrax/llm_adapters",
        (
            "ENT-A-CTR-01",
            "ENT-A-TYP-01",
            "ENT-A-PLG-01",
            "TEN-PROVIDER",
            "VBN-CONTRACT-LEAKAGE",
            "VBN-REPLACEABILITY-MATRIX",
            "VBN-SHADOW-SELECTION",
            "VBN-SILENT-FALLBACK",
        ),
    ),
    # state_x p0/r3 durable-state gates + tenant_x r4 state probe.
    ("intergrax/memory", ("ENT-A-STA-01", "ENT-A-STA-02", "ENT-A-STA-03", "TEN-STATE")),
    # state_x r3 redelivery/corrupt-state/same-run-id gates + tenant_x r4 async-queue and vendor-queue probes.
    (
        "intergrax/queueing",
        ("ENT-A-STA-03", "ENT-A-REC-01", "ENT-A-REC-02", "ENT-A-REC-03", "TEN-ASYNC", "TEN-RECOVERY"),
    ),
    # tenant_x r4 cross-tenant/state probes + ebh_2i contract surface purity + contract-leakage gate.
    ("intergrax/rag", ("ENT-A-CTR-01", "TEN-STATE", "TEN-CROSS-TENANT", "VBN-CONTRACT-LEAKAGE")),
    # gov_x2 governance catalog and proof-node gates + ebh_3 dependency ownership gate.
    ("intergrax/runtime/agent_governance", ("ENT-A-OWN-01", "ENT-A-GOV-01", "ENT-A-GOV-02", "ENT-A-GOV-03")),
    # hardening_3 layer boundary + ee_a1 execution engine ownership + ebh_3 ownership + gov_x2 governance gates.
    (
        "intergrax/runtime/architecture",
        (
            "ENT-A-BND-01",
            "ENT-A-BND-02",
            "ENT-A-OWN-01",
            "ENT-A-OWN-02",
            "ENT-A-OWN-03",
            "ENT-A-EXE-01",
            "ENT-A-EXE-02",
            "ENT-A-EXE-03",
            "ENT-A-GOV-01",
        ),
    ),
    # immutable security audit trail script + tenant_x r4 functional-evidence tenant probe.
    ("intergrax/runtime/attestation", ("ENT-A-SEC-01", "ENT-A-SEC-02", "TEN-EVIDENCE")),
    # tenant_x r4 functional-evidence tenant probe + ebh_2i contract surface purity.
    ("intergrax/runtime/evidence", ("ENT-A-CTR-01", "TEN-EVIDENCE")),
    # ee_a1 execution engine ownership + harness_w4 production tool boundedness + trace_x recorder tenant gate.
    (
        "intergrax/runtime/execution",
        ("ENT-A-EXE-01", "ENT-A-EXE-02", "ENT-A-EXE-03", "ENT-A-REL-01", "ENT-A-OBS-03", "TEN-TRACE"),
    ),
    # immutable security audit trail script + tenant_x r4 functional-evidence tenant probe.
    ("intergrax/runtime/execution_evidence", ("ENT-A-SEC-01", "ENT-A-SEC-02", "TEN-EVIDENCE")),
    # harness_w5 typed observability transport gate + ebh_5 observability/ops coupling gate.
    ("intergrax/runtime/integrations", ("ENT-A-OBS-01", "ENT-A-OBS-02", "VBN-OBS-OPS-COUPLING")),
    # ee_a1 execution engine ownership + harness_w4 tool boundedness + tenant provider probe + trace_x recorder.
    ("intergrax/runtime/llm", ("ENT-A-EXE-01", "ENT-A-EXE-02", "ENT-A-REL-01", "TEN-PROVIDER", "TEN-TRACE")),
    # harness_w4 tool boundedness + state_x r3 redelivery gate + tenant_x r4 async/background-identity probes.
    (
        "intergrax/runtime/long_running",
        ("ENT-A-EXE-01", "ENT-A-REL-01", "ENT-A-REC-02", "TEN-ASYNC", "TEN-AUTHORITY", "TEN-RECOVERY"),
    ),
    # state_x durable-state gates + tenant_x r4 state and functional-evidence probes (artifact store).
    (
        "intergrax/runtime/nexus/artifacts",
        ("ENT-A-STA-01", "ENT-A-STA-02", "ENT-A-STA-03", "ENT-A-REC-03", "TEN-EVIDENCE", "TEN-STATE"),
    ),
    # tenant_x r4 context cross-tenant, context view mismatch and context-llm probes.
    ("intergrax/runtime/nexus/context", ("ENT-A-BND-01", "TEN-STATE", "TEN-ASYNC", "TEN-CROSS-TENANT")),
    # ee_a1 execution engine ownership certification + state_x projections gate.
    ("intergrax/runtime/nexus/engine", ("ENT-A-EXE-01", "ENT-A-EXE-02", "ENT-A-EXE-03", "ENT-A-STA-02")),
    # ee_a1 execution engine ownership certification + harness_01 tool budget gate.
    ("intergrax/runtime/nexus/orchestration", ("ENT-A-EXE-01", "ENT-A-EXE-02", "ENT-A-EXE-03", "ENT-A-REL-02")),
    # harness_w4 production tool boundedness + harness_01 tool budget and zero-bypass gates.
    ("intergrax/runtime/nexus/tools", ("ENT-A-REL-01", "ENT-A-REL-02", "ENT-A-REL-03")),
    # harness_w5 observability transport gate + trace_x recorder tenant gate.
    ("intergrax/runtime/nexus/tracing", ("ENT-A-OBS-01", "ENT-A-OBS-02", "ENT-A-OBS-03", "TEN-TRACE")),
    # harness_w5 typed observability transport gate + trace_x recorder tenant gate + obs/ops coupling gate.
    (
        "intergrax/runtime/observability",
        ("ENT-A-OBS-01", "ENT-A-OBS-02", "ENT-A-OBS-03", "TEN-TRACE", "VBN-OBS-OPS-COUPLING"),
    ),
    # state_x durable-state and corrupt-state gates + ebh_5 sqlite-effective gate.
    (
        "intergrax/runtime/persistence",
        ("ENT-A-STA-01", "ENT-A-STA-03", "ENT-A-REC-01", "ENT-A-REC-03", "TEN-STATE", "VBN-SQLITE-EFFECTIVE"),
    ),
    # ebh_3 dependency ownership gate + factory/plugin one-pass + config_x shadow-selection gate.
    (
        "intergrax/runtime/registry",
        ("ENT-A-OWN-01", "ENT-A-OWN-02", "ENT-A-OWN-03", "ENT-A-PLG-01", "VBN-SHADOW-SELECTION"),
    ),
    # trace_x recorder tenant gate + state_x redelivery gate (replay models/engine).
    ("intergrax/runtime/replay", ("ENT-A-OBS-03", "ENT-A-REC-02", "ENT-A-STA-01", "TEN-TRACE")),
    # immutable security audit trail script + report redaction test + tenant_x credential/authority probes.
    (
        "intergrax/runtime/security",
        ("ENT-A-SEC-01", "ENT-A-SEC-02", "ENT-A-SEC-03", "TEN-AUTHORITY", "TEN-CREDENTIALS"),
    ),
    # ee_a1 execution engine ownership + tenant_x r4 async and background-identity probes.
    ("intergrax/runtime/task", ("ENT-A-EXE-01", "ENT-A-EXE-02", "TEN-ASYNC", "TEN-AUTHORITY")),
    # ebh_5 replaceability matrix + inventory drift + behavioral reachability gates.
    ("intergrax/runtime/vendor_knowledge", ("VBN-INVENTORY-DRIFT", "VBN-REACHABILITY", "VBN-REPLACEABILITY-MATRIX")),
    # factory/plugin one-pass + ebh_3 dependency ownership gate.
    ("intergrax/skills", ("ENT-A-OWN-01", "ENT-A-PLG-01", "ENT-A-PLG-02", "ENT-A-PLG-03")),
    # harness_w4 production tool boundedness + harness_01 budget gates + tenant_x r4 cross-tenant probe.
    (
        "intergrax/tools",
        ("ENT-A-REL-01", "ENT-A-REL-02", "ENT-A-REL-03", "TEN-CROSS-TENANT", "VBN-REPLACEABILITY-MATRIX"),
    ),
)


# Tier-3 root ``applications/`` prefix -> existing manifest check ids (B-P6-J, T-IGX-040-A). Kept apart from
# ``PDC_DOMAIN_IMPACT_RULES`` because that table holds ``intergrax/`` prefixes only. The root ``applications`` is
# not ``intergrax/applications``: longest prefix wins, so this rule matches paths under ``applications/`` only. It
# repeats the ids of the ``intergrax/applications`` rule (ebh_2i contract surface purity, ebh_2a public contract
# boundary, hardening_3 layer boundary gate, immutable security audit trail script); every id exists in the manifest.
PDC_TIER3_ROOT_IMPACT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "applications",
        (
            "ENT-A-BND-01",
            "ENT-A-CTR-01",
            "ENT-A-CTR-02",
            "ENT-A-CTR-03",
            "ENT-A-TYP-01",
            "ENT-A-SEC-01",
            "ENT-A-SEC-02",
            "VBN-CONTRACT-LEAKAGE",
        ),
    ),
)


def fail_closed_prefixes() -> tuple[str, ...]:
    return _FAIL_CLOSED_PREFIXES


def load_impact_map_from_path(path: Path) -> dict[str, frozenset[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    prefixes: dict[str, frozenset[str]] = {}
    for prefix, ids in raw.get("path_prefix_to_check_ids", {}).items():
        normalized = str(prefix).replace("\\", "/").rstrip("/")
        prefixes[normalized] = frozenset(str(i) for i in ids)
    return prefixes


def load_default_impact_map() -> dict[str, frozenset[str]]:
    pkg = resources.files("intergrax.platform.enterprise_audit")
    text = (pkg / "data" / _IMPACT_RESOURCE).read_text(encoding="utf-8")
    raw = json.loads(text)
    prefixes: dict[str, frozenset[str]] = {}
    for prefix, ids in raw.get("path_prefix_to_check_ids", {}).items():
        normalized = str(prefix).replace("\\", "/").rstrip("/")
        prefixes[normalized] = frozenset(str(i) for i in ids)
    return prefixes


def build_impact_map_from_manifest(checks: tuple[ManifestCheck, ...]) -> dict[str, frozenset[str]]:
    """Derive longest-prefix owners from manifest entrypoints + platform module paths."""
    mapping: dict[str, set[str]] = {}

    def add(prefix: str, check_id: str) -> None:
        key = prefix.replace("\\", "/").rstrip("/")
        mapping.setdefault(key, set()).add(check_id)

    for check in checks:
        ep = check.entrypoint.replace("\\", "/")
        if "::" in ep:
            ep = ep.split("::", 1)[0]
        add(ep, check.check_id)
        parts = Path(ep).parts
        if len(parts) > 1:
            add("/".join(parts[:-1]), check.check_id)

    platform_rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("intergrax/platform/enterprise_audit/changed_since.py", ("C024", "C025", "C029", "C072")),
        # The impact map and its SSOT own CHANGED-SINCE selection; the evol_x selection gates exercise them. These two
        # paths name real manifest ids (the legacy C0xx ids above are not manifest ids, so a change here would select
        # an id the manifest does not have).
        ("intergrax/platform/enterprise_audit/impact_map.py", ("ENT-A-EVOL-01", "ENT-A-EVOL-02", "ENT-A-EVOL-03")),
        (
            "intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json",
            ("ENT-A-EVOL-01", "ENT-A-EVOL-02", "ENT-A-EVOL-03"),
        ),
        ("intergrax/platform/enterprise_audit/sandbox.py", ("C014", "C015", "C016", "C017")),
        ("intergrax/platform/enterprise_audit/report.py", ("C013",)),
        (
            "intergrax/platform/enterprise_audit/data/schemas/enterprise_audit_report.v1.schema.json",
            ("C013", "C026", "C027"),
        ),
        ("intergrax/platform/enterprise_audit/baseline.py", ("C022", "C023", "C028", "C026")),
        ("intergrax/platform/enterprise_audit/runner.py", ("C004", "C002", "C009")),
        ("intergrax/platform/enterprise_audit/modes.py", ("C011", "C067", "C071")),
        ("intergrax/platform/enterprise_audit/validators.py", ("C026", "C027", "C069")),
        ("intergrax/platform/enterprise_audit/evidence.py", ("C006", "C068", "C073", "C049")),
        ("intergrax/platform/enterprise_audit/data/enterprise_audit_check_manifest.v1.json", ("C019", "C021", "C030")),
        ("scripts/enterprise_audit/build_impact_map.py", ("C029", "C024")),
        ("scripts/enterprise_audit/run_qual_wave.py", ("C051", "C074")),
    )
    for path, ids in platform_rules:
        for cid in ids:
            add(path, cid)

    for prefix, ids in (*PDC_DOMAIN_IMPACT_RULES, *PDC_TIER3_ROOT_IMPACT_RULES):
        for cid in ids:
            add(prefix, cid)

    return {k: frozenset(v) for k, v in mapping.items()}


def check_ids_for_changed_path(path: str, prefixes: dict[str, frozenset[str]]) -> frozenset[str]:
    normalized = path.replace("\\", "/")
    matched: set[str] = set()
    for prefix, ids in sorted(prefixes.items(), key=lambda kv: len(kv[0]), reverse=True):
        if normalized == prefix or normalized.startswith(prefix + "/"):
            matched.update(ids)
            break
    return frozenset(matched)


def unmapped_fail_closed_paths(
    changed: frozenset[str],
    *,
    prefixes: dict[str, frozenset[str]],
    entrypoint_covered: frozenset[str],
) -> frozenset[str]:
    unmapped: set[str] = set()
    for path in changed:
        if not any(path.startswith(p) for p in _FAIL_CLOSED_PREFIXES):
            continue
        if path in entrypoint_covered:
            continue
        if check_ids_for_changed_path(path, prefixes):
            continue
        unmapped.add(path)
    return frozenset(unmapped)


def selected_check_ids_from_impact(
    changed: frozenset[str],
    *,
    prefixes: dict[str, frozenset[str]],
) -> frozenset[str]:
    selected: set[str] = set()
    for path in changed:
        selected.update(check_ids_for_changed_path(path, prefixes))
    return frozenset(selected)
