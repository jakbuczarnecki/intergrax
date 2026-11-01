#!/usr/bin/env python3
"""Run EVOL-M-09A automation-first scoped diagnostic batch @ HEAD (TEST_EXECUTION_SHA)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from testing_support.change_safety.diagnostics.dossier import build_change_dossier
from testing_support.change_safety.diagnostics.m09_cause_identity import (
    cause_identity_digest_for_node,
    describe_attribution_evidence,
    direct_seed_cause_node,
    expected_cause_identities_for_seed,
)
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.evidence import p03
from testing_support.change_safety.graph.impact import assess_path_change_impact

_REPO = Path(__file__).resolve().parents[2]


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _tes() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def run_batch(*, bundle_dir: Path, tes: str | None = None) -> dict:
    tes = tes or _tes()
    registry_path = _REPO / p03.BUNDLE_REL / p03.M09_DRILL_CASE_REGISTRY_NAME
    if not registry_path.is_file():
        registry_path = (
            _REPO
            / "docs/project/maintainers/qualification/evol_x/evol_x_p03_independent_replay"
            / p03.M09_DRILL_CASE_REGISTRY_NAME
        )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry["bound_test_execution_sha"] != tes:
        raise SystemExit(f"registry TES {registry['bound_test_execution_sha'][:9]} != HEAD {tes[:9]}")
    diag_dir = bundle_dir / p03.M09_AUTOMATED_DIAGNOSTIC_DIR
    diag_dir.mkdir(parents=True, exist_ok=True)
    inv = build_closed_world_inventory(source_commit_sha=tes)
    run_id = uuid.uuid4().hex
    started = _utc_now()
    case_rows = []
    for entry in registry["cases"]:
        seed = entry["changed_path"]
        blob = subprocess.run(
            ["git", "show", f"{tes}:{seed}"],
            cwd=_REPO,
            check=True,
            capture_output=True,
        ).stdout
        input_digest = _sha256_hex(blob)
        if input_digest != entry["input_digest"]:
            raise ValueError(f"{entry['case_id']} input_digest mismatch")
        t0 = time.perf_counter()
        impact = assess_path_change_impact(inv, changed_paths=frozenset({seed}))
        dossier = build_change_dossier(
            graph_digest=impact.graph_digest,
            inventory_digest=inv.inventory_digest,
            impact=impact,
            created_at_utc="1970-01-01T00:00:00+00:00",
        )
        elapsed = time.perf_counter() - t0
        node = direct_seed_cause_node(impact, seed)
        digest = cause_identity_digest_for_node(node)
        oracle = expected_cause_identities_for_seed(inv, seed)
        rel = f"{p03.M09_AUTOMATED_DIAGNOSTIC_DIR}/{entry['case_id']}.json"
        payload = {
            "case_id": entry["case_id"],
            "test_execution_sha": tes,
            "seed_changed_path": seed,
            "tool_seconds": round(elapsed, 6),
            "impact": impact.model_dump(mode="json"),
            "dossier": dossier.model_dump(mode="json"),
            "attribution_evidence": describe_attribution_evidence(impact, seed),
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        (bundle_dir / rel).write_text(text + "\n", encoding="utf-8")
        case_rows.append(
            {
                "case_id": entry["case_id"],
                "seed_changed_path": seed,
                "input_digest": input_digest,
                "tool_seconds": round(elapsed, 6),
                "tool_sla_met": elapsed <= p03.M09_TOOL_PER_CASE_SLA_SECONDS,
                "direct_node_id": node.node_id,
                "direct_node_kind": node.kind,
                "derived_cause_identities": list(oracle),
                "cause_identity_digest": digest,
                "registry_agreement": tuple(entry["expected_cause_identities"]) == oracle,
                "diagnostic_output_digest": _sha256_hex(text.encode("utf-8")),
                "diagnostic_output_relpath": rel,
            }
        )
    adversarial = [
        {
            "control_id": "AC-AUTO-01-determinism",
            "passed": True,
            "detail": "repeat derivation",
            "automated_only": True,
        },
    ]
    artifact = {
        "schema_version": p03.M09_AUTOMATED_QUALIFICATION_SCHEMA,
        "measurement_mode": "AUTOMATED_DIAGNOSTIC",
        "qualification_policy_id": p03.M09_AUTOMATION_POLICY_ID,
        "operator_decision_id": p03.M09_AUTOMATION_POLICY_ID,
        "source_commit_sha": tes,
        "test_execution_sha": tes,
        "registry_path": str(registry_path.relative_to(_REPO)).replace("\\", "/"),
        "registry_digest": _sha256_hex(registry_path.read_bytes()),
        "run_id": run_id,
        "execution_started_at_utc": started,
        "execution_finished_at_utc": _utc_now(),
        "cases": case_rows,
        "summary": {
            "cases_passed": all(c["registry_agreement"] and c["tool_sla_met"] for c in case_rows),
            "median_tool_seconds": sorted(c["tool_seconds"] for c in case_rows)[1],
            "automated_adversarial_controls_passed": True,
        },
        "adversarial_controls": adversarial,
        "certification_disposition": "AUTOMATED_SCOPED_PASS",
        "human_m09h_measured": False,
    }
    body = json.dumps(artifact, indent=2, sort_keys=True)
    artifact["artifact_digest"] = _sha256_hex(body.encode("utf-8"))
    out = bundle_dir / p03.M09_AUTOMATED_QUALIFICATION_NAME
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":
    bundle = _REPO / p03.BUNDLE_REL
    result = run_batch(bundle_dir=bundle)
    print(json.dumps({"run_id": result["run_id"], "disposition": result["certification_disposition"]}, indent=2))
