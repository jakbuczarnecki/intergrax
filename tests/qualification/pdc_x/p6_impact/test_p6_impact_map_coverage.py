# © Artur Czarnecki. All rights reserved.

"""B-P6-G (T-IGX-036-V): the CHANGED-SINCE impact map covers the PDC-touched product modules.

Domain rules live in ``PDC_DOMAIN_IMPACT_RULES`` (``impact_map.py``) and are merged by
``build_impact_map_from_manifest``. They may only name check ids that exist in the shipped manifest; the
manifest itself stays at exactly 85 checks, and the shipped SSOT JSON is the generator output.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit.changed_since import _check_touched, git_changed_files
from intergrax.platform.enterprise_audit.impact_map import (
    PDC_DOMAIN_IMPACT_RULES,
    build_impact_map_from_manifest,
    check_ids_for_changed_path,
    fail_closed_prefixes,
    load_default_impact_map,
)
from intergrax.platform.enterprise_audit.manifest import load_default_manifest

REPO_ROOT = Path(__file__).resolve().parents[4]
LADDER = REPO_ROOT / "docs/project/maintainers/qualification/platform_domain_completeness/PDC_SHA_LADDER.md"
SSOT = REPO_ROOT / "intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json"
_PRODUCT_ROOTS = ("intergrax/", "agents/", "applications/")
_MANIFEST_SIZE = 85

# One representative module per product area the PDC campaign changed (stable, independent of git history).
_REPRESENTATIVE_MODULES = (
    "intergrax/agent_distribution/installation_service.py",
    "intergrax/applications/_shared/security_audit_trail_wiring.py",
    "intergrax/collaborative_work/content_storage.py",
    "intergrax/context/policy/egress.py",
    "intergrax/contracts/artifact_ref.py",
    "intergrax/core/security/raw_secret_gate.py",
    "intergrax/core/plugins/admission.py",
    "intergrax/eval/typed_run.py",
    "intergrax/hosting/engine/engine.py",
    "intergrax/integrations/providers/embedding_provider/ollama/register.py",
    "intergrax/llm_adapters/providers/claude_adapter.py",
    "intergrax/memory/retention_sweep.py",
    "intergrax/queueing/providers/document_store/document_store_task_queue.py",
    "intergrax/rag/retrieval/retrieval_service.py",
    "intergrax/runtime/agent_governance/pipeline.py",
    "intergrax/runtime/architecture/agent_promotion.py",
    "intergrax/runtime/attestation/host_attestation.py",
    "intergrax/runtime/evidence/eval_evidence_contracts.py",
    "intergrax/runtime/execution/agent_runtime_context.py",
    "intergrax/runtime/execution_evidence/attestor.py",
    "intergrax/runtime/integrations/observability.py",
    "intergrax/runtime/llm/model_call_runtime_evidence_adapter.py",
    "intergrax/runtime/long_running/scheduler.py",
    "intergrax/runtime/nexus/artifacts/spill.py",
    "intergrax/runtime/nexus/context/context_engine.py",
    "intergrax/runtime/nexus/engine/runtime_state.py",
    "intergrax/runtime/nexus/orchestration/run_artifact_bundle_builder.py",
    "intergrax/runtime/nexus/tools/plan_context_invocation.py",
    "intergrax/runtime/nexus/tracing/trace_models.py",
    "intergrax/runtime/observability/export_boundary.py",
    "intergrax/runtime/persistence/sqlite_opens.py",
    "intergrax/runtime/registry/agent_registry.py",
    "intergrax/runtime/replay/replay_engine.py",
    "intergrax/runtime/security/security_audit_trail.py",
    "intergrax/runtime/task/worker_bootstrap.py",
    "intergrax/runtime/vendor_knowledge/contribution_catalog.py",
    "intergrax/skills/host_lifecycle.py",
    "intergrax/tools/providers/rag/service.py",
    "intergrax/tools/providers/storage/service.py",
)


def _manifest_ids() -> frozenset[str]:
    return frozenset(check.check_id for check in load_default_manifest())


def _pdc_touched_product_modules() -> list[str]:
    match = re.search(r"\|\s*BASELINE_SHA\s*\|\s*`([0-9a-f]{40})`", LADDER.read_text(encoding="utf-8"))
    assert match is not None, "BASELINE_SHA row not found in PDC_SHA_LADDER.md"
    sha = match.group(1)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=REPO_ROOT, capture_output=True, check=False
    )
    assert ancestor.returncode == 0, f"PDC BASELINE_SHA {sha} is not an ancestor of HEAD (shallow clone?)"
    changed = git_changed_files(REPO_ROOT, sha)
    return sorted(p for p in changed if p.endswith(".py") and p.startswith(_PRODUCT_ROOTS))


def test_manifest_still_has_exactly_85_checks() -> None:
    assert len(load_default_manifest()) == _MANIFEST_SIZE


def test_every_pdc_domain_rule_names_only_existing_manifest_check_ids() -> None:
    manifest_ids = _manifest_ids()
    assert PDC_DOMAIN_IMPACT_RULES, "the PDC domain rule table must not be empty"
    for prefix, ids in PDC_DOMAIN_IMPACT_RULES:
        assert ids, f"{prefix}: a rule must name at least one check id"
        assert len(set(ids)) == len(ids), f"{prefix}: duplicate check id"
        unknown = sorted(set(ids) - manifest_ids)
        assert not unknown, f"{prefix}: check ids missing from the manifest: {unknown}"


def test_pdc_domain_rule_prefixes_are_unique_normalized_product_prefixes() -> None:
    prefixes = [prefix for prefix, _ in PDC_DOMAIN_IMPACT_RULES]
    assert len(set(prefixes)) == len(prefixes), "duplicate rule prefix"
    for prefix in prefixes:
        assert prefix.startswith("intergrax/"), prefix
        assert not prefix.endswith("/") and "\\" not in prefix, prefix
        assert not any(prefix.startswith(fc.rstrip("/")) for fc in fail_closed_prefixes()), (
            f"{prefix}: domain rules must not touch the fail-closed ENT-AUDIT prefixes"
        )


def test_generator_output_contains_every_domain_rule() -> None:
    mapping = build_impact_map_from_manifest(load_default_manifest())
    for prefix, ids in PDC_DOMAIN_IMPACT_RULES:
        assert frozenset(ids) <= mapping[prefix], prefix


def test_shipped_ssot_equals_generator_output() -> None:
    mapping = build_impact_map_from_manifest(load_default_manifest())
    expected = {
        "schema_version": 1,
        "path_prefix_to_check_ids": {k: sorted(v) for k, v in sorted(mapping.items())},
    }
    shipped = json.loads(SSOT.read_text(encoding="utf-8"))
    assert shipped == expected, "regenerate with scripts/enterprise_audit/build_impact_map.py --emit-ssot"
    assert SSOT.read_text(encoding="utf-8") == json.dumps(expected, indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize("module", _REPRESENTATIVE_MODULES)
def test_representative_pdc_module_maps_to_a_manifest_check(module: str) -> None:
    ids = check_ids_for_changed_path(module, load_default_impact_map())
    assert ids, f"{module} maps to no check id"
    assert ids <= _manifest_ids()


def test_every_pdc_touched_product_module_maps_to_at_least_one_check() -> None:
    modules = _pdc_touched_product_modules()
    assert modules, "the PDC campaign changed no product module; the test would be vacuous"
    prefixes = load_default_impact_map()
    checks = tuple(load_default_manifest())
    unmapped = [
        path
        for path in modules
        if not check_ids_for_changed_path(path, prefixes)
        and not any(_check_touched(check, frozenset({path})) for check in checks)
    ]
    assert unmapped == [], f"{len(unmapped)} PDC-touched product modules map to no check, e.g. {unmapped[:5]}"


@pytest.mark.parametrize(
    "path",
    (
        "intergrax/platform/enterprise_audit/impact_map.py",
        "intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json",
    ),
)
def test_impact_map_own_files_select_only_existing_manifest_check_ids(path: str) -> None:
    """A change to the map itself (it lands in the PDC range) must not select ids the manifest lacks."""
    ids = check_ids_for_changed_path(path, load_default_impact_map())
    assert ids, f"{path} maps to no check id"
    assert ids <= _manifest_ids(), sorted(ids - _manifest_ids())
