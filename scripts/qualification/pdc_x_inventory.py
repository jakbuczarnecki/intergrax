# © Artur Czarnecki. All rights reserved.

"""PDC-X P1 closed-world inventory (Pass A structural index + domain anchor links).

Deterministic, read-only scanner. It enumerates the platform surfaces relevant to the
eight PDC-X workstreams, assigns every surface exactly one category, and verifies that
every declared domain anchor (code owner, document, qualification suite) exists.

Categories: ``PDC-D1`` .. ``PDC-D8``, ``PLATFORM-FOUNDATION`` (owned by an already-certified
non-PDC stage or domain pair), ``CONSUMER`` (Tier-2 agents / Tier-3 applications), or
``UNMAPPED`` (a classification gap; ``--check`` fails).

Coverage status per surface: ``INSPECTED`` (semantically inspected during PDC-P0/P1),
``INDEX_ONLY`` (enumerated and categorized; outside PDC semantic scope), ``UNINSPECTED``
(assigned to a PDC domain but not yet semantically inspected -> tracked unknown for P2).

Usage::

    uv run python scripts/qualification/pdc_x_inventory.py --write
    uv run python scripts/qualification/pdc_x_inventory.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

REPO: Final[Path] = Path(__file__).resolve().parents[2]
OUTPUT: Final[Path] = (
    REPO / "docs/project/maintainers/qualification/platform_domain_completeness/PDC_CLOSED_WORLD_INVENTORY.json"
)
SCHEMA_VERSION: Final[str] = "pdc-x-inventory.v1"
VENDORED_MARKER: Final[str] = "docker/runtime-context"

DOMAINS: Final[dict[str, str]] = {
    "PDC-D1": "AGENT-LIFECYCLE",
    "PDC-D2": "CONTEXT-KNOWLEDGE",
    "PDC-D3": "MODEL-RUNTIME",
    "PDC-D4": "HOST-RUNTIME",
    "PDC-D5": "EVALUATION",
    "PDC-D6": "RUNTIME-SECURITY",
    "PDC-D7": "BACKGROUND-SCHED",
    "PDC-D8": "ARTIFACT-LIFECYCLE",
}
FOUNDATION: Final[str] = "PLATFORM-FOUNDATION"
CONSUMER: Final[str] = "CONSUMER"
UNMAPPED: Final[str] = "UNMAPPED"

# --- Tier-0 / Tier-1 package classification ------------------------------------------------

TIER0_PACKAGES: Final[dict[str, str]] = {
    "agent_distribution": "PDC-D1",
    "agent_kit": "PDC-D1",
    "agents": "PDC-D1",
    "skills": "PDC-D1",
    "context": "PDC-D2",
    "knowledge": "PDC-D2",
    "memory": "PDC-D2",
    "rag": "PDC-D2",
    "tokenizers": "PDC-D2",
    "ucl": "PDC-D2",
    "llm": "PDC-D3",
    "llm_adapters": "PDC-D3",
    "model_inference": "PDC-D3",
    "applications": "PDC-D4",
    "fastapi_core": "PDC-D4",
    "harness": "PDC-D4",
    "hosting": "PDC-D4",
    "supervisor": "PDC-D4",
    "eval": "PDC-D5",
    "experiments": "PDC-D5",
    "background_tasks": "PDC-D7",
    "queueing": "PDC-D7",
    "collaborative_work": "PDC-D8",
}
TIER0_FOUNDATION: Final[frozenset[str]] = frozenset(
    {
        "autonomous_work",
        "capability_acquisition",
        "capability_catalog",
        "capability_metering",
        "capability_qualification",
        "cli",
        "codecraft",
        "compat",
        "contracts",
        "core",
        "debug",
        "decision_system",
        "dev_support",
        "distributed",
        "globals",
        "integrations",
        "lab",
        "legacy",
        "marketplace",
        "multimedia",
        "platform",
        "prompts",
        "proof_data",
        "proofs",
        "scaffold",
        "speech_adapters",
        "tools",
        "utils",
        "websearch",
    }
)

TIER1_PACKAGES: Final[dict[str, str]] = {
    "agent_governance": "PDC-D1",
    "plugins": "PDC-D1",
    "registry": "PDC-D1",
    "context": "PDC-D2",
    "context_lifecycle": "PDC-D2",
    "task_memory": "PDC-D2",
    "token_optimization": "PDC-D2",
    "llm": "PDC-D3",
    "decision_verification_stages": "PDC-D5",
    "evaluation": "PDC-D5",
    "attestation": "PDC-D6",
    "policy": "PDC-D6",
    "sandbox": "PDC-D6",
    "security": "PDC-D6",
    "background_execution": "PDC-D7",
    "long_running": "PDC-D7",
    "task": "PDC-D7",
}
TIER1_FOUNDATION: Final[frozenset[str]] = frozenset(
    {
        "adaptive",
        "architecture",
        "cancellation",
        "capacity",
        "codecraft",
        "cognition",
        "config",
        "diagnostics",
        "enterprise_reliability",
        "events",
        "evidence",
        "execution",
        "execution_evidence",
        "external_operations",
        "governance",
        "hooks",
        "human",
        "integrations",
        "interactions",
        "interrupts",
        "invariants",
        "kernel",
        "metrics",
        "middleware",
        "migration",
        "modality",
        "nexus",
        "notifications",
        "observability",
        "organization",
        "persistence",
        "prediction",
        "prevention",
        "reliability",
        "replay",
        "resilience",
        "runtime_inspection",
        "runtime_intelligence",
        "schema",
        "self_healing",
        "tools",
        "transport",
        "user_profile",
        "vendor_knowledge",
        "wiring",
        "workspace",
    }
)

# --- Documentation classification (first matching rule wins; stem compared upper-case) -----

DOC_DOMAIN_RULES: Final[tuple[tuple[str, str], ...]] = (
    ("AGENT_CONTRACTS", "PDC-D1"),
    ("AGENT_DISTRIBUTION", "PDC-D1"),
    ("SKILLS", "PDC-D1"),
    ("PLATFORM_PLUGIN", "PDC-D1"),
    ("CONTEXT_ENGINEERING", "PDC-D2"),
    ("UNIFIED_CONTEXT_LIFECYCLE", "PDC-D2"),
    ("MEMORY", "PDC-D2"),
    ("RAG", "PDC-D2"),
    ("KNOWLEDGE_SOURCE", "PDC-D2"),
    ("TOKEN_OPTIMIZATION", "PDC-D2"),
    ("VLLM_PREFIX_CACHE", "PDC-D2"),
    ("LLM_ADAPTERS", "PDC-D3"),
    ("OLLAMA", "PDC-D3"),
    ("LANGCHAIN", "PDC-D3"),
    ("LANGGRAPH", "PDC-D3"),
    ("POST_LCI", "PDC-D3"),
    ("MODALITY", "PDC-D3"),
    ("APPLICATION_HOSTING", "PDC-D4"),
    ("TIER3_APPLICATION", "PDC-D4"),
    ("APPLICATION_DEPENDENCY", "PDC-D4"),
    ("APPLICATION_RUNTIME_GRAPH", "PDC-D4"),
    ("EXPERIMENTATION", "PDC-D5"),
    ("DECISION_VERIFICATION", "PDC-D5"),
    ("CRITIC_VERIFICATION", "PDC-D5"),
    ("E2E_SCENARIO", "PDC-D5"),
    ("BACKGROUND_TASKS", "PDC-D7"),
    ("COLLABORATIVE_WORK", "PDC-D8"),
)
DOC_FOUNDATION_PREFIXES: Final[tuple[str, ...]] = (
    "ADAPTIVE_HARNESS",
    "ARCHITECTURE_OVERVIEW",
    "AUDIT_IDEAL",
    "AUTONOMOUS_",
    "CAPABILITY_",
    "CODE_CRAFT",
    "COMPAT_X",
    "CONVERSATION_CHANNEL",
    "DECISION_",
    "DEPENDENCY_OWNERSHIP",
    "DIAGNOSTICS",
    "ELASTIC_CAPACITY",
    "ENTERPRISE_",
    "ERL_",
    "EVOL_X",
    "EXTERNAL_EFFECT",
    "GOVERNED_EXECUTION",
    "HARNESS_",
    "IDEAL_HARNESS",
    "INTEGRATIONS",
    "INTERGRAX_",
    "MCP_SERVER",
    "MSGRAPH_",
    "MULTIPLAYER",
    "NEXUS_",
    "OBSERVABILITY",
    "ORCHESTRATION",
    "PLATFORM_ENTERPRISE",
    "PLATFORM_FOUNDATION",
    "PRODUCT_",
    "PROOF_",
    "PROVIDER_CATEGORY",
    "README",
    "REASONING_",
    "RECONCILIATION",
    "RECOVERY_",
    "RELIABILITY_",
    "SCENARIO_",
    "TOOLS",
    "UNCERTAINTY",
    "UNIFIED_EXECUTION",
    "VENDOR_KNOWLEDGE",
    "WEB_CONTENT",
)
DOC_GLOBS: Final[tuple[str, ...]] = (
    "docs/project/architecture/*.md",
    "docs/project/maintainers/plans/*.md",
    "docs/project/capabilities/**/*.md",
)

# --- Qualification suites ---------------------------------------------------------------------

QUALIFICATION_SUITES: Final[dict[str, str]] = {
    "plug_03": "PDC-D1",
    "plug_04": "PDC-D1",
    "plug_final": "PDC-D1",
    "ce_01": "PDC-D2",
    "ce_02": "PDC-D2",
    "memory_behavior": "PDC-D2",
    "host_01": "PDC-D4",
    "tenant_x": "PDC-D6",
    "bg_01": "PDC-D7",
    "sched_01": "PDC-D7",
}
QUALIFICATION_FOUNDATION: Final[frozenset[str]] = frozenset(
    {
        "compat_x",
        "config_x",
        "control_planes",
        "dist_scale_x",
        "dup_x",
        "ebh_5",
        "ebh_6",
        "ebh_7",
        "ent_audit_x",
        "evol_x",
        "existing_capability_configuration",
        "external_contract_compatibility",
        "governance",
        "harness_01",
        "harness_02",
        "inspect_01",
        "mcp_server_x",
        "mp6",
        "multiplayer",
        "pdc_x",
        "prod_q",
        "qual_x",
        "roadmap_replay_x",
        "session_01",
        "state_x",
        "trace_x",
    }
)

# --- Semantic coverage (PDC-P0 planning sweeps + PDC-P1 inspection) ----------------------------

INSPECTED: Final[frozenset[str]] = frozenset(
    {
        # Tier-0 / Tier-1 packages
        "intergrax/agent_distribution",
        "intergrax/agent_kit",
        "intergrax/agents",
        "intergrax/skills",
        "intergrax/context",
        "intergrax/memory",
        "intergrax/rag",
        "intergrax/llm_adapters",
        "intergrax/hosting",
        "intergrax/harness",
        "intergrax/applications",
        "intergrax/eval",
        "intergrax/experiments",
        "intergrax/queueing",
        "intergrax/background_tasks",
        "intergrax/collaborative_work",
        "intergrax/runtime/registry",
        "intergrax/runtime/plugins",
        "intergrax/runtime/context_lifecycle",
        "intergrax/runtime/token_optimization",
        "intergrax/runtime/llm",
        "intergrax/runtime/evaluation",
        "intergrax/runtime/decision_verification_stages",
        "intergrax/runtime/security",
        "intergrax/runtime/sandbox",
        "intergrax/runtime/policy",
        "intergrax/runtime/background_execution",
        "intergrax/runtime/long_running",
        "intergrax/runtime/task",
        # Documentation
        "docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md",
        "docs/project/architecture/AGENT_DISTRIBUTION.md",
        "docs/project/architecture/MEMORY.md",
        "docs/project/architecture/RAG.md",
        "docs/project/architecture/CONTEXT_ENGINEERING.md",
        "docs/project/architecture/UNIFIED_CONTEXT_LIFECYCLE.md",
        "docs/project/architecture/LLM_ADAPTERS.md",
        "docs/project/architecture/OLLAMA_PROVIDER_CLASSIFICATION.md",
        "docs/project/architecture/APPLICATION_HOSTING.md",
        "docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md",
        "docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md",
        "docs/project/architecture/DECISION_VERIFICATION.md",
        "docs/project/architecture/BACKGROUND_TASKS.md",
        "docs/project/architecture/COLLABORATIVE_WORK.md",
        "docs/project/architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md",
        "docs/project/architecture/intergrax_runtime_architecture.md",
        "docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md",
        "docs/project/maintainers/plans/AGENT_DISTRIBUTION.md",
        "docs/project/maintainers/plans/PLATFORM_PLUGINS.md",
        "docs/project/maintainers/plans/MEMORY.md",
        "docs/project/maintainers/plans/RAG.md",
        "docs/project/maintainers/plans/CONTEXT_ENGINEERING.md",
        "docs/project/maintainers/plans/UNIFIED_CONTEXT_LIFECYCLE.md",
        "docs/project/maintainers/plans/LLM_ADAPTERS.md",
        "docs/project/maintainers/plans/APPLICATION_HOSTING.md",
        "docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md",
        "docs/project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md",
        "docs/project/maintainers/plans/CRITIC_VERIFICATION.md",
        "docs/project/maintainers/plans/DECISION_VERIFICATION.md",
        "docs/project/maintainers/plans/BACKGROUND_TASKS.md",
        "docs/project/capabilities/architecture/TOKEN_OPTIMIZATION.md",
        "docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md",
        "docs/project/capabilities/TOKEN_OPTIMIZATION_CLAIMS.md",
        "docs/project/capabilities/architecture/satellites/OLLAMA_NATIVE_ADAPTER_PARITY_MATRIX.md",
        "docs/project/capabilities/architecture/satellites/OLLAMA_NATIVE_ADAPTER_LIVE_PARITY_EVIDENCE.md",
        "docs/project/capabilities/architecture/satellites/NATIVE_OLLAMA_POST_CUTOVER_REGRESSION_CHECKPOINT.md",
        # Qualification suites
        "tests/qualification/ce_01",
        "tests/qualification/ce_02",
        "tests/qualification/host_01",
        "tests/qualification/tenant_x",
        "tests/qualification/bg_01",
        "tests/qualification/sched_01",
    }
)

# --- PDC-P1-02 domain anchor links (every path must exist) -------------------------------------

DOMAIN_LINKS: Final[dict[str, dict[str, object]]] = {
    "PDC-D1": {
        "architecture": [
            "docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md",
            "docs/project/architecture/AGENT_DISTRIBUTION.md",
            "docs/project/architecture/SKILLS.md",
            "docs/project/architecture/PLATFORM_PLUGINS.md",
        ],
        "plans": [
            "docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md",
            "docs/project/maintainers/plans/AGENT_DISTRIBUTION.md",
            "docs/project/maintainers/plans/SKILLS.md",
            "docs/project/maintainers/plans/PLATFORM_PLUGINS.md",
        ],
        "semantic_owners": [
            "intergrax/contracts/agent_contract_meta.py",
            "intergrax/contracts/agent_lifecycle_state.py",
            "intergrax/runtime/registry/agent_registry.py",
            "intergrax/agent_distribution/installation_service.py",
            "intergrax/agent_distribution/activation.py",
            "intergrax/agent_distribution/runtime_revision.py",
            "intergrax/skills/host_lifecycle.py",
        ],
        "composition_owners": ["intergrax/agent_distribution/admin_service.py"],
        "qualification_suites": ["tests/qualification/plug_03", "tests/qualification/plug_final"],
        "prior_certifications": {"SKILL-X": "552e379845a9512c275f3ebe520ac51767c97cf8"},
    },
    "PDC-D2": {
        "architecture": [
            "docs/project/architecture/MEMORY.md",
            "docs/project/architecture/RAG.md",
            "docs/project/architecture/CONTEXT_ENGINEERING.md",
            "docs/project/architecture/UNIFIED_CONTEXT_LIFECYCLE.md",
            "docs/project/capabilities/architecture/TOKEN_OPTIMIZATION.md",
        ],
        "plans": [
            "docs/project/maintainers/plans/MEMORY.md",
            "docs/project/maintainers/plans/RAG.md",
            "docs/project/maintainers/plans/CONTEXT_ENGINEERING.md",
            "docs/project/maintainers/plans/UNIFIED_CONTEXT_LIFECYCLE.md",
            "docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md",
        ],
        "semantic_owners": [
            "intergrax/context/planner.py",
            "intergrax/context/budget/resolver.py",
            "intergrax/runtime/nexus/context/context_compiler.py",
            "intergrax/memory/contracts/memory_control.py",
            "intergrax/runtime/context_lifecycle/contracts.py",
        ],
        "composition_owners": ["intergrax/memory/default_memory_control_plane.py"],
        "qualification_suites": [
            "tests/qualification/ce_01",
            "tests/qualification/ce_02",
            "tests/qualification/memory_behavior",
        ],
        "prior_certifications": {
            "EBH-2G": "ba41ef7592c07df04d75a15cd2efc36fe86bd041",
            "EBH-2H": "86fafaa24ccdfb47faba44ddcec1b6eaec5bf195",
            "HARNESS-RESIDUAL/CE-01": "a8d47a6396c3e5e73a979be74429ed59a93e1551",
            "TENANT-X": "5ccb72fccd4e3820ebdc9c18008afe6f7d59f558",
        },
    },
    "PDC-D3": {
        "architecture": [
            "docs/project/architecture/LLM_ADAPTERS.md",
            "docs/project/architecture/OLLAMA_PROVIDER_CLASSIFICATION.md",
        ],
        "plans": ["docs/project/maintainers/plans/LLM_ADAPTERS.md"],
        "semantic_owners": [
            "intergrax/llm_adapters/contracts/llm_adapter.py",
            "intergrax/llm_adapters/llm_provider_registry.py",
            "intergrax/llm_adapters/registry/model_catalog.py",
            "intergrax/llm_adapters/registry/model_router.py",
            "intergrax/llm_adapters/registry/failover_adapter.py",
            "intergrax/llm_adapters/tracking/llm_usage_track.py",
        ],
        "composition_owners": ["intergrax/llm_adapters/routing/composition.py"],
        "qualification_suites": ["tests/qualification/ebh_5"],
        "prior_certifications": {
            "EBH-2E": "6fa75172468461dc084a2221a5fdff6f23c1fa1c",
            "EBH-5": "33ecf19cc020b112d9e586c87b49e819202f5d6d",
        },
    },
    "PDC-D4": {
        "architecture": [
            "docs/project/architecture/APPLICATION_HOSTING.md",
            "docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md",
        ],
        "plans": [
            "docs/project/maintainers/plans/APPLICATION_HOSTING.md",
            "docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md",
        ],
        "semantic_owners": [
            "intergrax/hosting/contracts/profile.py",
            "intergrax/hosting/engine/engine.py",
            "intergrax/hosting/engine/health.py",
            "intergrax/hosting/shutdown.py",
            "intergrax/hosting/supervisor/supervisor.py",
            "intergrax/applications/contracts/environment_profile/root.py",
        ],
        "composition_owners": [
            "intergrax/hosting/runner.py",
            "intergrax/applications/_shared/environment_wiring.py",
        ],
        "qualification_suites": ["tests/qualification/host_01"],
        "prior_certifications": {"HARNESS-W7/HOST-01": "5d5442dae7769671a7532a973428c8203b492ec6"},
    },
    "PDC-D5": {
        "architecture": [
            "docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md",
            "docs/project/architecture/DECISION_VERIFICATION.md",
        ],
        "plans": [
            "docs/project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md",
            "docs/project/maintainers/plans/DECISION_VERIFICATION.md",
        ],
        "semantic_owners": [
            "intergrax/eval/eval_result.py",
            "intergrax/eval/eval_runner.py",
            "intergrax/runtime/architecture/evaluation_assets.py",
            "intergrax/runtime/architecture/agent_promotion.py",
            "intergrax/runtime/evidence/eval_evidence_contracts.py",
        ],
        "composition_owners": ["intergrax/applications/_shared/evaluation_wiring.py"],
        "qualification_suites": ["tests/qualification/control_planes"],
        "prior_certifications": {"CTRL-X": "318a4d835534ef5c48575f4649551c957dacd8ea"},
    },
    "PDC-D6": {
        "architecture": [
            "docs/project/architecture/GOVERNED_EXECUTION.md",
            "docs/project/maintainers/architecture/EXECUTION_ENGINE_SECURITY_BOUNDARY_AND_THREAT_MODEL.md",
        ],
        "plans": ["docs/project/maintainers/plans/GOVERNED_EXECUTION.md"],
        "semantic_owners": [
            "intergrax/runtime/policy/policy_engine.py",
            "intergrax/contracts/delegation_authority.py",
            "intergrax/integrations/contracts/credential.py",
            "intergrax/integrations/credentials/broker.py",
            "intergrax/runtime/sandbox/isolation_gate.py",
            "intergrax/runtime/security/defense_plugin.py",
        ],
        "composition_owners": [
            "intergrax/applications/_shared/security_assembly_resolver.py",
            "intergrax/applications/_shared/critical_action_signing_wiring.py",
        ],
        "qualification_suites": [
            "tests/qualification/tenant_x",
            "tests/qualification/governance",
            "tests/qualification/prod_q",
        ],
        "prior_certifications": {
            "TENANT-X": "5ccb72fccd4e3820ebdc9c18008afe6f7d59f558",
            "CTRL-X": "318a4d835534ef5c48575f4649551c957dacd8ea",
        },
    },
    "PDC-D7": {
        "architecture": ["docs/project/architecture/BACKGROUND_TASKS.md"],
        "plans": ["docs/project/maintainers/plans/BACKGROUND_TASKS.md"],
        "semantic_owners": [
            "intergrax/queueing/contracts/task_queue.py",
            "intergrax/queueing/worker/registry.py",
            "intergrax/background_tasks/registry.py",
            "intergrax/runtime/background_execution/bootstrap.py",
            "intergrax/runtime/long_running/scheduler.py",
            "intergrax/runtime/long_running/scheduled_resume.py",
        ],
        "composition_owners": [
            "intergrax/runtime/long_running/wiring.py",
            "intergrax/queueing/registry.py",
        ],
        "qualification_suites": ["tests/qualification/bg_01", "tests/qualification/sched_01"],
        "prior_certifications": {
            "HARNESS-W8/BG-01": "3a9b58e6fa1e14ddc9786bd9a447a1d76552d196",
            "HARNESS-RESIDUAL/SCHED-01": "433ff5fb97262c4bad84ea2a4173dc3711d7aa08",
            "TOOL-LIFE-X": "1548640708c850b3fbfca3f2db0e5ecfee4274dd",
        },
    },
    "PDC-D8": {
        "architecture": ["docs/project/architecture/COLLABORATIVE_WORK.md"],
        "plans": ["docs/project/maintainers/plans/COLLABORATIVE_WORK.md"],
        "semantic_owners": [
            "intergrax/contracts/collaborative_work.py",
            "intergrax/collaborative_work/content_storage.py",
            "intergrax/contracts/artifact_ref.py",
            "intergrax/contracts/task_artifacts.py",
            "intergrax/applications/contracts/application_artifacts.py",
            "intergrax/runtime/nexus/artifacts/store_base.py",
            "intergrax/runtime/context_lifecycle/repository.py",
        ],
        "composition_owners": [
            "intergrax/runtime/execution/run_artifact_bundle_composition.py",
            "intergrax/runtime/nexus/orchestration/run_artifact_bundle_builder.py",
        ],
        "qualification_suites": [],
        "prior_certifications": {},
    },
}


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _subdirs(base: Path) -> list[Path]:
    return sorted(
        p for p in base.iterdir() if p.is_dir() and not p.name.startswith((".", "_")) and p.name != "__pycache__"
    )


def _status(rel: str, category: str) -> str:
    if rel in INSPECTED:
        return "INSPECTED"
    if category in DOMAINS:
        return "UNINSPECTED"
    return "INDEX_ONLY"


def _item(kind: str, rel: str, category: str) -> dict[str, str]:
    return {"kind": kind, "path": rel, "category": category, "status": _status(rel, category)}


def classify_doc(stem: str) -> str:
    key = stem.upper()
    for token, domain in DOC_DOMAIN_RULES:
        if token in key:
            return domain
    if key.startswith(DOC_FOUNDATION_PREFIXES):
        return FOUNDATION
    return UNMAPPED


def scan() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for pkg in _subdirs(REPO / "intergrax"):
        if pkg.name == "runtime":
            continue
        cat = TIER0_PACKAGES.get(pkg.name) or (FOUNDATION if pkg.name in TIER0_FOUNDATION else UNMAPPED)
        items.append(_item("package", _rel(pkg), cat))
    for pkg in _subdirs(REPO / "intergrax/runtime"):
        cat = TIER1_PACKAGES.get(pkg.name) or (FOUNDATION if pkg.name in TIER1_FOUNDATION else UNMAPPED)
        items.append(_item("package", _rel(pkg), cat))
    for tier in ("agents", "applications"):
        for pkg in _subdirs(REPO / tier):
            items.append(_item("consumer", _rel(pkg), CONSUMER))
    seen: set[str] = set()
    for pattern in DOC_GLOBS:
        for doc in sorted(REPO.glob(pattern)):
            rel = _rel(doc)
            if rel in seen or VENDORED_MARKER in rel:
                continue
            seen.add(rel)
            items.append(_item("document", rel, classify_doc(doc.stem)))
    for suite in _subdirs(REPO / "tests/qualification"):
        cat = QUALIFICATION_SUITES.get(suite.name) or (
            FOUNDATION if suite.name in QUALIFICATION_FOUNDATION else UNMAPPED
        )
        items.append(_item("qualification_suite", _rel(suite), cat))
    items.sort(key=lambda i: (i["kind"], i["path"]))
    return items


def missing_anchors() -> list[str]:
    missing: list[str] = []
    for domain, links in DOMAIN_LINKS.items():
        for key in ("architecture", "plans", "semantic_owners", "composition_owners", "qualification_suites"):
            paths = links[key]
            assert isinstance(paths, list)
            for rel in paths:
                if not (REPO / rel).exists():
                    missing.append(f"{domain}:{key}:{rel}")
    return missing


def build() -> dict[str, object]:
    items = scan()
    summary: dict[str, dict[str, int]] = {}
    for item in items:
        bucket = summary.setdefault(item["category"], {"INSPECTED": 0, "INDEX_ONLY": 0, "UNINSPECTED": 0})
        bucket[item["status"]] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "domains": DOMAINS,
        "excluded_vendored_marker": VENDORED_MARKER,
        "summary": dict(sorted(summary.items())),
        "domain_links": DOMAIN_LINKS,
        "items": items,
    }


def render(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def check(payload: dict[str, object]) -> list[str]:
    errors: list[str] = []
    items = payload["items"]
    assert isinstance(items, list)
    errors.extend(f"UNMAPPED:{i['path']}" for i in items if i["category"] == UNMAPPED)
    errors.extend(f"VENDORED:{i['path']}" for i in items if VENDORED_MARKER in i["path"])
    errors.extend(f"MISSING_ANCHOR:{m}" for m in missing_anchors())
    if not OUTPUT.is_file():
        errors.append(f"MISSING_OUTPUT:{_rel(OUTPUT)}")
    elif OUTPUT.read_text(encoding="utf-8") != render(payload):
        errors.append(f"STALE_OUTPUT:{_rel(OUTPUT)} (re-run with --write)")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the inventory JSON")
    mode.add_argument("--check", action="store_true", help="fail on gaps or a stale committed JSON")
    args = parser.parse_args(argv)
    payload = build()
    if args.write:
        OUTPUT.write_text(render(payload), encoding="utf-8", newline="\n")
        print(f"[OK] wrote {_rel(OUTPUT)} ({len(payload['items'])} items)")  # type: ignore[arg-type]
        return 0
    errors = check(payload)
    if errors:
        for err in errors:
            print(f"[ERROR] {err}", file=sys.stderr)
        return 1
    print(f"[OK] pdc_x_inventory: {len(payload['items'])} items, 0 UNMAPPED, anchors resolved")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
