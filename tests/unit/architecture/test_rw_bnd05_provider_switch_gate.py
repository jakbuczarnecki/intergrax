# © Artur Czarnecki. All rights reserved.

"""RW-plan architecture guard NT-BND-05-2: provider selection stays at the closed list of composition roots.

Tier-1 and Tier-3 product modules may import ``intergrax.integrations.providers*`` or
``intergrax.integrations._shared*`` only when they are on the closed list of provider
composition roots (DP-18; initial membership ratified from the W-B classification, Q-WB-04).
The list may shrink but never grow: a new module that imports a provider package fails here
until the Operator adds it to the list. Pure guard: it tightens or loosens no existing gate.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCAN_ROOTS = ("intergrax/runtime", "intergrax/applications", "applications")
_AREA_PREFIX = "intergrax/runtime/vendor_knowledge/"
_PROVIDER_PREFIXES = ("intergrax.integrations.providers", "intergrax.integrations._shared")

# (module path, class): ROOT 19, CONST 22, IMPL 3, DEMO_PROOF 9, SHARED_DEMO 6; the
# provider-contribution area ``intergrax/runtime/vendor_knowledge/`` is one named area.
CLOSED_PROVIDER_ROOTS: tuple[tuple[str, str], ...] = (
    ("applications/attestation_demo/host/factory.py", "SHARED_DEMO"),
    ("applications/attestation_demo/host/integration_wiring.py", "SHARED_DEMO"),
    ("applications/attestation_demo/host/tool_wiring.py", "SHARED_DEMO"),
    ("applications/lab_application/serving/fastapi_router.py", "DEMO_PROOF"),
    ("applications/local_workspace_application/host/background_worker_constructor.py", "ROOT"),
    ("applications/local_workspace_application/host/environment_profile.py", "CONST"),
    ("applications/local_workspace_application/host/factory.py", "SHARED_DEMO"),
    ("applications/local_workspace_application/host/message_bus_wiring.py", "ROOT"),
    ("applications/local_workspace_application/host/settings.py", "CONST"),
    ("applications/local_workspace_application/model_runtime_proof/runner.py", "SHARED_DEMO"),
    ("applications/local_workspace_application/model_runtime_proof/runtime.py", "SHARED_DEMO"),
    ("applications/local_workspace_application/scripts/inspect_elasticsearch_failed_deliveries.py", "DEMO_PROOF"),
    ("applications/local_workspace_application/scripts/run-lkw-background-task-proof.py", "DEMO_PROOF"),
    ("applications/local_workspace_application/scripts/run-lkw-core-platform-proof.py", "DEMO_PROOF"),
    ("applications/local_workspace_application/scripts/run-lkw-file-watcher-e2e-proof.py", "DEMO_PROOF"),
    ("applications/local_workspace_application/scripts/run-lkw-hosting-proof.py", "DEMO_PROOF"),
    ("applications/local_workspace_application/scripts/run-lkw-managed-workspace-live-proof.py", "DEMO_PROOF"),
    ("applications/local_workspace_application/scripts/run-lkw-os-interaction-proof.py", "DEMO_PROOF"),
    ("applications/local_workspace_application/scripts/verify_lkw_mongodb_stack.py", "DEMO_PROOF"),
    ("applications/local_workspace_application/slack_companion/companion.py", "ROOT"),
    ("applications/local_workspace_application/workspaces/connected_source_discovery_atlassian.py", "CONST"),
    ("applications/local_workspace_application/workspaces/connected_source_discovery_google_workspace.py", "CONST"),
    ("applications/local_workspace_application/workspaces/connected_source_discovery_msgraph.py", "CONST"),
    ("applications/local_workspace_application/workspaces/connected_source_discovery_slack.py", "CONST"),
    ("applications/local_workspace_application/workspaces/connected_source_materializer.py", "CONST"),
    ("applications/local_workspace_application/workspaces/connected_source_tenant_binding.py", "CONST"),
    ("applications/local_workspace_application/workspaces/connected_source_wiring.py", "CONST"),
    ("applications/local_workspace_application/workspaces/document_store_factory.py", "ROOT"),
    ("applications/local_workspace_application/workspaces/slack_ask_orchestration.py", "CONST"),
    ("applications/local_workspace_application/workspaces/tenant_connection_auth_composition.py", "ROOT"),
    ("applications/local_workspace_application/workspaces/vendor_knowledge_extension_composition.py", "CONST"),
    ("intergrax/applications/_shared/guardrail_runtime_bridge.py", "ROOT"),
    ("intergrax/applications/_shared/host_queue_execution_wiring.py", "CONST"),
    ("intergrax/applications/_shared/memory_wiring.py", "ROOT"),
    ("intergrax/applications/_shared/modality_celery_wiring.py", "ROOT"),
    ("intergrax/applications/_shared/task_memory_wiring.py", "CONST"),
    ("intergrax/applications/_shared/tenant_storage_wiring.py", "CONST"),
    ("intergrax/runtime/capacity/production_adapters.py", "ROOT"),
    ("intergrax/runtime/events/store.py", "CONST"),
    ("intergrax/runtime/execution/postgresql_decision_checkpoint_persistence.py", "IMPL"),
    ("intergrax/runtime/human/store.py", "CONST"),
    ("intergrax/runtime/integrations/registry_v2.py", "CONST"),
    ("intergrax/runtime/interactions/factory.py", "ROOT"),
    ("intergrax/runtime/interactions/verification/factory.py", "ROOT"),
    ("intergrax/runtime/long_running/notification.py", "ROOT"),
    ("intergrax/runtime/long_running/postgresql_task_checkpoint_store.py", "IMPL"),
    ("intergrax/runtime/long_running/store.py", "CONST"),
    ("intergrax/runtime/migration/human_decision_legacy_disposition.py", "IMPL"),
    ("intergrax/runtime/nexus/tracing/store.py", "CONST"),
    ("intergrax/runtime/notifications/factory.py", "ROOT"),
    ("intergrax/runtime/observability/composition.py", "CONST"),
    ("intergrax/runtime/observability/elasticsearch_export_wiring.py", "ROOT"),
    ("intergrax/runtime/observability/sentry_export_wiring.py", "ROOT"),
    ("intergrax/runtime/persistence/integration_profile_wiring.py", "ROOT"),
    ("intergrax/runtime/persistence/postgresql_opens.py", "ROOT"),
    ("intergrax/runtime/persistence/sqlite_composition.py", "ROOT"),
    ("intergrax/runtime/task_memory/store.py", "CONST"),
    ("intergrax/runtime/tools/reference_idempotency_store.py", "CONST"),
    ("intergrax/runtime/transport/bootstrap.py", "ROOT"),
)


def _imports_provider_package(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            if node.module.startswith(_PROVIDER_PREFIXES):
                return True
        elif isinstance(node, ast.Import):
            if any(alias.name.startswith(_PROVIDER_PREFIXES) for alias in node.names):
                return True
    return False


def _provider_importing_modules() -> set[str]:
    found: set[str] = set()
    for root in _SCAN_ROOTS:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            relative = path.relative_to(_REPO_ROOT).as_posix()
            if "/tests/" in relative or relative.startswith(_AREA_PREFIX):
                continue
            text = path.read_text(encoding="utf-8")
            if "integrations." not in text:
                continue
            if _imports_provider_package(ast.parse(text)):
                found.add(relative)
    return found


def test_closed_list_has_the_ratified_class_counts() -> None:
    counts = Counter(kind for _path, kind in CLOSED_PROVIDER_ROOTS)
    assert dict(counts) == {
        "ROOT": 19,
        "CONST": 22,
        "IMPL": 3,
        "DEMO_PROOF": 9,
        "SHARED_DEMO": 6,
    }
    assert len({path for path, _kind in CLOSED_PROVIDER_ROOTS}) == 59


def test_nt_bnd_05_2_provider_imports_stay_on_the_closed_list() -> None:
    allowed = {path for path, _kind in CLOSED_PROVIDER_ROOTS}
    outside = sorted(_provider_importing_modules() - allowed)
    assert outside == [], "provider package imports outside the closed list of composition roots: " + ", ".join(outside)
