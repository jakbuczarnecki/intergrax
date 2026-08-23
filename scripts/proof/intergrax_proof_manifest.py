# © Artur Czarnecki. All rights reserved.

"""Canonical Intergrax proof manifest — single source of truth (PUBLIC-PROOF-GATE-1)."""

from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import ValidationError

from scripts.proof.intergrax_platform_proof_discovery import (
    PlatformProofDiscoveryError,
    discover_platform_proof_descriptors,
    merge_static_and_discovered_entries,
)
from scripts.proof.intergrax_proof_contracts import (
    EnvRequirement,
    EnvRequirementKind,
    IntergraxProofManifest,
    ProofArgvCommand,
    ProofManifestEntry,
    ProofProfile,
    ProofSafetyClass,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LKW_SCRIPTS = "applications/local_workspace_application/scripts"
_TOKEN_OFFLINE_CONFIG = "scripts/proof/configs/runtime_token_optimization_offline_smoke.toml"


def _uv_lkw_python(script_name: str, *extra: str) -> ProofArgvCommand:
    script = f"{_LKW_SCRIPTS}/{script_name}"
    return ProofArgvCommand(
        executable="uv",
        argv=(
            "run",
            "--project",
            "applications/local_workspace_application",
            "python",
            script,
            *extra,
        ),
    )


def _uv_repo_python(script: str, *extra: str) -> ProofArgvCommand:
    return ProofArgvCommand(
        executable="uv",
        argv=("run", "python", script, *extra),
    )


def build_manifest_entries() -> tuple[ProofManifestEntry, ...]:
    return (
        ProofManifestEntry(
            proof_id="RUNTIME-TOKEN-OPTIMIZATION-OFFLINE",
            title="Token optimization offline smoke proof",
            profiles=frozenset(
                {ProofProfile.QUICK, ProofProfile.FULL, ProofProfile.LIVE}
            ),
            proof_kind="token_optimization_offline_smoke",
            command=_uv_repo_python(
                "scripts/token_optimization/run_universal_proof.py",
                "--config",
                _TOKEN_OFFLINE_CONFIG,
                "--mode",
                "offline_smoke",
            ),
            safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
            timeout_seconds=120,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-MODEL-RUNTIME",
            title="LKW model runtime portability proof",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="model_runtime_portability",
            command=_uv_lkw_python("run-lkw-model-runtime-proof.py"),
            environment_requirements=(
                EnvRequirement(
                    kind=EnvRequirementKind.COMMAND_AVAILABLE, name="uv"
                ),
            ),
            safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
            timeout_seconds=300,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-OS-INTERACTION-WINDOWS",
            title="LKW OS interaction proof (Windows)",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="os_interaction",
            command=_uv_lkw_python(
                "run-lkw-os-interaction-proof.py",
                "--os-family",
                "windows",
            ),
            platform_requirements=frozenset({"windows"}),
            environment_requirements=(
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=900,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-OS-INTERACTION-LINUX",
            title="LKW OS interaction proof (Linux)",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="os_interaction",
            command=_uv_lkw_python(
                "run-lkw-os-interaction-proof.py",
                "--os-family",
                "linux",
            ),
            platform_requirements=frozenset({"linux"}),
            environment_requirements=(
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=900,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-OS-INTERACTION-MACOS",
            title="LKW OS interaction proof (macOS)",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="os_interaction",
            command=_uv_lkw_python(
                "run-lkw-os-interaction-proof.py",
                "--os-family",
                "macos",
            ),
            platform_requirements=frozenset({"macos"}),
            environment_requirements=(
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=900,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-CORE-PLATFORM-WINDOWS",
            title="LKW core platform proof (Windows)",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="core_platform",
            command=_uv_lkw_python(
                "run-lkw-core-platform-proof.py",
                "--os-family",
                "windows",
                "--wrapper-id",
                "windows_bat",
            ),
            platform_requirements=frozenset({"windows"}),
            environment_requirements=(
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=3600,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-CORE-PLATFORM-LINUX",
            title="LKW core platform proof (Linux)",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="core_platform",
            command=_uv_lkw_python(
                "run-lkw-core-platform-proof.py",
                "--os-family",
                "linux",
                "--wrapper-id",
                "linux_sh",
            ),
            platform_requirements=frozenset({"linux"}),
            environment_requirements=(
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=3600,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-CORE-PLATFORM-MACOS",
            title="LKW core platform proof (macOS)",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="core_platform",
            command=_uv_lkw_python(
                "run-lkw-core-platform-proof.py",
                "--os-family",
                "macos",
                "--wrapper-id",
                "macos_sh",
            ),
            platform_requirements=frozenset({"macos"}),
            environment_requirements=(
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=3600,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-PRODUCT-QUICKSTART-WINDOWS",
            title="LKW product quickstart proof (Windows)",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="product_quickstart",
            command=_uv_lkw_python(
                "run-lkw-product-quickstart.py",
                "--os-family",
                "windows",
                "--wrapper-id",
                "windows_bat",
            ),
            platform_requirements=frozenset({"windows"}),
            environment_requirements=(
                EnvRequirement(
                    kind=EnvRequirementKind.COMMAND_AVAILABLE, name="uv"
                ),
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=3600,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-PRODUCT-QUICKSTART-LINUX",
            title="LKW product quickstart proof (Linux)",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="product_quickstart",
            command=_uv_lkw_python(
                "run-lkw-product-quickstart.py",
                "--os-family",
                "linux",
                "--wrapper-id",
                "linux_sh",
            ),
            platform_requirements=frozenset({"linux"}),
            environment_requirements=(
                EnvRequirement(
                    kind=EnvRequirementKind.COMMAND_AVAILABLE, name="uv"
                ),
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=3600,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-PRODUCT-QUICKSTART-MACOS",
            title="LKW product quickstart proof (macOS)",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="product_quickstart",
            command=_uv_lkw_python(
                "run-lkw-product-quickstart.py",
                "--os-family",
                "macos",
                "--wrapper-id",
                "macos_sh",
            ),
            platform_requirements=frozenset({"macos"}),
            environment_requirements=(
                EnvRequirement(
                    kind=EnvRequirementKind.COMMAND_AVAILABLE, name="uv"
                ),
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=3600,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-BACKGROUND-TASK",
            title="LKW Kafka background task proof",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="platform_background_task",
            command=_uv_lkw_python("run-lkw-background-task-proof.py"),
            environment_requirements=(
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=900,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-HOSTING",
            title="LKW application hosting proof",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="platform_application_hosting",
            command=_uv_lkw_python("run-lkw-hosting-proof.py"),
            environment_requirements=(
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=900,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-FILE-WATCHER",
            title="LKW file watcher end-to-end proof",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="file_watcher_e2e",
            command=_uv_lkw_python("run-lkw-file-watcher-e2e-proof.py"),
            environment_requirements=(
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=1200,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="PLATFORM-WINDOWS-NATIVE-CERT",
            title="Windows native LKW runtime certification",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="windows_native_runtime",
            command=_uv_lkw_python("run-lkw-windows-native-certification.py"),
            platform_requirements=frozenset({"windows"}),
            environment_requirements=(
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=1800,
        ),
        ProofManifestEntry(
            proof_id="PLATFORM-LINUX-CONTAINER-CERT",
            title="Linux Docker container runtime certification",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="linux_docker_runtime",
            command=_uv_lkw_python("run-lkw-linux-container-certification.py"),
            environment_requirements=(
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=2400,
        ),
        ProofManifestEntry(
            proof_id="SLACK-CONVERSATION-LIVE",
            title="Slack conversation channel live proof",
            profiles=frozenset({ProofProfile.LIVE}),
            proof_kind="slack_conversation_channel",
            command=_uv_repo_python(
                "scripts/proof/slack_conversation_channel_live_proof.py"
            ),
            environment_requirements=(
                EnvRequirement(
                    kind=EnvRequirementKind.ENV_PRESENT,
                    name="INTERGRAX_SLACK_APP_TOKEN",
                ),
                EnvRequirement(
                    kind=EnvRequirementKind.ENV_PRESENT,
                    name="INTERGRAX_SLACK_BOT_TOKEN",
                ),
            ),
            external_provider="slack",
            safety_class=ProofSafetyClass.EXTERNAL_READ_ONLY,
            timeout_seconds=300,
        ),
        ProofManifestEntry(
            proof_id="SLACK-ASK-PREFLIGHT",
            title="Slack Ask workflow configuration preflight",
            profiles=frozenset({ProofProfile.LIVE}),
            proof_kind="slack_ask_configuration",
            command=_uv_lkw_python("run-lkw-slack-ask-configuration-preflight.py"),
            environment_requirements=(
                EnvRequirement(
                    kind=EnvRequirementKind.ENV_PRESENT,
                    name="INTERGRAX_SLACK_APP_TOKEN",
                ),
                EnvRequirement(
                    kind=EnvRequirementKind.ENV_PRESENT,
                    name="INTERGRAX_SLACK_BOT_TOKEN",
                ),
                EnvRequirement(
                    kind=EnvRequirementKind.ENV_PRESENT,
                    name="LOCAL_WORKSPACE_SLACK_COMPANION_ENABLED",
                ),
            ),
            external_provider="slack",
            safety_class=ProofSafetyClass.EXTERNAL_READ_ONLY,
            timeout_seconds=60,
        ),
        ProofManifestEntry(
            proof_id="SLACK-ASK-WORKFLOW",
            title="Slack Ask workflow proof checklist",
            profiles=frozenset({ProofProfile.LIVE}),
            proof_kind="slack_ask_workflow",
            command=_uv_lkw_python("run-lkw-slack-ask-workflow-proof.py"),
            environment_requirements=(
                EnvRequirement(
                    kind=EnvRequirementKind.ENV_PRESENT,
                    name="INTERGRAX_SLACK_APP_TOKEN",
                ),
                EnvRequirement(
                    kind=EnvRequirementKind.ENV_PRESENT,
                    name="INTERGRAX_SLACK_BOT_TOKEN",
                ),
            ),
            external_provider="slack",
            safety_class=ProofSafetyClass.EXTERNAL_READ_ONLY,
            timeout_seconds=60,
        ),
        ProofManifestEntry(
            proof_id="LKW-MANAGED-WORKSPACE-LIVE",
            title="LKW managed workspace folder sync live proof",
            profiles=frozenset({ProofProfile.LIVE}),
            proof_kind="managed_workspace_folder_sync",
            command=_uv_lkw_python("run-lkw-managed-workspace-live-proof.py"),
            environment_requirements=(
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=1800,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-ASK-WORKSPACE-LIVE",
            title="LKW Ask workspace Qdrant durability live proof",
            profiles=frozenset({ProofProfile.LIVE}),
            proof_kind="ask_workspace_durability",
            command=_uv_lkw_python("run-lkw-ask-workspace-live-proof.py"),
            environment_requirements=(
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=1800,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-WEB-URL-INDEXED-ASK",
            title="LKW WEB_URL indexed Hybrid Ask proof",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="web_url_indexed_ask",
            command=_uv_lkw_python("run-lkw-web-url-indexed-ask-proof.py"),
            environment_requirements=(
                EnvRequirement(
                    kind=EnvRequirementKind.COMMAND_AVAILABLE, name="uv"
                ),
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=3600,
            public_evidence_eligible=True,
        ),
        ProofManifestEntry(
            proof_id="LKW-HYBRID-ASK-INDEXED",
            title="LKW indexed Hybrid Ask branch proof",
            profiles=frozenset({ProofProfile.FULL, ProofProfile.LIVE}),
            proof_kind="hybrid_ask_indexed",
            command=_uv_lkw_python("run-lkw-hybrid-ask-indexed-proof.py"),
            environment_requirements=(
                EnvRequirement(
                    kind=EnvRequirementKind.COMMAND_AVAILABLE, name="uv"
                ),
                EnvRequirement(kind=EnvRequirementKind.DOCKER_AVAILABLE, name="docker"),
            ),
            safety_class=ProofSafetyClass.LOCAL_MUTATING,
            timeout_seconds=3600,
            public_evidence_eligible=True,
        ),
    )


class ManifestLoadError(RuntimeError):
    """Hard failure loading or validating the canonical manifest."""


def _validate_entry_paths(repo_root: Path, entry: ProofManifestEntry) -> None:
    for token in entry.command.argv:
        if token.endswith((".py", ".toml")):
            path = repo_root / token.replace("\\", "/")
            if not path.is_file():
                raise ManifestLoadError(
                    f"{entry.proof_id}: missing declared executable {token}"
                )
    if entry.command.executable not in {"uv", "python"}:
        if shutil.which(entry.command.executable) is None:
            raise ManifestLoadError(
                f"{entry.proof_id}: missing declared executable {entry.command.executable}"
            )


def load_manifest(*, repo_root: Path | None = None) -> IntergraxProofManifest:
    root = repo_root or _REPO_ROOT
    try:
        discovered = discover_platform_proof_descriptors(repo_root=root)
        merged_entries = merge_static_and_discovered_entries(
            build_manifest_entries(),
            discovered,
            repo_root=root,
        )
    except PlatformProofDiscoveryError as exc:
        raise ManifestLoadError(str(exc)) from exc

    try:
        manifest = IntergraxProofManifest(entries=merged_entries)
    except ValidationError as exc:
        raise ManifestLoadError(f"invalid manifest: {exc}") from exc

    for entry in manifest.entries:
        _validate_entry_paths(root, entry)
    return manifest


def expanded_profiles(profile: ProofProfile) -> frozenset[ProofProfile]:
    if profile == ProofProfile.LIVE:
        return frozenset({ProofProfile.QUICK, ProofProfile.FULL, ProofProfile.LIVE})
    if profile == ProofProfile.FULL:
        return frozenset({ProofProfile.QUICK, ProofProfile.FULL})
    return frozenset({ProofProfile.QUICK})
