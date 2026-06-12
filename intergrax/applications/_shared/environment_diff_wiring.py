# © Artur Czarnecki. All rights reserved.

"""Build ApplicationEnvironmentDiff from manifests and snapshots (APP-EVOL-6)."""

from __future__ import annotations

from typing import Any

from intergrax.applications._shared.environment_snapshot_wiring import capture_environment_snapshot
from intergrax.applications.contracts.application_environment_diff import (
    ApplicationEnvironmentDiff,
    DiffRiskLevel,
    FieldChange,
    RosterChangeKind,
    RosterEntryChange,
    StructuredDiff,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_snapshot import EnvironmentSnapshot
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.runtime.registry.semver_compat import SemVer


def _collect_field_changes(
    path: str,
    left: Any,
    right: Any,
    *,
    changes: list[FieldChange],
) -> None:
    if left == right:
        return
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        for key in keys:
            child_path = f"{path}.{key}" if path else str(key)
            _collect_field_changes(child_path, left.get(key), right.get(key), changes=changes)
        return
    if isinstance(left, list) and isinstance(right, list):
        if left != right:
            changes.append(FieldChange(path=path, left=left, right=right))
        return
    changes.append(FieldChange(path=path, left=left, right=right))


def diff_structured(left: dict[str, Any], right: dict[str, Any]) -> StructuredDiff:
    """Return a field-level diff between two JSON-like documents."""
    changes: list[FieldChange] = []
    _collect_field_changes("", left, right, changes=changes)
    return StructuredDiff(changes=changes)


def diff_profile(
    left: ApplicationEnvironmentProfile,
    right: ApplicationEnvironmentProfile,
) -> StructuredDiff:
    """Diff two environment profiles."""
    return diff_structured(
        left.model_dump(mode="json"),
        right.model_dump(mode="json"),
    )


def diff_graph(
    left: ApplicationEnvironmentProfile,
    right: ApplicationEnvironmentProfile,
) -> StructuredDiff | None:
    """Diff graph specs when either side declares one."""
    if left.graph_spec is None and right.graph_spec is None:
        return None
    left_payload = (
        left.graph_spec.model_dump(mode="json") if left.graph_spec is not None else {}
    )
    right_payload = (
        right.graph_spec.model_dump(mode="json") if right.graph_spec is not None else {}
    )
    return diff_structured(left_payload, right_payload)


def diff_envelope(
    left: ApplicationEnvironmentProfile,
    right: ApplicationEnvironmentProfile,
) -> StructuredDiff | None:
    """Diff organizational envelopes when either side declares one."""
    if left.organizational_policy is None and right.organizational_policy is None:
        return None
    left_payload = (
        left.organizational_policy.model_dump(mode="json")
        if left.organizational_policy is not None
        else {}
    )
    right_payload = (
        right.organizational_policy.model_dump(mode="json")
        if right.organizational_policy is not None
        else {}
    )
    return diff_structured(left_payload, right_payload)


def _binding_key(binding: AgentBinding) -> str:
    if binding.contract_id:
        return binding.contract_id.strip()
    if binding.import_path:
        return binding.import_path.rsplit(".", 1)[-1]
    return binding.resolved_agent_type().__name__


def diff_roster(
    left: list[AgentBinding],
    right: list[AgentBinding],
) -> list[RosterEntryChange]:
    """Diff enabled agent bindings between two manifests."""
    left_map = {_binding_key(binding): binding for binding in left if binding.enabled}
    right_map = {_binding_key(binding): binding for binding in right if binding.enabled}
    changes: list[RosterEntryChange] = []

    for key in sorted(set(left_map) - set(right_map)):
        binding = left_map[key]
        changes.append(
            RosterEntryChange(
                agent_key=key,
                kind=RosterChangeKind.REMOVED,
                left_capabilities=sorted(binding.capabilities),
            ),
        )
    for key in sorted(set(right_map) - set(left_map)):
        binding = right_map[key]
        changes.append(
            RosterEntryChange(
                agent_key=key,
                kind=RosterChangeKind.ADDED,
                right_capabilities=sorted(binding.capabilities),
            ),
        )
    for key in sorted(set(left_map) & set(right_map)):
        left_caps = sorted(left_map[key].capabilities)
        right_caps = sorted(right_map[key].capabilities)
        if left_caps != right_caps:
            changes.append(
                RosterEntryChange(
                    agent_key=key,
                    kind=RosterChangeKind.CAPABILITIES_CHANGED,
                    left_capabilities=left_caps,
                    right_capabilities=right_caps,
                ),
            )
    return changes


def assess_diff_risk(
    *,
    left_manifest: ApplicationManifest,
    right_manifest: ApplicationManifest,
    left_env: ApplicationEnvironmentProfile,
    right_env: ApplicationEnvironmentProfile,
    profile_diff: StructuredDiff,
    roster_diff: list[RosterEntryChange],
) -> tuple[DiffRiskLevel, list[str]]:
    """Classify deploy risk and enumerate breaking changes."""
    breaking: list[str] = []
    risk = DiffRiskLevel.LOW

    if left_env.execution_mode != right_env.execution_mode:
        breaking.append(
            f"execution_mode changed: {left_env.execution_mode.value} -> {right_env.execution_mode.value}",
        )
        risk = DiffRiskLevel.HIGH

    if left_manifest.version != right_manifest.version:
        breaking.append(
            f"app_version changed: {left_manifest.version} -> {right_manifest.version}",
        )
        try:
            left_ver = SemVer.parse(left_manifest.version)
            right_ver = SemVer.parse(right_manifest.version)
            if right_ver.major > left_ver.major:
                risk = DiffRiskLevel.HIGH
            elif risk is DiffRiskLevel.LOW:
                risk = DiffRiskLevel.MEDIUM
        except ValueError:
            if risk is DiffRiskLevel.LOW:
                risk = DiffRiskLevel.MEDIUM

    for entry in roster_diff:
        if entry.kind is RosterChangeKind.REMOVED:
            breaking.append(f"roster agent removed: {entry.agent_key}")
            risk = DiffRiskLevel.HIGH
        elif entry.kind is RosterChangeKind.CAPABILITIES_CHANGED:
            breaking.append(f"roster capabilities changed: {entry.agent_key}")
            if risk is DiffRiskLevel.LOW:
                risk = DiffRiskLevel.MEDIUM

    if profile_diff.changed and risk is DiffRiskLevel.LOW:
        risk = DiffRiskLevel.MEDIUM

    if right_env.execution_mode is ExecutionMode.STRICT and breaking:
        risk = DiffRiskLevel.HIGH

    return risk, breaking


def build_application_environment_diff(
    left_manifest: ApplicationManifest,
    left_env: ApplicationEnvironmentProfile,
    right_manifest: ApplicationManifest,
    right_env: ApplicationEnvironmentProfile,
    *,
    left_snapshot: EnvironmentSnapshot | None = None,
    right_snapshot: EnvironmentSnapshot | None = None,
) -> ApplicationEnvironmentDiff:
    """Build a full environment diff between two manifest/environment pairs."""
    left_snap = left_snapshot or capture_environment_snapshot(left_manifest, left_env)
    right_snap = right_snapshot or capture_environment_snapshot(right_manifest, right_env)

    profile_diff = diff_profile(left_env, right_env)
    graph_diff = diff_graph(left_env, right_env)
    envelope_diff = diff_envelope(left_env, right_env)
    roster_diff = diff_roster(left_manifest.enabled_agents(), right_manifest.enabled_agents())
    risk_level, breaking_changes = assess_diff_risk(
        left_manifest=left_manifest,
        right_manifest=right_manifest,
        left_env=left_env,
        right_env=right_env,
        profile_diff=profile_diff,
        roster_diff=roster_diff,
    )

    return ApplicationEnvironmentDiff(
        left_snapshot_id=left_snap.snapshot_id,
        right_snapshot_id=right_snap.snapshot_id,
        left_app_version=left_manifest.version,
        right_app_version=right_manifest.version,
        profile_diff=profile_diff,
        graph_diff=graph_diff,
        envelope_diff=envelope_diff,
        roster_diff=roster_diff,
        risk_level=risk_level,
        breaking_changes=breaking_changes,
    )


def format_application_environment_diff(diff: ApplicationEnvironmentDiff) -> str:
    """Human-readable summary for CLI output."""
    lines = [
        f"left:  {diff.left_app_version} ({diff.left_snapshot_id})",
        f"right: {diff.right_app_version} ({diff.right_snapshot_id})",
        f"risk:  {diff.risk_level.value}",
        f"profile changes: {len(diff.profile_diff.changes)}",
    ]
    if diff.graph_diff is not None:
        lines.append(f"graph changes: {len(diff.graph_diff.changes)}")
    if diff.envelope_diff is not None:
        lines.append(f"envelope changes: {len(diff.envelope_diff.changes)}")
    lines.append(f"roster changes: {len(diff.roster_diff)}")
    if diff.breaking_changes:
        lines.append("breaking:")
        lines.extend(f"  - {item}" for item in diff.breaking_changes)
    return "\n".join(lines)


def roster_digest_changed(left: ApplicationManifest, right: ApplicationManifest) -> bool:
    """Return whether roster digests differ."""
    from intergrax.applications._shared.environment_snapshot_wiring import compute_roster_digest

    return compute_roster_digest(left) != compute_roster_digest(right)


def profile_digest_changed(
    left: ApplicationEnvironmentProfile,
    right: ApplicationEnvironmentProfile,
) -> bool:
    """Return whether profile snapshot ids differ."""
    from intergrax.applications._shared.environment_snapshot_wiring import compute_profile_snapshot_id

    return compute_profile_snapshot_id(left) != compute_profile_snapshot_id(right)
