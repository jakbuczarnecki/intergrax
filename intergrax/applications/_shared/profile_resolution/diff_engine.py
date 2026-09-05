# © Artur Czarnecki. All rights reserved.

"""Domain-aware effective profile semantic diff (P1.2)."""

from __future__ import annotations

from typing import Protocol

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import CostProfile
from intergrax.applications.contracts.profile_resolution.diff import (
    EffectiveProfileDiff,
    ProfileDiffChangeKind,
    ProfileDiffEntry,
    ProfileDiffProvenanceRef,
)
from intergrax.applications.contracts.profile_resolution.resolution import ProfileResolution
from intergrax.applications.contracts.profile_resolution.revision import EffectiveProfileRevision
from intergrax.tools.registry.factory import enabled_tool_ids_for_profile


_COST_LIMIT_PATHS: tuple[tuple[str, str], ...] = (
    ("governance.cost.max_total_tokens", "max_total_tokens"),
    ("governance.cost.max_llm_calls", "max_llm_calls"),
    ("governance.cost.max_tool_calls", "max_tool_calls"),
    ("governance.cost.max_planner_iterations", "max_planner_iterations"),
)


class ProfileFieldDiffer(Protocol):
    """Bounded extension seam for additional semantic diff domains."""

    def diff(
        self,
        *,
        before: ApplicationEnvironmentProfile,
        after: ApplicationEnvironmentProfile,
        before_resolution: ProfileResolution,
        after_resolution: ProfileResolution,
    ) -> tuple[ProfileDiffEntry, ...]:
        """Return semantic diff entries owned by this differ."""


def _encode_scalar(value: object | None) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return str(enum_value)
    return str(value)


def _decision_index_for_path(
    resolution: ProfileResolution,
    path: str,
) -> tuple[ProfileDiffProvenanceRef, ...]:
    refs: list[ProfileDiffProvenanceRef] = []
    for index, decision in enumerate(resolution.decisions):
        if decision.path == path or decision.path.startswith(f"{path}.") or path.startswith(
            f"{decision.path}."
        ):
            refs.append(ProfileDiffProvenanceRef(path=decision.path, decision_index=index))
    return tuple(refs)


def _diff_execution_mode(
    *,
    before: ApplicationEnvironmentProfile,
    after: ApplicationEnvironmentProfile,
    before_resolution: ProfileResolution,
    after_resolution: ProfileResolution,
) -> ProfileDiffEntry | None:
    path = "meta.execution_mode"
    previous = before.execution_mode
    current = after.execution_mode
    if previous == current:
        return None
    return ProfileDiffEntry(
        path=path,
        before=_encode_scalar(previous),
        after=_encode_scalar(current),
        change_kind=ProfileDiffChangeKind.CHANGED,
        provenance=_decision_index_for_path(after_resolution, path)
        or _decision_index_for_path(before_resolution, path),
    )


def _diff_llm_provider(
    *,
    before: ApplicationEnvironmentProfile,
    after: ApplicationEnvironmentProfile,
    before_resolution: ProfileResolution,
    after_resolution: ProfileResolution,
) -> ProfileDiffEntry | None:
    path = "capabilities.llm.provider"
    previous = before.llm_profile.provider if before.llm_profile is not None else None
    current = after.llm_profile.provider if after.llm_profile is not None else None
    if previous == current:
        return None
    kind = ProfileDiffChangeKind.ADDED if previous is None else (
        ProfileDiffChangeKind.REMOVED if current is None else ProfileDiffChangeKind.CHANGED
    )
    return ProfileDiffEntry(
        path=path,
        before=_encode_scalar(previous),
        after=_encode_scalar(current),
        change_kind=kind,
        provenance=_decision_index_for_path(after_resolution, "capabilities.llm")
        or _decision_index_for_path(before_resolution, "capabilities.llm"),
    )


def _diff_llm_model(
    *,
    before: ApplicationEnvironmentProfile,
    after: ApplicationEnvironmentProfile,
    before_resolution: ProfileResolution,
    after_resolution: ProfileResolution,
) -> ProfileDiffEntry | None:
    path = "capabilities.llm.model"
    previous = before.llm_profile.model if before.llm_profile is not None else None
    current = after.llm_profile.model if after.llm_profile is not None else None
    if previous == current:
        return None
    kind = ProfileDiffChangeKind.ADDED if previous is None else (
        ProfileDiffChangeKind.REMOVED if current is None else ProfileDiffChangeKind.CHANGED
    )
    return ProfileDiffEntry(
        path=path,
        before=_encode_scalar(previous),
        after=_encode_scalar(current),
        change_kind=kind,
        provenance=_decision_index_for_path(after_resolution, "capabilities.llm")
        or _decision_index_for_path(before_resolution, "capabilities.llm"),
    )


def _effective_tool_ids(profile: ApplicationEnvironmentProfile) -> frozenset[str]:
    return frozenset(enabled_tool_ids_for_profile(profile.tool_profile))


def _diff_tools(
    *,
    before: ApplicationEnvironmentProfile,
    after: ApplicationEnvironmentProfile,
    before_resolution: ProfileResolution,
    after_resolution: ProfileResolution,
) -> tuple[ProfileDiffEntry, ...]:
    path = "capabilities.tools"
    previous_ids = _effective_tool_ids(before)
    current_ids = _effective_tool_ids(after)
    if previous_ids == current_ids:
        return ()
    entries: list[ProfileDiffEntry] = []
    provenance = _decision_index_for_path(after_resolution, path) or _decision_index_for_path(
        before_resolution,
        path,
    )
    for tool_id in sorted(current_ids - previous_ids):
        entries.append(
            ProfileDiffEntry(
                path=f"{path}.{tool_id}",
                before=None,
                after=tool_id,
                change_kind=ProfileDiffChangeKind.ADDED,
                provenance=provenance,
            )
        )
    for tool_id in sorted(previous_ids - current_ids):
        entries.append(
            ProfileDiffEntry(
                path=f"{path}.{tool_id}",
                before=tool_id,
                after=None,
                change_kind=ProfileDiffChangeKind.REMOVED,
                provenance=provenance,
            )
        )
    return tuple(entries)


def _diff_cost_limit(
    *,
    path: str,
    field_name: str,
    before_cost: CostProfile,
    after_cost: CostProfile,
    before_resolution: ProfileResolution,
    after_resolution: ProfileResolution,
) -> ProfileDiffEntry | None:
    previous = getattr(before_cost, field_name)
    current = getattr(after_cost, field_name)
    if previous == current:
        return None
    if previous is None and current is not None:
        kind = ProfileDiffChangeKind.ADDED
    elif previous is not None and current is None:
        kind = ProfileDiffChangeKind.REMOVED
    elif previous is not None and current is not None and current < previous:
        kind = ProfileDiffChangeKind.NARROWED
    elif previous is not None and current is not None and current > previous:
        kind = ProfileDiffChangeKind.WIDENED
    else:
        kind = ProfileDiffChangeKind.CHANGED
    return ProfileDiffEntry(
        path=path,
        before=_encode_scalar(previous),
        after=_encode_scalar(current),
        change_kind=kind,
        provenance=_decision_index_for_path(after_resolution, "governance.cost")
        or _decision_index_for_path(before_resolution, "governance.cost"),
    )


def _diff_cost(
    *,
    before: ApplicationEnvironmentProfile,
    after: ApplicationEnvironmentProfile,
    before_resolution: ProfileResolution,
    after_resolution: ProfileResolution,
) -> tuple[ProfileDiffEntry, ...]:
    entries: list[ProfileDiffEntry] = []
    for path, field_name in _COST_LIMIT_PATHS:
        entry = _diff_cost_limit(
            path=path,
            field_name=field_name,
            before_cost=before.cost_profile,
            after_cost=after.cost_profile,
            before_resolution=before_resolution,
            after_resolution=after_resolution,
        )
        if entry is not None:
            entries.append(entry)
    return tuple(entries)


def _default_field_differs() -> tuple[ProfileFieldDiffer, ...]:
    return (
        _ExecutionModeFieldDiffer(),
        _LLMFieldDiffer(),
        _ToolsFieldDiffer(),
        _CostFieldDiffer(),
    )


class _ExecutionModeFieldDiffer:
    def diff(
        self,
        *,
        before: ApplicationEnvironmentProfile,
        after: ApplicationEnvironmentProfile,
        before_resolution: ProfileResolution,
        after_resolution: ProfileResolution,
    ) -> tuple[ProfileDiffEntry, ...]:
        entry = _diff_execution_mode(
            before=before,
            after=after,
            before_resolution=before_resolution,
            after_resolution=after_resolution,
        )
        return (entry,) if entry is not None else ()


class _LLMFieldDiffer:
    def diff(
        self,
        *,
        before: ApplicationEnvironmentProfile,
        after: ApplicationEnvironmentProfile,
        before_resolution: ProfileResolution,
        after_resolution: ProfileResolution,
    ) -> tuple[ProfileDiffEntry, ...]:
        entries = [
            _diff_llm_provider(
                before=before,
                after=after,
                before_resolution=before_resolution,
                after_resolution=after_resolution,
            ),
            _diff_llm_model(
                before=before,
                after=after,
                before_resolution=before_resolution,
                after_resolution=after_resolution,
            ),
        ]
        return tuple(entry for entry in entries if entry is not None)


class _ToolsFieldDiffer:
    def diff(
        self,
        *,
        before: ApplicationEnvironmentProfile,
        after: ApplicationEnvironmentProfile,
        before_resolution: ProfileResolution,
        after_resolution: ProfileResolution,
    ) -> tuple[ProfileDiffEntry, ...]:
        return _diff_tools(
            before=before,
            after=after,
            before_resolution=before_resolution,
            after_resolution=after_resolution,
        )


class _CostFieldDiffer:
    def diff(
        self,
        *,
        before: ApplicationEnvironmentProfile,
        after: ApplicationEnvironmentProfile,
        before_resolution: ProfileResolution,
        after_resolution: ProfileResolution,
    ) -> tuple[ProfileDiffEntry, ...]:
        return _diff_cost(
            before=before,
            after=after,
            before_resolution=before_resolution,
            after_resolution=after_resolution,
        )


def diff_effective_profile_revisions(
    before: EffectiveProfileRevision,
    after: EffectiveProfileRevision,
    *,
    field_differs: tuple[ProfileFieldDiffer, ...] | None = None,
) -> EffectiveProfileDiff:
    """Compare two admitted effective revisions using domain-aware semantics."""
    differs = field_differs or _default_field_differs()
    entries: list[ProfileDiffEntry] = []
    for differ in differs:
        entries.extend(
            differ.diff(
                before=before.effective_profile,
                after=after.effective_profile,
                before_resolution=before.resolution,
                after_resolution=after.resolution,
            )
        )
    entries.sort(key=lambda item: item.path)
    return EffectiveProfileDiff(
        from_revision_id=before.revision_id,
        to_revision_id=after.revision_id,
        from_fingerprint=before.fingerprint,
        to_fingerprint=after.fingerprint,
        entries=tuple(entries),
    )
